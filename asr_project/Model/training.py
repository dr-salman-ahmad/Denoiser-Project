import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer, cer
import yaml

from asr_project.Data.ingestion import load_data
from asr_project.Data.transformation import encode_single_sample


def plot_spectogram(train_dataset, num_to_char, wavs_path, df_train):
    fig = plt.figure(figsize=(8, 5))
    for batch in train_dataset.take(1):
        spectrogram = batch[0][0].numpy()
        spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
        label = batch[1][0]
        # Spectrogram
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        ax = plt.subplot(2, 1, 1)
        ax.imshow(spectrogram, vmax=1)
        ax.set_title(label)
        ax.axis("off")
        # Wav
        file = tf.io.read_file(wavs_path + list(df_train["file_name"])[0] + ".wav")
        audio, _ = tf.audio.decode_wav(file)
        audio = audio.numpy()
        ax = plt.subplot(2, 1, 2)
        plt.plot(audio)
        ax.set_title("Signal Wave")
        ax.set_xlim(0, len(audio))
        display.display(display.Audio(np.transpose(audio), rate=16000))
    plt.show()


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


# A utility function to decode the output of the network
def decode_batch_predictions(pred, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset, num_to_char, model):
        super().__init__()
        self.dataset = dataset
        self.num_to_char = num_to_char
        self.model = model

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions, self.num_to_char)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(self.num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        cer_score = cer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f} and CER: {cer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)


def main():
    with open('../../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    data_path = config['data']['asr_data_path']
    wave_path = data_path + "/wavs/"
    df_train, df_val = load_data(data_path, config)
    # The set of characters accepted in the transcription.
    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    # Mapping characters to integers
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    print(
        f"The vocabulary is: {char_to_num.get_vocabulary()} "
        f"(size ={char_to_num.vocabulary_size()})"
    )

    # An integer scalar Tensor. The window length in samples.
    frame_length = config['asr_training']['frame_length']
    # An integer scalar Tensor. The number of samples to step.
    frame_step = config['asr_training']['frame_step']
    # An integer scalar Tensor. The size of the FFT to apply.
    # If not provided, uses the smallest power of 2 enclosing frame_length.
    fft_length = config['asr_training']['fft_length']

    batch_size = config['asr_training']['batch_size']
    # Define the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_train["file_name"]), list(df_train["normalized_transcription"]))
    )
    train_dataset = (
        train_dataset.map(
            lambda wav_file, label: encode_single_sample(wav_file, label, wave_path, char_to_num, frame_length,
                                                         frame_step, fft_length), num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Define the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
    )
    validation_dataset = (
        validation_dataset.map(
            lambda wav_file, label: encode_single_sample(wav_file, label, wave_path, char_to_num, frame_length,
                                                         frame_step, fft_length), num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    plot_spectogram(train_dataset, num_to_char, wave_path, df_train)
    # Get the model
    model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=512,
    )
    model.summary(line_length=110)

    # Define the number of epochs.
    epochs = config['asr_training']['epochs']
    # Callback function to check transcription on the val set.
    validation_callback = CallbackEval(validation_dataset, num_to_char, model)
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[validation_callback],
    )

    # Save the model
    model.save("model.keras")

if __name__ == "__main__":
    main()
