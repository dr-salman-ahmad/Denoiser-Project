import numpy as np
import tensorflow as tf
from tensorflow import keras
from jiwer import wer, cer
from training import decode_batch_predictions, CTCLoss
from asr_project.Data.ingestion import load_data
from asr_project.Data.transformation import encode_single_sample
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import torch


def wave2vec_testing(df_val, config):
    # Load pretrained model and tokenizer (processor)
    model_name = config['wave2vec']['model_name']
    built_in_model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    transcriptions = []
    for i in range(len(df_val)):
        # Example usage: transcribe an audio file
        audio_file = config['data']['asr_data_path'] + "/wavs/" + str(df_val.iloc[i]['file_name']) + ".wav"
        input_audio, _ = librosa.load(audio_file, sr=config['wave2vec']['sample_rate'])  # Adjust sample rate as per your audio

        input_values = processor(input_audio, return_tensors="pt", sampling_rate=config['wave2vec']['sample_rate']).input_values
        with torch.no_grad():
            logits = built_in_model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        transcriptions.append(transcription.lower())

    return transcriptions


def main():
    with open('../../config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data_path = config['data']['asr_data_path']
    wave_path = data_path + "/wavs/"
    df_train, df_val = load_data(data_path, config)

    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    frame_length = config['asr_testing']['frame_length']
    frame_step = config['asr_testing']['frame_step']
    fft_length = config['asr_testing']['fft_length']
    batch_size = config['asr_testing']['batch_size']

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
    )
    test_dataset = (
        test_dataset.map(
            lambda wav_file, label: encode_single_sample(wav_file, label, wave_path, char_to_num, frame_length,
                                                         frame_step, fft_length),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = keras.models.load_model("model.keras", custom_objects={"CTCLoss": CTCLoss})

    predictions = []
    targets = []
    for batch in test_dataset:
        X, y = batch
        batch_predictions = model.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions, num_to_char)
        predictions.extend(batch_predictions)
        for label in y:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            targets.append(label)

    wave2vec_predictions = wave2vec_testing(df_val, config)
    wer_score = wer(targets, predictions)
    cer_score = cer(targets, predictions)

    wave2vec_wer_score = wer(targets, wave2vec_predictions)
    wave2vec_cer_score = wer(targets, wave2vec_predictions)

    print("-" * 100)
    print("My Model Results")
    print(f"Word Error Rate: {wer_score:.4f} and CER: {cer_score:.4f}")
    print("Wave2vec Model Results")
    print(f"Word Error Rate: {wave2vec_wer_score:.4f} and CER: {wave2vec_cer_score:.4f}")
    print("-" * 100)
    for i in range(len(predictions)):
        print(f"Target    : {targets[i]}")
        print(f"Prediction: {predictions[i]}")
        print(f"Wave2vec Prediction: {wave2vec_predictions[i]}")
        print("-" * 100)


if __name__ == "__main__":
    main()
