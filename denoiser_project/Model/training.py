import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate
from pathlib import Path
from util import setup_logger, load_config
from denoiser_project.Data.ingestion import process_files
from denoiser_project.Data.transformation import load_audio_files

LOG = setup_logger()
CONFIG = load_config("../../config.yaml")


def create_directories(*dirs):
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_and_process_data(file_list, src_dir, dest_dir, sample_rate, target_duration):
    process_files(file_list, src_dir, dest_dir)
    audio_files = load_audio_files(dest_dir, sr=sample_rate, target_duration=target_duration)
    audio_data = np.array(audio_files)
    audio_data = np.expand_dims(audio_data, axis=-1)
    return audio_data


def build_unet_model(input_shape):
    inputs = Input(shape=input_shape)
    activation = CONFIG['model']['activation']
    # Encoder
    conv1 = Conv1D(64, 3, activation=activation, padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(128, 3, activation=activation, padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(256, 3, activation=activation, padding='same')(pool2)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    # Middle
    conv4 = Conv1D(512, 3, activation=activation, padding='same')(pool3)

    # Decoder
    up5 = UpSampling1D(size=2)(conv4)
    concat5 = concatenate([conv3, up5], axis=-1)
    conv5 = Conv1D(256, 3, activation=activation, padding='same')(concat5)

    up6 = UpSampling1D(size=2)(conv5)
    concat6 = concatenate([conv2, up6], axis=-1)
    conv6 = Conv1D(128, 3, activation=activation, padding='same')(concat6)

    up7 = UpSampling1D(size=2)(conv6)
    concat7 = concatenate([conv1, up7], axis=-1)
    conv7 = Conv1D(64, 3, activation=activation, padding='same')(concat7)

    # Output
    output = Conv1D(1, 3, activation='linear', padding='same')(conv7)

    model = Model(inputs=inputs, outputs=output)
    return model


def build_denoising_autoencoder_model(input_shape):
    input_signal = Input(shape=input_shape)

    # Encoder
    x = Conv1D(16, 3, activation=CONFIG['model']['activation'], padding='same')(input_signal)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(8, 3, activation=CONFIG['model']['activation'], padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)

    # Decoder
    x = Conv1D(8, 3, activation=CONFIG['model']['activation'], padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation=CONFIG['model']['activation'], padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation=CONFIG['model']['output_activation'], padding='same')(x)

    autoencoder = Model(input_signal, decoded)
    return autoencoder


def main():
    if CONFIG is None:
        LOG.error("Config file not provided.")
        return

    noisy_data_path = Path(CONFIG['data']['noisy_data_path'])
    clean_data_path = Path(CONFIG['data']['clean_data_path'])
    noisy_chunk_path = Path(CONFIG['data']['noisy_chunk_path'])
    clean_chunk_path = Path(CONFIG['data']['clean_chunk_path'])
    sample_rate = CONFIG['data']['sample_rate']
    data_duration = CONFIG['data']['data_duration']
    optimizer = CONFIG['model']['optimizer']
    loss = CONFIG['model']['loss']
    batch_size = CONFIG['training']['batch_size']
    epochs = CONFIG['training']['epochs']
    validation_split = CONFIG['training']['validation_split']

    # Create necessary directories
    create_directories('../Chunks', noisy_chunk_path, clean_chunk_path)

    # Process audio files
    noisy_data = load_and_process_data(
        list(noisy_data_path.iterdir())[:1000],
        noisy_data_path,
        noisy_chunk_path,
        sample_rate,
        data_duration
    )

    clean_data = load_and_process_data(
        list(clean_data_path.iterdir())[:1000],
        clean_data_path,
        clean_chunk_path,
        sample_rate,
        data_duration
    )

    # Define input shape based on clean data
    input_shape = clean_data.shape[1:]

    # Build and compile U-Net model
    model = build_unet_model(input_shape)
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    # Train and save U-Net model
    model.fit(noisy_data, clean_data, batch_size=batch_size, epochs=epochs,
              validation_split=validation_split)
    model.save("model.keras")

    # Build and compile Denoising Autoencoder model
    autoencoder = build_denoising_autoencoder_model(input_shape)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.summary()

    # Train and save Denoising Autoencoder model
    autoencoder.fit(noisy_data, clean_data, batch_size=batch_size,
                    epochs=epochs, validation_split=validation_split)
    autoencoder.save("autoencoder.keras")

    LOG.info("Processing complete!")


if __name__ == '__main__':
    main()
