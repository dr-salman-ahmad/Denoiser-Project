import os
import numpy as np
import yaml
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate

from Data.ingestion import process_files
from Data.transformation import load_audio_files


def create_directories(*dirs):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def load_and_process_data(file_list, src_dir, dest_dir, sample_rate, target_duration):
    process_files(file_list, src_dir, dest_dir)
    audio_files = load_audio_files(dest_dir, sr=sample_rate, target_duration=target_duration)
    audio_data = np.array(audio_files)
    audio_data = np.expand_dims(audio_data, axis=-1)
    return audio_data


def build_unet_model(input_shape, config):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv1D(64, 3, activation=config['model']['activation'], padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(128, 3, activation=config['model']['activation'], padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(256, 3, activation=config['model']['activation'], padding='same')(pool2)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    # Middle
    conv4 = Conv1D(512, 3, activation=config['model']['activation'], padding='same')(pool3)

    # Decoder
    up5 = UpSampling1D(size=2)(conv4)
    concat5 = concatenate([conv3, up5], axis=-1)
    conv5 = Conv1D(256, 3, activation=config['model']['activation'], padding='same')(concat5)

    up6 = UpSampling1D(size=2)(conv5)
    concat6 = concatenate([conv2, up6], axis=-1)
    conv6 = Conv1D(128, 3, activation=config['model']['activation'], padding='same')(concat6)

    up7 = UpSampling1D(size=2)(conv6)
    concat7 = concatenate([conv1, up7], axis=-1)
    conv7 = Conv1D(64, 3, activation=config['model']['activation'], padding='same')(concat7)

    # Output
    output = Conv1D(1, 3, activation='linear', padding='same')(conv7)

    model = Model(inputs=inputs, outputs=output)
    return model


def build_denoising_autoencoder_model(input_shape, config):
    input_signal = Input(shape=input_shape)

    # Encoder
    x = Conv1D(16, 3, activation=config['model']['activation'], padding='same')(input_signal)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(8, 3, activation=config['model']['activation'], padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)

    # Decoder
    x = Conv1D(8, 3, activation=config['model']['activation'], padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation=config['model']['activation'], padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation=config['model']['output_activation'], padding='same')(x)

    autoencoder = Model(input_signal, decoded)
    return autoencoder


def main():
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create necessary directories
    create_directories('../Chunks', config['data']['noisy_chunk_path'], config['data']['clean_chunk_path'])

    # Process audio files
    noisy_data = load_and_process_data(
        os.listdir(config['data']['noisy_data_path'])[:1000],
        config['data']['noisy_data_path'],
        config['data']['noisy_chunk_path'],
        config['data']['sample_rate'],
        config['data']['data_duration']
    )

    clean_data = load_and_process_data(
        os.listdir(config['data']['clean_data_path'])[:1000],
        config['data']['clean_data_path'],
        config['data']['clean_chunk_path'],
        config['data']['sample_rate'],
        config['data']['data_duration']
    )

    # Define input shape based on clean data
    input_shape = clean_data.shape[1:]

    # Build and compile U-Net model
    model = build_unet_model(input_shape, config)
    model.compile(optimizer=config['model']['optimizer'], loss=config['model']['loss'])
    model.summary()

    # Train and save U-Net model
    model.fit(noisy_data, clean_data, batch_size=config['training']['batch_size'], epochs=config['training']['epochs'],
              validation_split=config['training']['validation_split'])
    model.save("model.keras")

    # Build and compile Denoising Autoencoder model
    autoencoder = build_denoising_autoencoder_model(input_shape, config)
    autoencoder.compile(optimizer=config['model']['optimizer'], loss=config['model']['loss'])
    autoencoder.summary()

    # Train and save Denoising Autoencoder model
    autoencoder.fit(noisy_data, clean_data, batch_size=config['training']['batch_size'],
                    epochs=config['training']['epochs'], validation_split=config['training']['validation_split'])
    autoencoder.save("autoencoder.keras")

    print("Processing complete!")


if __name__ == '__main__':
    main()
