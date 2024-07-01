import os
import numpy as np
import yaml
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate

from Data.ingestion import process_files
from Data.transformation import load_audio_files

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def unet_model(input_shape):
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


def denoising_autoencoder_model(input_shape):
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


if __name__ == '__main__':
    # Path to the directory containing the noisy and clean audio files
    noisy_dir = config['data']['noisy_data_path']
    clean_dir = config['data']['clean_data_path']

    os.makedirs('../Chunks', exist_ok=True)

    # Directory to save the processed audio files
    x_processed_dir = config['data']['noisy_chunk_path']
    y_processed_dir = config['data']['clean_chunk_path']

    os.makedirs(x_processed_dir, exist_ok=True)
    os.makedirs(y_processed_dir, exist_ok=True)

    all_noisy_files = os.listdir(noisy_dir)
    all_clean_files = os.listdir(clean_dir)

    # Limit to the first 1000 files
    x_files_to_process = all_noisy_files[:1000]
    y_files_to_process = all_clean_files[:1000]

    process_files(x_files_to_process, noisy_dir, x_processed_dir)
    process_files(y_files_to_process, clean_dir, y_processed_dir)

    print("Processing complete!")

    # Load audio files for further processing
    noisy_audio_files = load_audio_files(x_processed_dir, sr=config['data']['sample_rate'], target_duration=config['data']['data_duration'])
    clean_audio_files = load_audio_files(y_processed_dir, sr=config['data']['sample_rate'], target_duration=config['data']['data_duration'])

    # Convert lists to numpy arrays
    clean_data = np.array(clean_audio_files)
    noisy_data = np.array(noisy_audio_files)

    # Expand dimensions for compatibility with Conv1D input shape
    clean_data = np.expand_dims(clean_data, axis=-1)
    noisy_data = np.expand_dims(noisy_data, axis=-1)

    input_shape = clean_data.shape[1:]  # Assuming input shape based on clean_data
    model = unet_model(input_shape)
    model.summary()

    # Compile the model
    model.compile(optimizer=config['model']['optimizer'], loss=config['model']['loss'])

    # Train the model
    model.fit(noisy_data, clean_data, batch_size=config['training']['batch_size'], epochs=config['training']['epochs'], validation_split=config['training']['validation_split'])

    model.save("model.keras")

    # Denoising Auto Encoder
    autoencoder = denoising_autoencoder_model(input_shape)
    autoencoder.summary()

    autoencoder.compile(optimizer=config['model']['optimizer'], loss=config['model']['loss'])
    autoencoder.fit(noisy_data, clean_data, batch_size=config['training']['batch_size'], epochs=config['training']['epochs'], validation_split=config['training']['validation_split'])
    autoencoder.save("autoencoder.keras")
