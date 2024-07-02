import os
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from tensorflow import keras
from Data.transformation import load_audio_files
import yaml
import numpy as np

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize the model
model = pretrained.dns64().cpu()

# Directories containing noisy and clean audio files
noisy_dir = config['testing']['noisy_data_path']
clean_dir = config['testing']['clean_data_path']

unet_model_path = config['testing']['unet_model_path']
autoencoder_model_path = config['testing']['autoencoder_model_path']

unet_model = keras.models.load_model(unet_model_path)
autoencoder_model = keras.models.load_model(autoencoder_model_path)

# Initialize lists to store results
results = []
unet_model_results = []
autoencoder_model_results = []

# Process each file in the noisy directory
for filename in os.listdir(noisy_dir):
    if filename.endswith('.wav'):
        noisy_path = os.path.join(noisy_dir, filename)
        clean_path = os.path.join(clean_dir, filename)

        # Load the noisy and clean audio
        noisy_wav, sr = torchaudio.load(noisy_path)
        clean_wav, sr = torchaudio.load(clean_path)

        # Convert audio to the model's sample rate and channels
        noisy_wav = convert_audio(noisy_wav.cpu(), sr, model.sample_rate, model.chin)
        clean_wav = convert_audio(clean_wav.cpu(), sr, model.sample_rate, model.chin)

        # Denoise the noisy audio
        with torch.no_grad():
            denoised = model(noisy_wav[None])[0]

        # Convert to numpy arrays
        noisy_np = noisy_wav.squeeze().cpu().numpy()
        clean_np = clean_wav.squeeze().cpu().numpy()
        denoised_np = denoised.squeeze().cpu().numpy()

        # Compute PESQ (using narrowband mode 'nb')
        pesq_score = pesq(model.sample_rate, clean_np, denoised_np, 'nb')

        # Compute STOI
        stoi_score = stoi(clean_np, denoised_np, model.sample_rate, extended=False)

        # Store the results
        results.append({
            'filename': filename,
            'pesq': pesq_score,
            'stoi': stoi_score
        })

        # Save the original and denoised audio files
        # torchaudio.save(f'original_{filename}', noisy_wav.cpu(), model.sample_rate)
        # torchaudio.save(f'denoised_{filename}', denoised.cpu(), model.sample_rate)

# Print the results
for result in results:
    print(f"File: {result['filename']}, PESQ: {result['pesq']}, STOI: {result['stoi']}")

# Calculate average PESQ and STOI scores
average_pesq = sum(result['pesq'] for result in results) / len(results)
average_stoi = sum(result['stoi'] for result in results) / len(results)
print(f'\nAverage PESQ Score: {average_pesq}')
print(f'Average STOI Score: {average_stoi}')

# Load audio files for further processing
noisy_audio_files = load_audio_files(config['testing']['noisy_data_path'], sr=config['data']['sample_rate'], target_duration=config['data']['data_duration'])
clean_audio_files = load_audio_files(config['testing']['clean_data_path'], sr=config['data']['sample_rate'], target_duration=config['data']['data_duration'])

# Convert lists to numpy arrays
clean_data = np.array(clean_audio_files)
noisy_data = np.array(noisy_audio_files)

# Expand dimensions for compatibility with Conv1D input shape
noisy_data = np.expand_dims(noisy_data, axis=-1)

for i in range(len(noisy_data)):
    # Processing with UNet model
    unet_result = np.squeeze(np.transpose(unet_model.predict(np.transpose(noisy_data[i]))), axis=-1).flatten()
    pesq_score_unet = pesq(config['data']['sample_rate'], clean_data[i], unet_result, 'nb')
    stoi_score_unet = stoi(clean_data[i], unet_result, config['data']['sample_rate'], extended=False)
    unet_model_results.append({'pesq': pesq_score_unet, 'stoi': stoi_score_unet})

    # Processing with Autoencoder model
    autoencoder_result = np.squeeze(np.transpose(autoencoder_model.predict(np.transpose(noisy_data[i]))), axis=-1).flatten()
    pesq_score_autoencoder = pesq(config['data']['sample_rate'], clean_data[i], autoencoder_result, 'nb')
    stoi_score_autoencoder = stoi(clean_data[i], autoencoder_result, config['data']['sample_rate'], extended=False)
    autoencoder_model_results.append({'pesq': pesq_score_autoencoder, 'stoi': stoi_score_autoencoder})

# UNet model results
print("\nUNet Model Results:")
for result in unet_model_results:
    print(f"PESQ: {result['pesq']}, STOI: {result['stoi']}")
average_pesq_unet = sum(result['pesq'] for result in unet_model_results) / len(unet_model_results)
average_stoi_unet = sum(result['stoi'] for result in unet_model_results) / len(unet_model_results)
print(f"Average PESQ Score: {average_pesq_unet}")
print(f"Average STOI Score: {average_stoi_unet}")

# Autoencoder model results
print("\nAutoencoder Model Results:")
for result in autoencoder_model_results:
    print(f"PESQ: {result['pesq']}, STOI: {result['stoi']}")
average_pesq_autoencoder = sum(result['pesq'] for result in autoencoder_model_results) / len(autoencoder_model_results)
average_stoi_autoencoder = sum(result['stoi'] for result in autoencoder_model_results) / len(autoencoder_model_results)
print(f"Average PESQ Score: {average_pesq_autoencoder}")
print(f"Average STOI Score: {average_stoi_autoencoder}")