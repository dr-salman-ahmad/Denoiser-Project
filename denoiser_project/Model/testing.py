import os
import yaml
import torch
import torchaudio
import numpy as np
from pesq import pesq
from pystoi import stoi
from tensorflow import keras
from denoiser import pretrained
from denoiser.dsp import convert_audio
from denoiser_project.Data.transformation import load_audio_files


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def load_audio(path):
    wav, sr = torchaudio.load(path)
    return wav.cpu(), sr


def denoise_audio(model, audio):
    with torch.no_grad():
        denoised = model(audio[None])[0]
    return denoised.squeeze().cpu().numpy()


def calculate_scores(clean_audio, denoised_audio, sample_rate):
    pesq_score = pesq(sample_rate, clean_audio, denoised_audio, 'nb')
    stoi_score = stoi(clean_audio, denoised_audio, sample_rate, extended=False)
    return pesq_score, stoi_score


def calculate_average_scores(results):
    pesq_scores = [result['pesq'] for result in results]
    stoi_scores = [result['stoi'] for result in results]
    average_pesq = sum(pesq_scores) / len(results)
    average_stoi = sum(stoi_scores) / len(results)
    return average_pesq, average_stoi


def process_audio_files(noisy_dir, clean_dir, model):
    results = []
    for filename in os.listdir(noisy_dir):
        if filename.endswith('.wav'):
            try:
                noisy_path = os.path.join(noisy_dir, filename)
                clean_path = os.path.join(clean_dir, filename)

                noisy_wav, sr = load_audio(noisy_path)
                clean_wav, _ = load_audio(clean_path)

                noisy_wav = convert_audio(noisy_wav, sr, model.sample_rate, model.chin)
                clean_wav = convert_audio(clean_wav, sr, model.sample_rate, model.chin)

                denoised_np = denoise_audio(model, noisy_wav)
                pesq_score, stoi_score = calculate_scores(clean_wav.squeeze().numpy(), denoised_np, model.sample_rate)

                results.append({
                    'filename': filename,
                    'pesq': pesq_score,
                    'stoi': stoi_score
                })
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        else:
            print(f"Skipping {filename}")

    return results


def process_models(noisy_data, clean_data, unet_model, autoencoder_model, sample_rate):
    unet_model_results = []
    autoencoder_model_results = []

    for i in range(len(noisy_data)):
        unet_result = np.squeeze(np.transpose(unet_model.predict(np.transpose(noisy_data[i]))), axis=-1).flatten()
        autoencoder_result = np.squeeze(np.transpose(autoencoder_model.predict(np.transpose(noisy_data[i]))),
                                        axis=-1).flatten()

        pesq_unet, stoi_unet = calculate_scores(clean_data[i], unet_result, sample_rate)
        pesq_autoencoder, stoi_autoencoder = calculate_scores(clean_data[i], autoencoder_result, sample_rate)

        unet_model_results.append({'pesq': pesq_unet, 'stoi': stoi_unet})
        autoencoder_model_results.append({'pesq': pesq_autoencoder, 'stoi': stoi_autoencoder})

    return unet_model_results, autoencoder_model_results


def print_results(results, label):
    print(f"\n{label} Results:")
    for result in results:
        print(f"PESQ: {result['pesq']}, STOI: {result['stoi']}")


def print_average_scores(average_pesq, average_stoi):
    print(f"Average PESQ Score: {average_pesq}")
    print(f"Average STOI Score: {average_stoi}")


# Main execution
def main():
    config = load_config('../../config.yaml')

    # Initialize models
    model = pretrained.dns64().cpu()
    unet_model = keras.models.load_model(config['testing']['unet_model_path'])
    autoencoder_model = keras.models.load_model(config['testing']['autoencoder_model_path'])

    # Process noisy and clean audio files
    results = process_audio_files(config['testing']['noisy_data_path'], config['testing']['clean_data_path'], model)
    print_results(results, "Denoiser Model")

    # Calculate average scores
    average_pesq, average_stoi = calculate_average_scores(results)
    print_average_scores(average_pesq, average_stoi)

    # Load additional audio files for further processing
    noisy_audio_files = load_audio_files(config['testing']['noisy_data_path'], sr=config['data']['sample_rate'],
                                         target_duration=config['data']['data_duration'])
    clean_audio_files = load_audio_files(config['testing']['clean_data_path'], sr=config['data']['sample_rate'],
                                         target_duration=config['data']['data_duration'])

    # Convert lists to numpy arrays
    clean_data = np.array(clean_audio_files)
    noisy_data = np.array(noisy_audio_files)

    # Expand dimensions for Conv1D input shape compatibility
    noisy_data = np.expand_dims(noisy_data, axis=-1)

    # Process with UNet and Autoencoder models
    unet_model_results, autoencoder_model_results = process_models(noisy_data, clean_data, unet_model,
                                                                   autoencoder_model, config['data']['sample_rate'])

    # Print results and average scores for UNet and Autoencoder models
    print_results(unet_model_results, "UNet Model")
    average_pesq_unet, average_stoi_unet = calculate_average_scores(unet_model_results)
    print_average_scores(average_pesq_unet, average_stoi_unet)

    print_results(autoencoder_model_results, "Autoencoder Model")
    average_pesq_autoencoder, average_stoi_autoencoder = calculate_average_scores(autoencoder_model_results)
    print_average_scores(average_pesq_autoencoder, average_stoi_autoencoder)


if __name__ == '__main__':
    main()
