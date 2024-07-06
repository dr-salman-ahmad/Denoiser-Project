from denoiser import pretrained
from denoiser.dsp import convert_audio
import torch
import torchaudio
import os
import yaml
import shutil


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def denoise_audio(noisy_file, denoised_file, model):
    try:
        # Load the noisy audio file
        wav, sr = torchaudio.load(noisy_file)
        wav = convert_audio(wav.cpu(), sr, model.sample_rate, model.chin)

        # Denoise the audio
        with torch.no_grad():
            denoised = model(wav[None])[0]

        # Save the denoised audio file
        torchaudio.save(denoised_file, denoised.cpu(), model.sample_rate)
        print(f'Denoised file saved: {denoised_file}')

    except Exception as e:
        print(f'Error denoising {noisy_file}: {str(e)}')


def main():
    config = load_config('../config.yaml')
    # Load the pretrained model
    model = pretrained.dns64().cpu()

    # Define input and output directories
    noisy_dir = config['inference']['noisy_data_path']
    denoised_dir = config['inference']['denoised_data_path']

    # Remove existing denoised files (if any)
    shutil.rmtree(denoised_dir, ignore_errors=True)

    # Create denoised directory if it doesn't exist
    os.makedirs(denoised_dir, exist_ok=True)

    # Process each file in the noisy directory
    for filename in os.listdir(noisy_dir):
        noisy_file = os.path.join(noisy_dir, filename)
        denoised_file = os.path.join(denoised_dir, filename)

        if os.path.isfile(noisy_file) and filename.endswith('.wav'):
            denoise_audio(noisy_file, denoised_file, model)
        else:
            print(f'Skipping {filename}')


if __name__ == '__main__':
    main()
