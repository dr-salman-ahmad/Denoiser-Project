from denoiser import pretrained
from denoiser.dsp import convert_audio
import torch
import torchaudio
import os


def denoise_audio(noisy_file, denoised_file, model):
    # Load the noisy audio file
    wav, sr = torchaudio.load(noisy_file)
    wav = convert_audio(wav.cpu(), sr, model.sample_rate, model.chin)

    # Denoise the audio
    with torch.no_grad():
        denoised = model(wav[None])[0]

    # Save the denoised audio file
    torchaudio.save(denoised_file, denoised.cpu(), model.sample_rate)


# Load the pretrained model
model = pretrained.dns64().cpu()

# Define input and output directories
noisy_dir = 'Dataset/noisy_dir'
denoised_dir = 'Dataset/denoised_dir'

# Create denoised directory if it doesn't exist
os.makedirs(denoised_dir, exist_ok=True)

# Process each file in the noisy directory
for filename in os.listdir(noisy_dir):
    noisy_file = os.path.join(noisy_dir, filename)
    denoised_file = os.path.join(denoised_dir, filename)

    if os.path.isfile(noisy_file):
        denoise_audio(noisy_file, denoised_file, model)
        print(f'Denoised file saved: {denoised_file}')
