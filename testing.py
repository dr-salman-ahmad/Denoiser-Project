import os
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi

# Initialize the model
model = pretrained.dns64().cpu()

# Directories containing noisy and clean audio files
noisy_dir = 'Dataset/noisy_dir'
clean_dir = 'Dataset/clean_dir'

# Initialize lists to store results
results = []

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
total_pesq = sum(result['pesq'] for result in results)
total_stoi = sum(result['stoi'] for result in results)
average_pesq = total_pesq / len(results)
average_stoi = total_stoi / len(results)

print(f'\nAverage PESQ Score: {average_pesq}')
print(f'Average STOI Score: {average_stoi}')