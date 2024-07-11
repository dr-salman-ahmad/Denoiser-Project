from denoiser import pretrained
from denoiser.dsp import convert_audio
import torch
import torchaudio
from pathlib import Path
import shutil
from util import setup_logger, load_config

LOG = setup_logger()
CONFIG = load_config("../../config.yaml")


def denoise_audio(noisy_file, denoised_file, model):
    try:
        # Load the noisy audio file
        wav, sr = torchaudio.load(str(noisy_file))
        wav = convert_audio(wav.cpu(), sr, model.sample_rate, model.chin)

        # Denoise the audio
        with torch.no_grad():
            denoised = model(wav[None])[0]

        # Save the denoised audio file
        torchaudio.save(str(denoised_file), denoised.cpu(), model.sample_rate)
        LOG.info(f'Denoised file saved: {denoised_file}')

    except Exception as e:
        LOG.error(f'Error denoising {noisy_file}: {str(e)}')


def main():
    if CONFIG is None:
        LOG.error("Config file not provided.")
        return
    # Load the pretrained model
    model = pretrained.dns64().cpu()

    # Define input and output directories
    noisy_dir = CONFIG['inference']['noisy_data_path']
    denoised_dir = CONFIG['inference']['denoised_data_path']

    # Remove existing denoised files (if any)
    shutil.rmtree(denoised_dir, ignore_errors=True)

    # Create denoised directory if it doesn't exist
    Path(denoised_dir).mkdir(parents=True, exist_ok=True)

    # Process each file in the noisy directory
    for file_path in Path(noisy_dir).iterdir():
        denoised_file = Path(denoised_dir) / f"{file_path.stem}.wav"

        if file_path.is_file() and file_path.suffix == '.wav':
            denoise_audio(file_path, denoised_file, model)
        else:
            LOG.error(f'Skipping {file_path}')


if __name__ == '__main__':
    main()
