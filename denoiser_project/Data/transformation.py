import librosa
import numpy as np
from pathlib import Path
from util import setup_logger

LOG = setup_logger()


def is_audio_empty(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return audio.size == 0
    except Exception as e:
        LOG.error(f"Error while reading {file_path}: {e}")
        return True


def is_audio_completely_silent(audio, silence_threshold=0.01):
    amplitude = np.abs(audio)
    return np.all(amplitude < silence_threshold)


def load_audio_files(directory, sr=16000, max_files=10, target_duration=10):
    target_length = int(sr * target_duration)
    audio_files = []
    file_count = 0
    directory = Path(directory)

    for file_path in directory.glob('*.wav'):
        if not is_audio_empty(file_path):
            audio, _ = librosa.load(file_path, sr=sr)

            # Padding for same shape
            if audio.shape[0] < target_length:
                audio = np.pad(audio, (0, target_length - audio.shape[0]), 'constant')
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            if is_audio_completely_silent(audio):
                LOG.error(f"Skipping completely silent file: {file_path}")
                continue
            audio_files.append(audio[:target_length])
            file_count += 1
            if file_count >= max_files:
                break
        else:
            LOG.error(f"Skipping file: {file_path}")
    return np.array(audio_files)
