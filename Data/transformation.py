import os
import librosa
import numpy as np


def is_audio_empty(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return audio.size == 0
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return True


def is_audio_completely_silent(audio, silence_threshold=0.01):
    amplitude = np.abs(audio)
    return np.all(amplitude < silence_threshold)


def load_audio_files(directory, sr=16000, max_files=10, target_duration=10):
    target_length = int(sr * target_duration)
    audio_files = []
    file_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            if not is_audio_empty(file_path):
                audio, _ = librosa.load(file_path, sr=sr)

                # Padding for same shape
                if audio.shape[0] < target_length:
                    audio = np.pad(audio, (0, target_length - audio.shape[0]), 'constant')
                # Normalize audio
                audio = audio / np.max(np.abs(audio))
                if is_audio_completely_silent(audio):
                    print(f"Skipping completely silent file: {file_path}")
                    continue
                audio_files.append(audio[:target_length])
                file_count += 1
                if file_count >= max_files:
                    break
            else:
                print(f"Skipping empty or invalid file: {file_path}")
    return audio_files
