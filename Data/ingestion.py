import os
from tqdm import tqdm
from pydub import AudioSegment
from pydub.utils import make_chunks


def process_audio(file_path, output_dir):
    audio = AudioSegment.from_file(file_path)
    chunk_length_ms = 10000  # 10 seconds in milliseconds

    chunks = make_chunks(audio, chunk_length_ms)

    for i, chunk in enumerate(chunks):
        chunk_length_sec = len(chunk) / 1000
        if 1 <= chunk_length_sec <= 10:
            chunk_name = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_chunk{i}.wav")
            chunk.export(chunk_name, format="wav")


def process_files(files_to_process, data_dir, processed_dir):
    for file in tqdm(files_to_process):
        file_path = os.path.join(data_dir, file)
        process_audio(file_path, processed_dir)
