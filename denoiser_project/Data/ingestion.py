from tqdm import tqdm
from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path
from util import setup_logger

LOG = setup_logger()


def process_audio(file_path, output_dir):
    audio = AudioSegment.from_file(file_path)
    chunk_length_ms = 10000  # 10 seconds in milliseconds

    chunks = make_chunks(audio, chunk_length_ms)

    for index, chunk in enumerate(chunks):
        chunk_length_sec = len(chunk) / 1000
        try:
            if 1 <= chunk_length_sec <= 10:
                file_path = Path(file_path)
                output_dir = Path(output_dir)
                chunk_name = output_dir / f"{file_path.stem}_chunk{index}.wav"
                chunk.export(chunk_name, format="wav")
        except Exception as e:
            LOG.error(f"Error while writing chunk {index}: {e}")


def process_files(files_to_process, data_dir, processed_dir):
    data_dir = Path(data_dir)
    processed_dir = Path(processed_dir)
    for file in tqdm(files_to_process):
        file_path = data_dir / file
        process_audio(file_path, processed_dir)
