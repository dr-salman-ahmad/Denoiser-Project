from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import torch
from util import setup_logger, load_config

LOG = setup_logger()
CONFIG = load_config("../../config.yaml")


def main():
    if CONFIG is None:
        LOG.error("Config file not provided.")
        return

    # Load pretrained model and tokenizer (processor)
    model_name = CONFIG['asr_inference']['model_name']
    data_path = CONFIG['data']['asr_data_path']
    sampling_rate = CONFIG['asr_inference']['sample_rate']

    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # Example usage: transcribe an audio file
    audio_file = f"{data_path}/wavs/LJ001-0001.wav"
    input_audio, _ = librosa.load(audio_file, sr=sampling_rate)  # Adjust sample rate as per your audio
    input_values = processor(input_audio, return_tensors="pt", sampling_rate=sampling_rate).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    LOG.info("Transcription:", transcription.lower())


if __name__ == "__main__":
    main()
