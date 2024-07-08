from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import torch
import yaml


def main():
    with open('../../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # Load pretrained model and tokenizer (processor)
    model_name = config['wave2vec']['model_name']
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # Example usage: transcribe an audio file
    audio_file = config['data']['asr_data_path'] + "/wavs/LJ001-0001.wav"
    input_audio, _ = librosa.load(audio_file, sr=config['wave2vec']['sample_rate'])  # Adjust sample rate as per your audio
    input_values = processor(input_audio, return_tensors="pt", sampling_rate=config['wave2vec']['sample_rate']).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print("Transcription:", transcription.lower())


if __name__ == "__main__":
    main()
