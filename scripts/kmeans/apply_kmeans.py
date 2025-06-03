import torch
import torchaudio
import numpy as np
import joblib
from transformers import Wav2Vec2Processor, HubertModel

def extract_features(waveform, model, processor):
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.hidden_states[9].squeeze(0).cpu().numpy()

def assign_tokens(audio_path, model, processor, kmeans):
    features = extract_features(audio_path, model, processor)
    tokens = kmeans.predict(features)
    return tokens

def main():
    model_name = "Ansu/mHubert-basque-ASR"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.eval()
    kmeans = joblib.load("checkpoints/kmeans/hubert_kmeans.pkl")
    
    audio_path = "data/composite_eu/audios/sample_1.wav"
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    waveform = waveform.squeeze(0)
    tokens = assign_tokens(audio_path, model, processor, kmeans)
    print("Assigned tokens:", tokens)

if __name__ == "__main__":
    main()