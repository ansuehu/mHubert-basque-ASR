import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, HubertModel

def extract_features(audio_path, model, processor):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    waveform = waveform.squeeze(0)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    return outputs.hidden_states[9].squeeze(0).cpu().numpy()

def main():
    model_name = ''
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.eval()
    
    feature_list = []
    audio_folder = "path_to_dataset"
    
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav"):
            features = extract_features(os.path.join(audio_folder, filename), model, processor)
            feature_list.append(features)
    
    all_features = np.vstack(feature_list)
    np.save("hubert_features.npy", all_features)

if __name__ == "__main__":
    main()