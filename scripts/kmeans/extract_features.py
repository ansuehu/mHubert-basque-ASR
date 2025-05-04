import argparse
import os
import numpy as np
import torch
from transformers import Wav2Vec2Processor, HubertModel
from datasets import load_dataset, Audio, load_from_disk
from tqdm import tqdm

def extract_features_batch(audio, model, device, target_length=12*16000):
    # Pad or truncate audio arrays to the target length
    array = audio["input_values"]
    if len(array) < target_length:
        array = np.pad(array, (0, target_length - len(array)), mode='constant')
    else:
        array = array[:target_length]

    input_values = torch.tensor(array, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    features = outputs.hidden_states[9].cpu().numpy()
    return features


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Extract HuBERT features from a HuggingFace dataset")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or path to local model")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="HuggingFace dataset name or path to local dataset")
    parser.add_argument("--split", type=str, default="All",
                        help="Dataset split to process (train, validation, test)")
    parser.add_argument("--output_file", type=str, default="hubert_features.npy",
                        help="Output file to save extracted features")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Use only local files for loading model and dataset")
    
    # Parse arguments
    args = parser.parse_args()
    
    print(f"Starting feature extraction using model: {args.model_name}")
    print(f"Processing dataset: {args.dataset_path} (split: {args.split})")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    print("Loading model...")
    model = HubertModel.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset['train']
    print(f"Dataset size: {len(dataset)} samples")
    
    # Feature extraction
    features_list = []
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            features = extract_features_batch(sample, model, device)
            features_list.append(features)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
    
    if features_list:
        all_features = np.vstack(features_list)
        print(f"Total extracted features shape: {all_features.shape}")
        np.save(args.output_file, all_features)
        print(f"Features saved to: {args.output_file}")
    else:
        print("No features were extracted. Check the dataset and model.")

if __name__ == "__main__":
    main()