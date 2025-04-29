import argparse
import os
import numpy as np
import torch
from transformers import Wav2Vec2Processor, HubertModel
from datasets import load_dataset, Audio, load_from_disk
from tqdm import tqdm

def extract_features_batch(batch, model, processor, device):
   
    batch_audio = [b["array"] for b in batch] 
    # Process the audio inputs
    inputs = processor(
        batch_audio, 
        sampling_rate=batch[0]["sampling_rate"],
        return_tensors="pt", 
        padding="longest"
    ).to(device)
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # Take the mean of the hidden states to get fixed-length embeddings
    features = outputs.hidden_states[9].squeeze(0).cpu().numpy()
    
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
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--num_proc", type=int, default=4,
                       help="Number of processes for dataset loading")
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
    print("Loading model and processor...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = HubertModel.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset {args.dataset_path}...")
    dataset = load_from_disk(
        args.dataset_path, 
    )
    
    dataset = dataset['train']
    # Ensure dataset has audio format
    if not any(column.startswith("audio") for column in dataset.column_names):
        print("Adding audio column to dataset...")
        dataset = dataset.cast_column("audio", Audio())
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Process dataset in batches
    features_list = []
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    
    for i in range(0, len(dataset), args.batch_size):
        batch_idx = i // args.batch_size
        end_idx = min(i + args.batch_size, len(dataset))
        batch_size = end_idx - i
        
        print(f"Processing batch {batch_idx+1}/{total_batches} (samples {i}-{end_idx-1})...")
        # try:
            # Get batch and extract features
        batch = dataset[i:end_idx]
        features = extract_features_batch(batch['audio'], model, processor, device)
        features_list.append(features)
        
        print(f"  Processed {batch_size} samples, feature shape: {features.shape}")
        # except Exception as e:
        #     print(f"  Error processing batch {batch_idx+1}: {str(e)}")
    
    if features_list:
        # Concatenate all features
        all_features = np.vstack(features_list)
        print(f"Total extracted features shape: {all_features.shape}")
        
        # Save to file
        np.save(args.output_file, all_features)
        print(f"Features saved to: {args.output_file}")
    else:
        print("No features were extracted. Check the dataset and model.")

if __name__ == "__main__":
    main()