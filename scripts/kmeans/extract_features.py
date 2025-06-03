import argparse
import os
import numpy as np
import torch
from transformers import Wav2Vec2Processor, HubertModel
from datasets import load_dataset, Audio, load_from_disk
from tqdm import tqdm
import time
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from npy_append_array import NpyAppendArray

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
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch Size for processing")
    parser.add_argument("--feat_file", type=str, required=True,
                        help="Output file for extracted features")
    parser.add_argument("--len_file", type=str, required=True,
                        help="Output file for lengths of extracted features")
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
    processor = Wav2Vec2Processor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = HubertModel.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset['train']
    dataset = dataset.select(range(100))
    print(f"Dataset size: {len(dataset)} samples")
    
    # Define a custom collate function for padding
    # def collate_fn_pad(batch):
    #     has = time.time()
    #     max_length = 8 * 16000  # Define the fixed length for padding
    #     audio_tensors = [torch.tensor(sample["input_values"]).to(device) for sample in batch]
    #     padded_audio = torch.stack([
    #         torch.nn.functional.pad(audio, (0, max(0, max_length - audio.size(0))), mode='constant', value=0)[:max_length]
    #         for audio in audio_tensors
    #     ])
    #     buk = time.time()
    #     print(f"Time taken for collate function: {buk - has:.2f} seconds")
    #     return padded_audio

    def collate_fn_pad(batch):
        # max_length = 12 * 16000  # Define the fixed length for padding (128,000 samples)

        # Convert input values to tensors and stack them into a single tensor
        audio_tensors = torch.stack([torch.tensor(s["input_values"]) for s in batch]).to(device)

        audio_tensors = processor(audio_tensors, sampling_rate=16000, return_tensors="pt")

        return audio_tensors
        # batch_size = len(audio_tensors)

        # # Create a padded tensor with the desired max_length
        # padded_audio = torch.zeros((batch_size, max_length), device=device)
        # for i, audio in enumerate(audio_tensors):
        #     length = min(audio.size(0), max_length)
        #     padded_audio[i, :length] = audio[:length]

        # return padded_audio

    # Create a DataLoader with the custom collate function
    print(f"Creating DataLoader with batch size {args.batch_size}...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_pad)
    # Feature extraction

    print(f"Extracting features from {len(dataloader)} batches...")
    if os.path.isfile(args.feat_file):
        print(f"Feature file already exists: {args.feat_file}")
        os.remove(args.feat_file)
        print(f"Removed: {args.feat_file}")
    with NpyAppendArray(args.feat_file) as feat_f:
        with open(args.len_file, "w") as leng_f:
            for batch in tqdm(dataloader):
                # print(batch.shape)
                print(batch['input_values'].shape)
                # input_values = batch.squeeze(0).to(torch.float32)  # Extract audio data from the batch and ensure it's in the correct format
                with torch.no_grad():
                    outputs = model(batch['input_values'].squeeze(0).to(device), output_hidden_states=True)
                # Extract features from the 9th hidden state
                features = outputs.hidden_states[9].cpu().numpy()
                print(f"Feature shape: {features.shape}")
                for f in features:
                    feat_f.append(f)
                    leng_f.write(f"{len(f)}\n")
    print(f"Features saved to: {args.feat_file}")
    print(f"Lengths saved to: {args.len_file}")

    # # Save features to disk
    # if features_list:
    #     all_features = np.vstack(features_list)
    #     print(f"Total extracted features shape: {all_features.shape}")
    #     np.save(args.output_file, all_features)
    #     print(f"Features saved to: {args.output_file}")
    # else:
    #     print("No features were extracted. Check the dataset and model.")

if __name__ == "__main__":
    main()