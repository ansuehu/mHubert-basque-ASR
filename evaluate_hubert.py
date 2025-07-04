import torch
from scripts.utils import setup_processor
from transformers import HubertForCTC
from scripts.utils import load_data
from evaluate import load
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    HubertForCTC,
)
import argparse
from tqdm import tqdm
import librosa
import soundfile as sf
from datasets import load_dataset
import os
import json

def map_to_result(batch, model, processor):
    """Map model predictions to text for evaluation."""
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch

def evaluate_model(data, model, processor):
    """Evaluate the trained model on test data."""
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Define mapping function for evaluation
    map_fn = lambda batch: map_to_result(batch, model, processor)
    
    # Apply mapping function to test data
    results = data.map(map_fn, remove_columns=data.column_names)
    
    # Calculate WER
    wer_metric = load("wer", trust_remote_code=True)
    cer_metric = load("cer", trust_remote_code=True)

    metric = {}

    metric['test_wer'] = wer_metric.compute(predictions=results["pred_str"], references=results["text"])
    metric['test_cer'] = cer_metric.compute(predictions=results["pred_str"], references=results["text"])
   
    return results, metric

def evaluate_model_manifest(data, model, processor):
    wer_metric = load("wer", trust_remote_code=True)
    cer_metric = load("cer", trust_remote_code=True)

    predictions = []
    references = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    for i in tqdm(range(len(data))):
        wav_path = data[i]['audio_filepath']
        text = data[i]['text']
        # Read audio
        audio_input, sample_rate = sf.read(wav_path)
        # Convert to mono if stereo
        if len(audio_input.shape) == 2:
            audio_input = audio_input.mean(axis=1)
        # Resample if needed
        if sample_rate != processor.feature_extractor.sampling_rate:
            audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=processor.feature_extractor.sampling_rate)
            sample_rate = processor.feature_extractor.sampling_rate
        # Preprocess
        inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        # Predict
        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        predictions.append(pred_str)
        references.append(text)
        

    # Calculate metrics
    results = {}
    results['pred_str'] = predictions
    results['text'] = references
    results['test_wer'] = wer_metric.compute(predictions=predictions, references=references)
    results['test_cer'] = cer_metric.compute(predictions=predictions, references=references)
    return results

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Evaluate Basque Speech Recognition Model")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, 
                        default="/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-30ep/checkpoint-302700",
                        help="Path to the model checkpoint for evaluation")
    parser.add_argument("--vocab_path", type=str,
                        default="basque_vocab.json",
                        help="Path to the vocabulary JSON file")
    parser.add_argument("--data_dir", type=str,
                        default="/home/andoni.sudupe/mHubert_finetune/data/preprocessed_data",
                        help="Directory containing preprocessed evaluation data")
    parser.add_argument("--manifest", type=bool, 
                        default=False,
                        help="Manifest json files?")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["test_cv", "test_parl", "test_oslr"],
                        help="List of test datasets to evaluate")
    parser.add_argument("--num_samples", type=int,
                        default=5,
                        help="Number of samples to display predictions for")
    
    # Parse arguments
    args = parser.parse_args()
    
    print('Starting evaluation')
    print(f'Model path: {args.model_path}')
    print(f'Evaluating on datasets: {args.datasets}')
    
    

    if args.manifest:
        tokenizer = Wav2Vec2CTCTokenizer(
                args.model_path+'/'+args.vocab_path, 
                unk_token="[UNK]", 
                pad_token="[PAD]", 
                word_delimiter_token="|"
            )
        
        # Initialize feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)

        # Combine into processor
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        # Load model
        model = HubertForCTC.from_pretrained(args.model_path, local_files_only=True)

        data = load_dataset("json", data_files=args.data_dir)

        model_name = os.path.join(*args.model_path.split(os.sep)[-2:]).replace("/", "-")
        dataset_name = os.path.join(*args.data_dir.split(os.sep)[-2:]).replace("/", "-")

        for split_name in args.datasets:
            if split_name not in data:
                print(f"\nWarning: Dataset '{split_name}' not found in loaded data. Skipping.")
                continue
            print(f"\n{'-' * 50}")
            

            results = evaluate_model_manifest(data[split_name], model, processor)
            # Save results to file
            # with open(f"/home/andoni.sudupe/mHubert_finetune/results/{dataset_name}_{model_name}_results.json", "w") as f:
            #     json.dump(results, f, indent=4)
            # Print metrics
            print(f"\n{split_name.upper()} WER: {results['test_wer']:.3f}")
            print(f"{split_name.upper()} CER: {results['test_cer']:.3f}")
            
            # Display sample predictions
            print(f"\nSample predictions for {split_name}:")
            for i in range(min(args.num_samples, len(results))):
                print(f"Reference: {results['text'][i]}")
                print(f"Prediction: {results['pred_str'][i]}")
                print("---")

    else:
        # Initialize tokenizer
        tokenizer = Wav2Vec2CTCTokenizer(
            args.model_path+'/'+args.vocab_path, 
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            word_delimiter_token="|"
        )

        # Initialize feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)

        # Combine into processor
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        # Load model
        model = HubertForCTC.from_pretrained(args.model_path, local_files_only=True)  
        # Load data
        data = load_data(args.data_dir)
        
        # Evaluate on each specified test dataset
        for dataset_name in args.datasets:
            if dataset_name not in data:
                print(f"\nWarning: Dataset '{dataset_name}' not found in loaded data. Skipping.")
                continue
                
            print(f"\n{'-' * 50}")
            print(f"Evaluating on {dataset_name}...")
            
            # Evaluate model on this dataset
            results, metric = evaluate_model(data[dataset_name], model, processor)

            # Print metrics
            print(f"\n{dataset_name.upper()} WER: {metric['test_wer']:.3f}")
            print(f"{dataset_name.upper()} CER: {metric['test_cer']:.3f}")
            
            # Display sample predictions
            print(f"\nSample predictions for {dataset_name}:")
            for i in range(min(args.num_samples, len(results))):
                print(f"Reference: {results['text'][i]}")
                print(f"Prediction: {results['pred_str'][i]}")
                print("---")

if __name__ == "__main__":
    main()