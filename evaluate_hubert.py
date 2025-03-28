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

def main():
    
    print('Starting evaluation')
    model_name = '/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-30ep/checkpoint-146000'

    tokenizer = Wav2Vec2CTCTokenizer(
        '/home/andoni.sudupe/mHubert_finetune/data/vocab.json', 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        word_delimiter_token="|"
    )

    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    # Combine into processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = HubertForCTC.from_pretrained(model_name, local_files_only=True)

    data = load_data('/home/andoni.sudupe/mHubert_finetune/data/preprocessed_data')
    # 6. Evaluate model
    print("\nStep 6: Evaluating model...")
    results, metric = evaluate_model(data['test_cv'], model, processor)

    print(f"\nTest CV WER: {metric['test_wer']:.3f}")
    print(f"\nTest CV CER: {metric['test_cer']:.3f}")
    
    # Display sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(results))):
        print(f"Reference: {results['text'][i]}")
        print(f"Prediction: {results['pred_str'][i]}")
        print("---")

    print('--------------')

    results, metric = evaluate_model(data['test_parl'], model, processor)

    print(f"\nTest Parl WER: {metric['test_wer']:.3f}")
    print(f"\nTest Parl CER: {metric['test_cer']:.3f}")
    
    # Display sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(results))):
        print(f"Reference: {results['text'][i]}")
        print(f"Prediction: {results['pred_str'][i]}")
        print("---")
    
    print('--------------')

    results, metric = evaluate_model(data['test_oslr'], model, processor)

    print(f"\nTest OSLR WER: {metric['test_wer']:.3f}")
    print(f"\nTest OSLR CER: {metric['test_cer']:.3f}")
    
    # Display sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(results))):
        print(f"Reference: {results['text'][i]}")
        print(f"Prediction: {results['pred_str'][i]}")
        print("---")

if __name__ == "__main__":
    main()