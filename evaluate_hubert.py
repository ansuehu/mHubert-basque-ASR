import torch
from jiwer import wer
from scripts.utils import setup_processor
from transformers import HubertForCTC
from scripts.utils import load_data

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
    results = data.map(map_fn, remove_columns=data["test"].column_names)
    
    # Calculate WER
    test_wer = wer(hypothesis=results['test']["pred_str"], reference=results['test']["text"])
    print(f"\nTest WER: {test_wer:.3f}")
    
    # Display sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(results['test']))):
        print(f"Reference: {results['test']['text'][i]}")
        print(f"Prediction: {results['test']['pred_str'][i]}")
        print("---")
    
    return results

def main():
    
    model_name = 'checkpoints/path'
    model = HubertForCTC(model_name)
    processor = setup_processor(model_name)

    data = load_data('/home/andoni.sudupe/mHubert_finetune/data/preprocessed_data')
    # 6. Evaluate model
    print("\nStep 6: Evaluating model...")
    results = evaluate_model(data, model, processor)
    
    print("\nSpeech recognition pipeline completed!")
    print(f"Final WER: {wer(hypothesis=results['test']['pred_str'], reference=results['test']['text']):.3f}")

if __name__ == "__main__":
    main()