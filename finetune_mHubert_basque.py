import re
import json
import random
import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    HubertForCTC,
    TrainingArguments,
    Trainer
)
from jiwer import wer
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# Define the data collator class for CTC
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences.
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length.
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


def remove_special_characters(batch):
    """Remove special characters from text."""
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\'\'\'\`\']'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def extract_all_chars(batch):
    """Extract all unique characters to create vocabulary."""
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def prepare_dataset(batch, processor):
    """Prepare dataset by converting audio to input values and text to labels."""
    audio = batch["audio"]

    # Batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


def compute_metrics(pred, processor):
    """Compute Word Error Rate (WER) metric."""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # We do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_score = wer(hypothesis=pred_str, reference=label_str)

    return {"wer": wer_score}


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


def load_and_prepare_data(sample_size=1000):
    """Load and prepare the Basque Common Voice dataset."""
    # Load the dataset
    common_voice = load_dataset(
        "mozilla-foundation/common_voice_13_0", "eu", split="test"
    )
    print(f"Original dataset size: {len(common_voice)}")
    
    # Take a subset of the dataset
    common_voice = common_voice.select(range(sample_size))
    print(f"Subset size: {len(common_voice)}")
    
    # Split the dataset
    common_voice = common_voice.train_test_split(test_size=0.2)
    print(f"Train size: {len(common_voice['train'])}, Test size: {len(common_voice['test'])}")
    
    # Select only relevant columns
    common_voice = common_voice.select_columns(["audio", "sentence"])
    
    # Clean the text data
    common_voice = common_voice.map(remove_special_characters)
    
    return common_voice


def create_vocabulary(common_voice):
    """Create and save vocabulary from the dataset."""
    # Extract all unique characters to create vocabulary
    vocabs = common_voice.map(
        extract_all_chars, 
        batched=True, 
        batch_size=-1, 
        keep_in_memory=True, 
        remove_columns=common_voice.column_names["train"]
    )
    
    # Combine vocabularies from train and test sets
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    
    # Create vocabulary dictionary
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    
    # Handle special tokens
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(f"Vocabulary size: {len(vocab_dict)}")
    
    # Save vocabulary to JSON file
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    
    print("Vocabulary saved to vocab.json")
    
    return vocab_dict


def setup_processor(vocab_path="./vocab.json"):
    """Set up the processor with tokenizer and feature extractor."""
    # Initialize tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        word_delimiter_token="|"
    )
    
    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147")
    
    # Combine into processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    return processor, feature_extractor.sampling_rate


def process_audio_data(common_voice, processor, sampling_rate):
    """Process audio data for the model."""
    # Set the sampling rate for audio
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))
    
    # Display sample audio information
    rand_int = random.randint(0, len(common_voice["train"])-1)
    print("\nSample audio information:")
    print(f"Target text: {common_voice['train'][rand_int]['sentence']}")
    print(f"Input array shape: {np.asarray(common_voice['train'][rand_int]['audio']['array']).shape}")
    print(f"Sampling rate: {common_voice['train'][rand_int]['audio']['sampling_rate']}")
    
    # Process the dataset
    prepare_fn = lambda batch: prepare_dataset(batch, processor)
    common_voice = common_voice.map(
        prepare_fn, 
        remove_columns=common_voice.column_names["train"], 
        num_proc=4
    )
    
    return common_voice


def train_model(common_voice, processor, output_dir="mHubert_basque"):
    """Train the HuBERT model."""
    # Initialize model
    model = HubertForCTC.from_pretrained("utter-project/mHuBERT-147")
    
    # Freeze base model for transfer learning
    model.freeze_base_model()
    
    # Initialize data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )
    
    # Set up metrics function
    metrics_fn = lambda pred: compute_metrics(pred, processor)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=metrics_fn,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        tokenizer=processor.feature_extractor,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the model
    model.save_pretrained("./hubert-finetuned-eu")
    
    # Push to HuggingFace Hub if credentials are available
    try:
        trainer.push_to_hub()
        print(f"Model pushed to HuggingFace Hub: {output_dir}")
    except Exception as e:
        print(f"Could not push to HuggingFace Hub: {e}")
    
    return model


def evaluate_model(common_voice, model, processor):
    """Evaluate the trained model on test data."""
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Define mapping function for evaluation
    map_fn = lambda batch: map_to_result(batch, model, processor)
    
    # Apply mapping function to test data
    results = common_voice.map(map_fn, remove_columns=common_voice["test"].column_names)
    
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
    """Main function to orchestrate the entire pipeline."""
    print("Starting Basque Speech Recognition Pipeline")
    print("-------------------------------------------")
    
    # 1. Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    common_voice = load_and_prepare_data(sample_size=1000)
    
    # 2. Create vocabulary
    print("\nStep 2: Creating vocabulary...")
    create_vocabulary(common_voice)
    
    # 3. Setup processor
    print("\nStep 3: Setting up processor...")
    processor, sampling_rate = setup_processor()
    
    # 4. Process audio data
    print("\nStep 4: Processing audio data...")
    common_voice = process_audio_data(common_voice, processor, sampling_rate)
    
    # 5. Train model
    print("\nStep 5: Training model...")
    model = train_model(common_voice, processor)
    
    # 6. Evaluate model
    print("\nStep 6: Evaluating model...")
    results = evaluate_model(common_voice, model, processor)
    
    print("\nSpeech recognition pipeline completed!")
    print(f"Final WER: {wer(hypothesis=results['test']['pred_str'], reference=results['test']['text']):.3f}")


if __name__ == "__main__":
    main()
