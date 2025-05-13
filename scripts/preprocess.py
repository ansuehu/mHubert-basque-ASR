from datasets import load_from_disk, Audio, concatenate_datasets
import re
import json
import os
from utils import *
import os

def remove_special_characters(batch):
    """Remove special characters from text."""
    chars_to_ignore_regex = r'[,\?\.\!\-\;\:\"\'\'\'\`\'\±\ã\¿\¡\|]'
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

def load_and_prepare_data(data_name, sample_size=None):
    """Load and prepare the Basque Common Voice dataset."""
    # Load the dataset
    data = load_from_disk(data_name)
    print(f"Original dataset size: {len(data)}")
    
    # Take a subset of the dataset
    if sample_size != None:
        data = data.select(range(sample_size))
        print(f"Subset size: {len(data)}")
    
    # Split the dataset
    #data = data.train_test_split(test_size=0.2)
    for key in data.keys():
        print(f"  Length of {key} Set: {len(data[key])}")
    
    # Select only relevant columns
    data = data.select_columns(["audio", "sentence"])
    
    # Clean the text data
    data = data.map(remove_special_characters)
    
    return data

def create_vocabulary(data, dataset, save = True):
    """Create and save vocabulary from the dataset."""
    # Extract all unique characters to create vocabulary
    vocabs = data.map(
        extract_all_chars, 
        batched=True, 
        batch_size=-1, 
        keep_in_memory=True, 
        remove_columns=data.column_names["train"]
    )
    
    # Combine vocabularies from train and test sets
    # vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    
    combined_vocab = set()

    for split in vocabs.keys():
        # Add vocab from each split to the combined vocab set
        combined_vocab.update(vocabs[split]["vocab"][0])

    # Convert the combined vocab set back to a list (optional)
    vocab_list = list(combined_vocab)
    # Create vocabulary dictionary
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    
    # Handle special tokens
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(f"Vocabulary size: {len(vocab_dict)}")

    print(vocab_dict)
    
    # Save vocabulary to JSON file
    if save:
        with open(f'./data/{dataset}/vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)
    
    print("Vocabulary saved to vocab.json")
    
    return vocab_dict

def process_audio_data(data, processor, sampling_rate):
    """Process audio data for the model."""
    # Set the sampling rate for audio
    data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))
    
    # Process the dataset
    prepare_fn = lambda batch: prepare_dataset(batch, processor)
    data = data.map(
        prepare_fn, 
        remove_columns=data.column_names["train"], 
        # num_proc=4
    )
    
    return data

def main():
    """Main function to orchestrate the dataset preprocessing."""
    print("Starting Dataset preprocessing")
    print("-------------------------------------------")
    os.chdir('/home/andoni.sudupe/mHubert_finetune')
    
    model_name = "utter-project/mHuBERT-147"
    # 1. Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    dataset = 'composite_eues'
    data_name = f'/home/andoni.sudupe/mHubert_finetune/data/{dataset}/raw_data'
    sample_size = None
    data = load_and_prepare_data(data_name, sample_size)
    
    # 2. Create vocabulary
    print("\nStep 2: Creating vocabulary...")
    create_vocabulary(data, dataset)
    
    # 3. Setup processor
    print("\nStep 3: Setting up processor...")
    processor, sampling_rate = setup_processor(model_name, vocab_path=f"./data/{dataset}/vocab.json")
    
    # 4. Process audio data
    print("\nStep 4: Processing audio data...")
    data = process_audio_data(data, processor, sampling_rate)

    test_splits = [ds for k, ds in data.items() if k.startswith("test")]

# Combine into one test set
    if test_splits:
        data["test"] = concatenate_datasets(test_splits)

    # # Optionally, remove the old test splits
    # for k in list(data.keys()):
    #     if k.startswith("test") and k != "test":
    #         del data[k]

    print("\nStep 5: Saving data...")
    data.save_to_disk(os.path.join('data', dataset, 'preprocessed_data'))

if __name__ == "__main__":
    main()