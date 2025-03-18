import torch
from datasets import load_from_disk, Audio
from transformers import (
    Wav2Vec2Processor,
    HubertForCTC,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from scripts.utils import setup_processor, compute_metrics, load_data
from evaluate import load


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

def train_model(model_name, data_train, data_dev, processor, ctc_only = False, output_dir="checkpoints/mHubert_basque", push_to_hub=False):
    """Train the HuBERT model."""
    # Initialize model
    model = HubertForCTC.from_pretrained(
        model_name,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.1,
        mask_time_prob=0.05,
        layerdrop=0.1,
        final_dropout=0.3,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    
    # Freeze base model for transfer learning
    if ctc_only:
        model.freeze_base_model()
    else:
        model.freeze_feature_encoder()
    
    # Initialize data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        num_train_epochs=10,
        fp16=False,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-08,
        warmup_ratio=0.1,
        save_total_limit=5,
        push_to_hub=False,
        group_by_length=False, #Bestela asko tardatzen du hasten entrenamendua. Erabili nahi bada datuek 'length' izeneko zutabe bat izan behar dute.
        load_best_model_at_end=True,
    )
    
    # Set up metrics function
    wer_metric = load("wer", trust_remote_code=True)
    cer_metric = load("cer", trust_remote_code=True)

    metrics_fn = lambda pred: compute_metrics(pred, processor, wer_metric, cer_metric)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=metrics_fn,
        train_dataset=data_train,
        eval_dataset=data_dev,
        processing_class=processor.feature_extractor,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
       
    # Push to HuggingFace Hub if credentials are available
    if push_to_hub:
        trainer.push_to_hub()
        print(f"Model pushed to HuggingFace Hub: {output_dir}")
    
    
    return model

def main():
    """Main function to orchestrate the entire pipeline."""
    print("Starting Basque Speech Recognition Pipeline")
    print("-------------------------------------------")
    
    # 1. Load and prepare data
    print("\nStep 1: Loading and preparing data...")

    sample_size=None
    data = load_data('data/preprocessed_data', split='All', sample_size=sample_size)
    
    model_name = "utter-project/mHuBERT-147"

    processor, _ = setup_processor(model_name)

    # 2. Train model
    print("\nStep 2: Training model...")
    model = train_model(model_name, data['train'], data['dev'], processor, ctc_only=True, output_dir='checkpoints/mHubert-ASR-eu')

    # 3. Save model
    print("\nStep 3: Saving model...")
    model.save_pretrained("checkpoints/hubert-finetuned-eu")
    
if __name__ == "__main__":
    main()
