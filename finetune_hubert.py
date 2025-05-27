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
import argparse
from accelerate import Accelerator

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

def train_model(model_name,
                data_train,
                data_dev,
                processor,
                ctc_only = False,
                output_dir="checkpoints/mHubert_basque",
                push_to_hub=False,
                learning_rate=1e-5,
                batch_size=8,
                num_train_epochs=30,
                dataloader_num_workers=4,):
    
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
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
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
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(batch_size/2),
        eval_strategy="steps",
        num_train_epochs=num_train_epochs,
        fp16=False,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-08,
        warmup_ratio=0.1,
        save_total_limit=5,
        push_to_hub=False,
        group_by_length=False, #Bestela asko tardatzen du hasten entrenamendua. Erabili nahi bada datuek 'length' izeneko zutabe bat izan behar dute.
        load_best_model_at_end=True,
        dataloader_num_workers=dataloader_num_workers,
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

    accelerator = Accelerator()
    trainer = accelerator.prepare(trainer)
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
       
    # Push to HuggingFace Hub if credentials are available
    if push_to_hub != None:
        trainer.push_to_hub(token=push_to_hub)
        print(f"Model pushed to HuggingFace Hub: {output_dir}")
    
    
    return model

def continue_train_model(model_name,
                         data_train,
                         data_dev,
                         processor,
                         ctc_only, 
                         output_dir, 
                         push_to_hub,
                         learning_rate,
                         batch_size,
                         num_train_epochs,
                         gradient_accumulation_steps,
                         dataloader_num_workers
                         ):
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
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size/2,
        eval_strategy="steps",
        num_train_epochs=num_train_epochs,
        fp16=False,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-08,
        warmup_ratio=0.1,
        save_total_limit=5,
        push_to_hub=False,
        group_by_length=False, #Bestela asko tardatzen du hasten entrenamendua. Erabili nahi bada datuek 'length' izeneko zutabe bat izan behar dute.
        load_best_model_at_end=True,
        dataloader_num_workers=dataloader_num_workers
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
    accelerator = Accelerator()
    trainer = accelerator.prepare(trainer)
    
    # Train the model
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint = True)
       
    # Push to HuggingFace Hub if credentials are available
    if push_to_hub != None:
        trainer.push_to_hub(token=push_to_hub)
        print(f"Model pushed to HuggingFace Hub: {output_dir}")
    
    
    return model


def main():
    """Main function to orchestrate the entire Basque Speech Recognition pipeline."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Basque Speech Recognition Pipeline")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/preprocessed_data", 
                        help="Directory containing preprocessed data")
    parser.add_argument("--split", type=str, default="All", 
                        help="Data split to use (train, dev, test, or All)")
    parser.add_argument("--sample_size", type=int, default=None, 
                        help="Sample size for training data (None for all data)")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, 
                        default="utter-project/mHuBERT-147",
                        help="Pre-trained model name or path to local model")
    parser.add_argument("--continue_training", action="store_true", 
                        help="Continue training from a checkpoint")
    parser.add_argument("--ctc_only", action="store_true", 
                        help="Use CTC loss only (no language model)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, 
                        default="checkpoints/mHubert-basque-ASR",
                        help="Directory to save model checkpoints")
    parser.add_argument("--push_to_hub", type=str, default=None,
                        help="Repository name to push model to HuggingFace Hub (None to disable)")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--dataloader_num_workers", type=int, default=1,
                        help="DataLoader num workers")

    # Parse arguments
    args = parser.parse_args()

    print("Starting Basque Speech Recognition Pipeline")
    print("-------------------------------------------")
    
    # 1. Load and prepare data
    print(f"\nStep 1: Loading and preparing data from {args.data_dir}...")
    data = load_data(args.data_dir, split=args.split, sample_size=args.sample_size)
    
    # Set up processor
    processor, _ = setup_processor(args.model_name, args.data_dir + "/../vocab.json")
    
    # 2. Train model
    print("\nStep 2: Training model...")
    if args.continue_training:
        print(f"Continuing training from checkpoint: {args.model_name}")
        model = continue_train_model(
            args.model_name, 
            data['train'], 
            data['dev'], 
            processor, 
            ctc_only=args.ctc_only, 
            output_dir=args.output_dir, 
            push_to_hub=args.push_to_hub,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_num_workers=args.dataloader_num_workers
        )
    else:
        print(f"Starting training from pre-trained model: {args.model_name}")
        model = train_model(
            args.model_name, 
            data['train'], 
            data['dev'], 
            processor, 
            ctc_only=args.ctc_only, 
            output_dir=args.output_dir, 
            push_to_hub=args.push_to_hub,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            dataloader_num_workers=args.dataloader_num_workers
        )
    
    # 3. Save final model
    print(f"\nStep 3: Saving final model to {args.output_dir}...")
    model.save_pretrained(f"{args.output_dir}/final")
    
if __name__ == "__main__":
    main()