from datasets import load_dataset
import os

ds = load_dataset("HiTZ/composite_corpus_eseu_v1.0", split="train")

# Save the dataset to the `data` folder
ds.save_to_disk(os.path.join('data', 'composite_eues', 'raw_data'))