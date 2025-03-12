from datasets import load_dataset
import os

ds = load_dataset("asierhv/composite_corpus_eu_v2.1")

# Save the dataset to the `data` folder
ds.save_to_disk(os.path.join('data', 'raw_data'))