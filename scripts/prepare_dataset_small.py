from datasets import load_dataset
import os
import soundfile as sf

ds = load_dataset("HiTZ/composite_corpus_eu_v2.1", split="train", streaming=True)

# iterable_dataset = ds.to_iterable_dataset(num_shards=1)

# Save the dataset to the `data` folder

# Save the audio to a WAV file
for i, sample in enumerate(ds):
    audio_data = sample['audio']['array']
    sample_rate = sample['audio']['sampling_rate']
    sf.write(f'/home/andoni.sudupe/mHubert_finetune/data/composite_eu/audios/sample_{i}.wav', audio_data, sample_rate)
    if i > 10:
        break

# iterable_dataset.save_to_disk(os.path.join('data', 'composite_eu', 'small_raw_data'))