---
library_name: transformers
license: cc-by-nc-sa-4.0
base_model: utter-project/mHuBERT-147
tags:
- generated_from_trainer
datasets:
- asierhv/composite_corpus_eu_v2.1
language:
- eu
metrics:
- wer
- cer
model-index:
- name: hubert_for_basque
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# hubert_for_basque

This model is a fine-tuned version of [utter-project/mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147) on the composite_corpus_eu_v2.1 dataset.

Test WER: 0.137

Test CER: 0.024

## Training procedure

All the training and evaluation code is on https://github.com/ansuehu/mHubert-basque-ASR 

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 64
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- num_epochs: 7
- mixed_precision_training: Native AMP

### Training results

| Steps | Eval Loss | WER (%) |
| ----- | --------- | ------- |
| 1000  | 9.779     | 99.99   |
| 2000  | 7.505     | 99.99   |
| 3000  | 4.986     | 99.99   |
| 4000  | 3.284     | 99.99   |
| 5000  | 2.880     | 99.99   |
| 6000  | 2.819     | 99.99   |
| 7000  | 2.746     | 99.99   |
| 8000  | 1.164     | 85.65   |
| 9000  | 0.634     | 66.24   |
| 10000 | 0.419     | 56.65   |
| 11000 | 0.332     | 48.98   |
| 12000 | 0.290     | 44.82   |
| 13000 | 0.265     | 41.41   |
| 14000 | 0.248     | 39.06   |
| 15000 | 0.240     | 37.38   |
| 16000 | 0.229     | 36.15   |
| 17000 | 0.217     | 34.58   |
| 18000 | 0.211     | 34.10   |
| 19000 | 0.207     | 33.06   |
| 20000 | 0.199     | 32.45   |
| 21000 | 0.193     | 31.75   |
| 22000 | 0.188     | 31.00   |
| 23000 | 0.183     | 30.45   |
| 24000 | 0.181     | 30.08   |
| 25000 | 0.175     | 29.49   |
| 26000 | 0.173     | 29.35   |
| 27000 | 0.170     | 29.01   |
| 28000 | 0.166     | 28.64   |
| 29000 | 0.165     | 28.53   |
| 30000 | 0.165     | 28.30   |
| 31000 | 0.163     | 27.83   |
| 32000 | 0.161     | 27.79   |
| 33000 | 0.157     | 27.42   |
| 34000 | 0.155     | 27.09   |
| 35000 | 0.156     | 26.95   |
| 36000 | 0.153     | 26.84   |
| 37000 | 0.151     | 26.71   |
| 38000 | 0.149     | 26.50   |
| 39000 | 0.148     | 26.35   |
| 40000 | 0.149     | 26.14   |
| 41000 | 0.147     | 25.92   |
| 42000 | 0.145     | 26.04   |
| 43000 | 0.145     | 26.02   |
| 44000 | 0.146     | 25.94   |
| 45000 | 0.145     | 25.81   |

### Framework versions

- Transformers 4.48.3
- Pytorch 2.5.1+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0

## Test results

Map: 100%|██████████| 16359/16359 [09:32<00:00, 28.58 examples/s]

Test WER: 0.137

Test CER: 0.024

### Sample predictions: 

Reference: honek garrantzi handia zuen ehun urteko gerran 
Prediction: honek garrantzi handia zuen neun urteko gerran 

Reference: osasuna aurkari zuzena da eta beraz puntuek balio bikoitza dute
Prediction: osasuna aurkari zuzena da eta beraz puntuek balio bikoitza dute

Reference: irungo familia boteretsu bat da olazabal familia
Prediction: eirungo familia goteretsu bat da olazabal familia

Reference: hezkuntzak prestatu zituen probak pisa eta antzekoak eredu
Prediction: hezkuntzak prestatu zituen probak isa eta antzekoak ere du

Reference: bestalde botilek abangoardiako diseinu orijinalak dituzte
Prediction: bestalde botileka ban bardiako diseinu originalak dituzte


## How to use

```python
from transformers import AutoProcessor, AutoModelForCTC
import torch
from datasets import load_dataset

# Load model and processor
processor = AutoProcessor.from_pretrained("Ansu/mHubert_basque_ASR")
model = AutoModelForCTC.from_pretrained("Ansu/mHubert_basque_ASR")

# Load audio from dataset
ds = load_dataset("asierhv/composite_corpus_eu_v2.1", split="test")
audio_input = ds[0]["audio"]

#Load audio from local file
audio = AudioSegment.from_file('path/to/audio')
audio = audio.set_frame_rate(16000)  # Set frame rate to 16kHz

# Convert to raw PCM audio data
# Create a BytesIO object to simulate an in-memory file
with io.BytesIO() as wav_file:
    # Export the audio to the in-memory file
    audio.export(wav_file, format='wav')
    # Seek to the beginning of the file before reading
    wav_file.seek(0)
    # Read the audio data as a NumPy array
    audio_input = wavfile.read(wav_file)[1]  # read data from wave file

# Process audio
inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# Decode output
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription[0])
```
