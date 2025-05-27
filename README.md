# mHubert-basque-ASR

This model is a fine-tuned version of [utter-project/mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147) on the [composite_corpus_eu_v2.1](https://huggingface.co/datasets/asierhv/composite_corpus_eu_v2.1) dataset for ASR in Basque.

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
- num_epochs: 24
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.48.3
- Pytorch 2.5.1+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0

### Sample predictions: 

Test CV WER: 0.074

Test CV CER: 0.013

Sample predictions:

- Reference: honek garrantzi handia zuen ehun urteko gerran
- Prediction: honek garrantzi handia zuen eun urteko gerran

- Reference: osasuna aurkari zuzena da eta beraz puntuek balio bikoitza dute
- Prediction: osasuna aurkari zuzena da eta beraz puntuek balio bikoitza dute

- Reference: irungo familia boteretsu bat da olazabal familia
- Prediction: irungo familia boteretsu bat da olazabal familia

- Reference: hezkuntzak prestatu zituen probak pisa eta antzekoak eredu
- Prediction: hezkuntzak prestatu zituen probak pisa eta antzekoak eredu

- Reference: bestalde botilek abangoardiako diseinu orijinalak dituzte
- Prediction: bestalde botillek abanbardiako diseinu originalak dituzte

--------------

Test Parl WER: 0.068

Test Parl CER: 0.018

Sample predictions:

- Reference: por iñigo cabacas eskerrik asko eskerrik asko
- Prediction: por inigo cabacas eskerrik asko eskerrik asko

- Reference: eta ikusita obra hau hamar urteetan bueltaka ibili dela eta ikusten da zaharkitutako
- Prediction: eta ikusita obra hau hamar urteetan bueltaka ibili dela eta ikusten da zaharkitutako

- Reference: dena legearen garapen zuzena oztopatzeko helburuarekin ez dut nik esango ez eskatzaile guztiek
- Prediction: dena legearen garapen zuzena oztopatzeko helburuarekin ez dut nik esango ez eskatzaile guztiek

- Reference: eginda da eginikoa da ea gaurko adostasunak
- Prediction: eginda da eginekoa da ea gaurko adostasunak

- Reference: kontatu gabe eta udalen ordezkarien izenean izena joan gabe
- Prediction: kontatu gabe eta udalen ordezkarien izenea izenean joan gabe

--------------

Test OSLR WER: 0.204

Test OSLR CER: 0.042

Sample predictions:
- Reference: new yorkeko aireportuan eskala egin genuen kaliforniara bidean
- Prediction: new yyorkeko aireportua neskala egin genuen kaliforniara bidean

- Reference: janet jackson michael jackson abeslari ospetsuaren arreba da
- Prediction: janez jason mikel jaxon abeslari ospetsuaren arreba da

- Reference: londreseko heathrow aireportua munduko handienetarikoena da
- Prediction: londreseko hitrow aireportua munduko handienetarikoa da

- Reference: hamabietan izango da txupinazoa eta udaletxeko balkoitik botako dute urtero bezala
- Prediction: hamabitan izango da txupinasoa eta udaletxeko palkoitik botako dute urtero bezala

- Reference: motorolaren telefono berria erostekotan nabil
- Prediction: motrolaren telefono berria erostekotan nabil


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

## Train your model

```bash
sbatch train.slurm
```

To continue a training, change some things form train.slurm (Entrenamendurako kode zatiye komentau ta continue_training kode zatiye deskomentau).
After finishing a continued training, some thing may need to change in order to work. Inside of the last checkpoint it will be a file named trainer_state.json and another file named config.json. In the first one, the line that says "best_model_checkpoint" need to be changed to the name of the checkpoint, and the same in the second file, but the line that says "_name_or_path"