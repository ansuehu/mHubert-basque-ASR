# Technical Report: Fine-Tuning the HuBERT Model
## Introduction
HuBERT (Hidden-Unit BERT) is a self-supervised speech representation learning model designed to process and understand speech data effectively. By leveraging a masked prediction objective, HuBERT learns meaningful representations directly from raw audio, enabling it to excel in various speech-related tasks. This report focuses on the fine-tuning of the multilingual variant, mHuBERT-147, to enhance its performance on Basque for the automatic speech recognition (ASR) task.

## State of the Art


## Training
### Model
The fine-tuning process was initiated using the pre-trained mHuBERT-147 model available through the Hugging Face Transformers library. This model was chosen for its multilingual capabilities and proven effectiveness in ASR applications. The use of pre-trained weights provided a robust foundation, reducing the need for training from scratch and accelerating convergence.

### Dataset
The dataset used for fine-tuning was the Composite Corpus for Basque from Hitz, which aggregates publicly available Basque audio resources, such as Common Voice and Basque Parliament. This dataset comprises approximately 700 hours of Basque speech data paired with transcriptions.

### Process
The fine-tuning workflow was executed using the Hugging Face Transformers library, which offers tools for training and evaluation of transformer-based models. The process involved the following steps:

#### Data Preprocessing:
Text data was cleaned by removing special characters and used to construct a vocabulary. Audio data was resampled to 16 kHz and processed accordingly. Both the audio and text data were formatted and stored for efficient loading during training.

#### Hyperparameter Tuning:
Although extensive tuning was not performed, we adopted hyperparameters commonly used in previous mHuBERT fine-tuning experiments.

#### Training Strategy:
Multiple training setups were explored, varying both the model type and the training mode. We compared the original HuBERT with the multilingual mHuBERT. Since these models are not task-specific, a Connectionist Temporal Classification (CTC) head was added for ASR. We experimented with two approaches: (a) fine-tuning only the CTC head, and (b) fine-tuning both the CTC head and the base model.

#### Validation:
A validation split was reserved from the dataset to evaluate the model’s performance during training and to mitigate overfitting.

The training objective focused on minimizing the Word Error Rate (WER) and Character Error Rate (CER), which are standard evaluation metrics for ASR systems.

## Experiments and Results
The longest training run spanned 50 epochs over 10 days. The results indicate that the multilingual mHuBERT model consistently outperformed the original HuBERT model in terms of learning speed and final accuracy on Basque. Notably, fine-tuning only the CTC head did not yield meaningful results, suggesting that updating the base model's parameters is crucial for learning language-specific features. The best-performing configuration was the mHuBERT model trained for 50 epochs, achieving a Character Error Rate (CER) of 3.4% and a Word Error Rate (WER) of 9.9%.