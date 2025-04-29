# Technical Report: Fine-Tuning the HuBERT Model  

## Introduction  
HuBERT (Hidden-Unit BERT) is a self-supervised speech representation learning model designed to process and understand speech data effectively. By leveraging a masked prediction objective, HuBERT learns meaningful representations directly from raw audio, enabling it to excel in various speech-related tasks. This work focuses on the fine-tuning of the multilingual variant, mHuBERT-147, to enhance its performance on Basque by performing the automatic speech recognition (ASR) task.

## State of the Art  
Recent advancements in self-supervised learning have revolutionized the field of speech processing. Models like Wav2Vec 2.0, HuBERT, and their multilingual extensions have set new benchmarks in ASR tasks by leveraging large-scale unlabeled audio data. These models employ innovative training objectives, such as contrastive learning and masked prediction, to learn robust speech representations.  

mHuBERT-147, the multilingual variant of HuBERT, extends its capabilities to handle diverse linguistic datasets. It is particularly effective in multilingual ASR tasks, where the availability of labeled data for certain languages is limited. By pre-training on a wide range of languages, mHuBERT-147 provides a strong foundation for fine-tuning on specific datasets, making it a valuable tool for advancing speech recognition in underrepresented languages.  

## Training  
### Model  
The fine-tuning process began with the pre-trained mHuBERT-147 model, available through the HuggingFace Transformers library. This model was selected for its ability to process multilingual speech data and its proven performance in ASR tasks. The pre-trained weights provided a robust starting point, reducing the need for extensive training from scratch.  

### Dataset  
The dataset used was the Composite Corpus for Basque from Hitz, which is a collection of public available Basque data like Common Voice or Baque Parliament. This dataset consists of almost 700 hours of Basque audio along with transcriptions.  

### Process  
The fine-tuning process was conducted using the HuggingFace Transformers library, which provides useful functions for training and evaluating transformer-based models. The following steps were taken:  
1. **Data Preprocessing**: The audio and text data was converted into a suitable format. First, we process the text. We remove the special characters and create a vocabulary. After that, we process the audio data. Before passing the audio through the processor we ensure the audio is sampled in 16kHz. And the processed audio and text is saved in the disk. 
2. **Hyperparameter Tuning**: We did not perform an extensive hyperparameter tuning. We followed the hyperparameters seen in another mHubert fine-tunings.  
3. **Training**: We performed different trainings that differ on models or the training mode. As for the models, we tried the original Hubert along with the multimodal mHubert to see the difference between them. Also, as the these models are not trained for specific tasks, a CTC head is needed for performing the fine-tuning. We tried only fine-tuning the CTC head and also changing the base model.
4. **Validation**: A portion of the dataset was reserved for validation to monitor the model's performance and prevent overfitting.  

The training process emphasized minimizing word error rate (WER) and character error rate (CER), two critical metrics for evaluating ASR systems.  

## Experiments and Results  
The longest training was 50 epochs and lasted 10 days. Seeing the results, the first thing we can see is that the multilingual model outperforms the original Hubert model and is takes less to reach to a good level of Basque. Also, the model couldn't learn anything by only training the CTC head so we can conclude that also changing the parameters of the base model is necessary. The training that got the best results was the mHubert trained on 50 epochs, achieving 3.4% on CER and 9.9% on WER.