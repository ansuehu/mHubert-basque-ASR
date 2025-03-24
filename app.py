import gradio as gr
import torch
from transformers import HubertForCTC, Wav2Vec2Processor
import librosa

# Load the model and processor from Hugging Face Hub
model_name = "Ansu/mHubert_basque_ASR"  # Change this to your model
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = HubertForCTC.from_pretrained(model_name)

# Function to transcribe audio
def transcribe(audio):
    # Load audio file
    audio, _ = librosa.load(audio, sr=16000)
    
    # Process input
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Get model predictions
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode predictions
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

# Create Gradio interface
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs="text",
    title="HuBERT Basque ASR Demo",
    description="Upload an audio file and get the transcription.",
)

iface.launch()
