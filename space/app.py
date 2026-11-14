
import gradio as gr
import librosa
import torch
from transformers import pipeline

MODEL_ID = "gio2499/whisper-small-ka"

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=0 if torch.cuda.is_available() else -1,
)


def transcribe(audio_path):
    if audio_path is None:
        return ""
    # Read audio with librosa (no ffmpeg needed), resampled to the 16 kHz Whisper expects.
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    result = asr(
        {"raw": audio, "sampling_rate": sr},
        chunk_length_s=30,
        generate_kwargs={"language": "georgian", "task": "transcribe"},
    )
    return result["text"]


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Georgian speech"),
    outputs=gr.Textbox(label="ტრანსკრიფცია / Transcription"),
    title="Georgian Speech Recognition — Whisper fine-tuned",
    description="Whisper fine-tuned on Mozilla Common Voice Georgian. Speak or upload a short clip.",
)

if __name__ == "__main__":
    demo.launch()
