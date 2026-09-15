import io
import base64
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGET_SR = 16000
N_FFT = 512
HOP_LENGTH = 256


def load_audio_from_bytes(file_bytes: bytes):
    audio_buffer = io.BytesIO(file_bytes)
    y, sr = sf.read(audio_buffer, dtype='float32')

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val * 0.95

    return torch.from_numpy(y).unsqueeze(0).float(), TARGET_SR


def wav_to_base64(waveform: torch.Tensor, sample_rate: int = 16000) -> str:
    audio_np = waveform.squeeze().detach().cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:audio/wav;base64,{encoded}"


def generate_spectrogram_base64(waveform: torch.Tensor, title: str = "Spectrogram") -> str:
    y = waveform.squeeze().detach().cpu().numpy()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    librosa.display.specshow(D, sr=TARGET_SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_title(title, color='#f8fafc', fontsize=10, fontweight='bold')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buffer.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    s_power = np.mean(signal ** 2)
    n_power = np.mean(noise ** 2)
    if n_power < 1e-10:
        return 35.0
    return float(10 * np.log10(max(s_power / n_power, 1e-10)))