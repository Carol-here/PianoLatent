"""PianoLatent CRNN inference adapter.

This module reuses the exact model, feature extraction, sliding-window inference,
and note decoder from the existing project-root main2.py. Keep that trained source
file and its checkpoint in the project root; do not duplicate/change the model.
"""
from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The project's existing main.py defines the trained architecture and the
# exact preprocessing/inference/decoding implementation used during evaluation.
core = importlib.import_module("main2")

SAMPLE_RATE = int(core.SAMPLE_RATE)
CHECKPOINT_DEFAULT = PROJECT_ROOT / "outputs" / "spectrograms" / "checkpoints" / "pianolatent_v2_best.pt"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_model(checkpoint_path: str | None = None):
    configured_path = checkpoint_path or os.environ.get("PIANOLATENT_CHECKPOINT")
    path = Path(configured_path) if configured_path else CHECKPOINT_DEFAULT
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"PianoLatent V2 checkpoint not found: {path}. Expected outputs/spectrograms/checkpoints/pianolatent_v2_best.pt or set PIANOLATENT_CHECKPOINT."
        )

    device = get_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict")
    if state is None:
        raise ValueError("Checkpoint does not contain model_state_dict")

    if not core._all_finite_in_state(state):
        raise FloatingPointError("V2 checkpoint contains non-finite model weights")

    model = core.PianoLatentCRNN().to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. This adapter requires the "
            "PianoLatent V2 checkpoint pianolatent_v2_best.pt, not V1 or E1."
        ) from exc
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, device, payload, str(path)


def transcribe_audio(
    audio_path: str | Path,
    onset_threshold: float = 0.35,
    offset_threshold: float = 0.30,
    frame_threshold: float = 0.35,
    batch_size: int = 16,
    checkpoint_path: str | None = None,
):
    """Run the production CRNN pipeline on an uploaded audio file."""
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if audio.size < 4096:
        raise ValueError("Audio is too short. Upload at least 0.2 seconds of audio.")
    duration = float(audio.size / SAMPLE_RATE)

    features = core.extract_training_features(audio)
    model, device, checkpoint, checkpoint_used = load_model(checkpoint_path)
    predictions = core.infer_recording(model, features, device, batch_size=batch_size)
    events = core.decode_note_events(
        predictions,
        onset_threshold=float(onset_threshold),
        offset_threshold=float(offset_threshold),
        frame_threshold=float(frame_threshold),
    )

    # Convert NumPy scalar values to plain Python values for JSON serialization.
    clean_events = [
        {
            "pitch": int(e["pitch"]),
            "onset": float(e["onset"]),
            "offset": float(e["offset"]),
            "duration": float(e["offset"] - e["onset"]),
            "velocity": int(e["velocity"]),
            "note_name": librosa.midi_to_note(int(e["pitch"])),
        }
        for e in events
    ]
    return {
        "events": clean_events,
        "duration": duration,
        "sample_rate": SAMPLE_RATE,
        "note_count": len(clean_events),
        "checkpoint": checkpoint_used,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device),
    }