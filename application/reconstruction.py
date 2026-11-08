"""Optional differentiable-synth reconstruction refinement for PianoLatent.

This is an optional experimental post-processing stage. It optimizes only
onset, key-down duration, and velocity for the CRNN's detected pitches. It does
not infer new pitches, pedal CC64, or guarantee better note-level F1.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
core = importlib.import_module("main2")


def refine_events_against_audio(
    audio_path: str | Path,
    events: list[dict],
    steps: int = 25,
    max_seconds: float = 8.0,
    max_notes: int = 60,
):
    """Fit CRNN event timing/velocity to an uploaded audio segment.

    The analytical synth is deliberately used as an optional approximation;
    it is not the original MAESTRO soundfont. A regularizer keeps parameters
    close to CRNN estimates to reduce unconstrained drift.
    """
    if not events:
        return events, {"used": False, "reason": "No notes to refine"}

    import librosa
    audio, _ = librosa.load(str(audio_path), sr=core.SAMPLE_RATE, mono=True)
    duration = min(float(max_seconds), len(audio) / core.SAMPLE_RATE)
    samples = int(duration * core.SAMPLE_RATE)
    if samples < 4096:
        return events, {"used": False, "reason": "Audio segment too short"}
    audio = audio[:samples].astype(np.float32)

    selected = [e for e in events if float(e["onset"]) < duration][:max_notes]
    if not selected:
        return events, {"used": False, "reason": "No notes in refinement window"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synth = core.DifferentiablePianoSynth(sample_rate=core.SAMPLE_RATE, harmonics=24).to(device)
    synth.eval()
    for parameter in synth.parameters():
        parameter.requires_grad_(False)

    target = torch.from_numpy(audio).to(device)
    initial = np.asarray([
        [e["pitch"], e["onset"], max(0.04, e["offset"] - e["onset"]), e["velocity"] / 127.0]
        for e in selected
    ], dtype=np.float32)
    base = torch.from_numpy(initial).to(device)

    # Unconstrained variables are mapped into safe ranges at each forward pass.
    onset_raw = torch.nn.Parameter(base[:, 1].clone())
    duration_raw = torch.nn.Parameter(base[:, 2].clamp_min(0.04).log())
    velocity_raw = torch.nn.Parameter(torch.logit(base[:, 3].clamp(0.02, 0.98)))
    optimizer = torch.optim.Adam([onset_raw, duration_raw, velocity_raw], lr=0.008)
    loss_fn = core.MultiResolutionSTFTLoss().to(device)

    best_loss = float("inf")
    best = base.clone()
    first_loss = None
    steps = max(1, min(int(steps), 100))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        onsets = onset_raw.clamp(0.0, max(0.0, duration - 0.02))
        note_durations = duration_raw.exp().clamp(0.04, 4.0)
        velocities = velocity_raw.sigmoid().clamp(0.02, 1.0)
        notes = torch.stack((base[:, 0], onsets, note_durations, velocities), dim=1)
        rendered = synth(notes, duration)
        if rendered.ndim == 2:
            rendered = rendered.mean(dim=0)
        audio_loss = loss_fn(rendered, target)
        regularizer = (
            ((onsets - base[:, 1]) / 0.08).square().mean()
            + ((note_durations - base[:, 2]) / 0.20).square().mean()
            + ((velocities - base[:, 3]) / 0.15).square().mean()
        )
        loss = audio_loss + 0.025 * regularizer
        if not torch.isfinite(loss):
            break
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([onset_raw, duration_raw, velocity_raw], 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            with torch.no_grad():
                best = torch.stack((base[:, 0], onsets, note_durations, velocities), dim=1).detach().clone()

    refined = [dict(e) for e in events]
    best_np = best.cpu().numpy()
    # Map the selected original event objects back to their list positions.
    selected_indices = [i for i, event in enumerate(events) if any(event is item for item in selected)]
    for event_index, row in zip(selected_indices, best_np):
        target_event = refined[event_index]
        target_event["onset"] = float(row[1])
        target_event["offset"] = float(min(duration, row[1] + row[2]))
        target_event["duration"] = float(target_event["offset"] - target_event["onset"])
        target_event["velocity"] = int(np.clip(round(row[3] * 127), 1, 127))

    return refined, {
        "used": True,
        "optimized_note_count": len(selected),
        "steps": steps,
        "initial_objective": first_loss,
        "best_objective": best_loss,
        "warning": "Audio reconstruction loss is not equivalent to note-level transcription accuracy.",
    }