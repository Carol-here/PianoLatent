from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .crnn_synth import transcribe_audio
from .midi_exporter import export_midi
from .reconstruction import refine_events_against_audio

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

app = FastAPI(title="PianoLatent API", version="2.0.0")
JOBS: dict[str, dict] = {}


@app.get("/api/health")
def health():
    import torch
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    onset_threshold: float = Form(0.35),
    offset_threshold: float = Form(0.30),
    frame_threshold: float = Form(0.35),
    refine: bool = Form(False),
    refine_steps: int = Form(20),
):
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported audio format. Use WAV, MP3, FLAC, OGG, or M4A.")
    if not (0.05 <= onset_threshold <= 0.95 and 0.05 <= offset_threshold <= 0.95 and 0.05 <= frame_threshold <= 0.95):
        raise HTTPException(400, "Decoder thresholds must be between 0.05 and 0.95")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds the 100 MB limit")

    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / f"input{suffix}"
    midi_path = job_dir / "pianolatent.mid"
    audio_path.write_bytes(content)

    try:
        result = await run_in_threadpool(
            transcribe_audio,
            audio_path,
            onset_threshold,
            offset_threshold,
            frame_threshold,
        )
        refinement_report = {"used": False}
        events = result["events"]
        if refine:
            events, refinement_report = await run_in_threadpool(
                refine_events_against_audio,
                audio_path,
                events,
                refine_steps,
            )
        midi_info = await run_in_threadpool(export_midi, events, midi_path)
    except Exception as exc:
        # Keep the server response concise; detailed exception remains in server logs.
        raise HTTPException(500, f"Transcription failed: {type(exc).__name__}: {exc}") from exc

    JOBS[job_id] = {"midi_path": midi_path, "audio_path": audio_path}
    return {
        "job_id": job_id,
        "filename": file.filename,
        "duration": result["duration"],
        "sample_rate": result["sample_rate"],
        "device": result["device"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "note_count": len(events),
        "notes": events,
        "midi": midi_info,
        "refinement": refinement_report,
        "methodology": "Supervised CRNN transcription; optional analytical-synth reconstruction refinement.",
    }


@app.get("/api/download/{job_id}")
def download_midi(job_id: str):
    job = JOBS.get(job_id)
    if not job or not Path(job["midi_path"]).exists():
        raise HTTPException(404, "MIDI export not found. Run a transcription first.")
    return FileResponse(
        path=job["midi_path"],
        media_type="audio/midi",
        filename="pianolatent_transcription.mid",
    )


# In production, build the Vite frontend and copy frontend/dist to application/web.
WEB_DIR = ROOT / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="frontend")