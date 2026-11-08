"""PianoLatent Standard MIDI File exporter."""
from __future__ import annotations

from pathlib import Path
import math
import pretty_midi


def export_midi(events, destination, sustain_events=None, program: int = 0):
    destination = Path(destination)
    if destination.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("Destination must have a .mid or .midi extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not 0 <= int(program) <= 127:
        raise ValueError("General MIDI program must be between 0 and 127")

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=int(program), name="PianoLatent Piano")
    normalized = []
    for index, event in enumerate(events):
        pitch = int(event["pitch"])
        velocity = int(event["velocity"])
        onset = float(event["onset"])
        offset = float(event.get("offset", onset + float(event.get("duration", 0.1))))
        if not 0 <= pitch <= 127:
            raise ValueError(f"Invalid MIDI pitch at event {index}: {pitch}")
        if not 1 <= velocity <= 127:
            raise ValueError(f"Invalid MIDI velocity at event {index}: {velocity}")
        if not math.isfinite(onset) or not math.isfinite(offset) or onset < 0 or offset <= onset:
            raise ValueError(f"Invalid note timing at event {index}")
        normalized.append((pitch, velocity, onset, offset))

    normalized.sort(key=lambda item: (item[2], item[0], item[3]))
    for pitch, velocity, onset, offset in normalized:
        piano.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=onset, end=offset))

    controls = []
    for control in sustain_events or []:
        time = float(control["time"])
        value = int(control["value"])
        if not math.isfinite(time) or time < 0 or not 0 <= value <= 127:
            raise ValueError("Invalid CC64 sustain event")
        controls.append((time, value))
    for time, value in sorted(controls):
        piano.control_changes.append(pretty_midi.ControlChange(number=64, value=value, time=time))

    midi.instruments.append(piano)
    midi.write(str(destination))
    return {"path": str(destination), "note_count": len(normalized), "sustain_event_count": len(controls)}
