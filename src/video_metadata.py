"""
video_metadata.py

Shared data structures used across the pipeline (audio_extractor, transcriber,
frame_extractor, chunker). Keeping these in one place ensures every module
speaks the same schema -- this is the contract you hand off to Team B.
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class VideoMetadata:
    """Identifies a single video being processed through the pipeline."""
    video_path: str
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: Optional[str] = None
    subject: Optional[str] = None  # e.g. "Biology", "Physics" -- used to build a
                                     # transcription prompt hint; None = generic/no bias

    def __post_init__(self):
        if self.title is None:
            # derive a readable title from the filename if not given
            from pathlib import Path
            self.title = Path(self.video_path).stem


@dataclass
class TranscriptSegment:
    """A single unit of transcribed speech with timing."""
    text: str
    start: float   # seconds
    end: float     # seconds
    source: str = "audio"  # "audio" or "slide" (OCR text tagged this way in Step 5)


@dataclass
class Chunk:
    """
    The final output unit pushed to Team B's vector DB.
    Field names are the API contract -- do not rename without syncing with
    Team B/C first.
    """
    text: str
    doc_id: str
    source_type: str        # "video"
    timestamp_start: float
    timestamp_end: float
    source_path: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "timestamp_start": round(self.timestamp_start, 2),
            "timestamp_end": round(self.timestamp_end, 2),
            "source_path": self.source_path,
        }