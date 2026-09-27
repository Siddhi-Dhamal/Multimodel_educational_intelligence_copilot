"""
chunker.py

STEP 6 of the pipeline: Timestamped Segments -> Chunks (NOT YET IMPLEMENTED)

This will merge transcript + slide segments into one sorted timeline and
group them into ~60-90 second windows, producing the final Chunk objects
that get handed off to Team B's embedding/vector-DB step. Ask for this
once transcriber.py (and optionally frame_extractor.py) is working.
"""

from typing import List
from src.video_metadata import TranscriptSegment, Chunk, VideoMetadata


def chunk_segments(
    segments: List[TranscriptSegment],
    meta: VideoMetadata,
    window_seconds: float = 75.0,
) -> List[Chunk]:
    """
    Placeholder. Will group segments into time-windowed chunks with
    doc_id/timestamp metadata, ready for Chunk.to_dict() hand-off.
    """
    raise NotImplementedError("chunker.py: Step 6 not implemented yet")