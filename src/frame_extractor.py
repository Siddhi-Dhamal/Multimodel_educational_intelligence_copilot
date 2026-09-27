"""
frame_extractor.py

STEP 5 of the pipeline: Video -> Slide Frames + OCR (NOT YET IMPLEMENTED)

This will extract frames on scene-change (slide transitions) and run OCR
on them, so slide text gets merged into the transcript timeline for
lectures that are slide-heavy. Ask for this once transcriber.py is working.
"""

from typing import List
from src.video_metadata import TranscriptSegment, VideoMetadata


def extract_slide_text(meta: VideoMetadata, output_dir: str = "output/frames") -> List[TranscriptSegment]:
    """
    Placeholder. Will return a list of TranscriptSegment(source="slide", ...)
    built from OCR'd frame text, timestamped by frame position in the video.
    """
    raise NotImplementedError("frame_extractor.py: Step 5 not implemented yet")