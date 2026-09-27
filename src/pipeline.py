"""
pipeline.py

Orchestrates the video-processing pipeline end to end.
Currently wired for Steps 2-3 (audio extraction + transcription).
Steps 5-6 (frame_extractor, chunker) will be added as they're built --
this file is the single place that calls them in order.
"""

from src.video_metadata import VideoMetadata
from src.audio_extractor import extract_audio
from src.transcriber import transcribe_audio
import time

# Not wired in yet -- uncomment once frame_extractor.py / chunker.py are implemented
# from src.frame_extractor import extract_slide_text
# from src.chunker import chunk_segments


def run_pipeline(video_path: str, subject: str = None) -> dict:
    """
    Runs the video through every implemented step in order.

    Args:
        video_path: path to the input MP4
        subject: optional subject label (e.g. "Biology") used to build a
                 light, safe transcription prompt -- does not hardcode
                 vocabulary for any one subject

    Returns a summary dict with paths/results produced so far.
    """
    pipeline_start = time.time()
    meta = VideoMetadata(video_path=video_path, subject=subject)
    print(f"[pipeline] Starting: {meta.title} (doc_id={meta.doc_id}, subject={meta.subject or 'none'})")

    # Step 2: video -> audio
    t0 = time.time()
    audio_path = extract_audio(meta)
    print(f"[pipeline] Audio extraction took {time.time() - t0:.1f}s")

    # Step 3: audio -> timestamped transcript
    t0 = time.time()
    segments = transcribe_audio(audio_path, meta)
    print(f"[pipeline] Transcription took {time.time() - t0:.1f}s")

    # Step 5 (TODO): video -> slide text via OCR
    # slide_segments = extract_slide_text(meta)
    # segments = sorted(segments + slide_segments, key=lambda s: s.start)

    # Step 6 (TODO): segments -> final chunks for Team B
    # chunks = chunk_segments(segments, meta)

    total_time = time.time() - pipeline_start
    print(f"[pipeline] Done. {len(segments)} transcript segments produced.")
    print(f"[pipeline] TOTAL TIME: {total_time:.1f}s")

    return {
        "doc_id": meta.doc_id,
        "audio_path": audio_path,
        "num_segments": len(segments),
        "segments": segments,
        "total_time_seconds": total_time,
    }