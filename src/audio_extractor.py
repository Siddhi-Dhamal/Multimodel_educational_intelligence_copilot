"""
audio_extractor.py

STEP 2 of the pipeline: Video -> Audio

Extracts a 16kHz mono WAV from an uploaded MP4, ready for faster-whisper.
Requires ffmpeg installed on the system (not the Python package -- the CLI tool).
"""

import subprocess
import os
from pathlib import Path
from typing import List, Tuple

from src.video_metadata import VideoMetadata


def extract_audio(meta: VideoMetadata, output_dir: str = "output/audio") -> str:
    """
    Extracts audio from an MP4 file and saves it as a 16kHz mono WAV.

    Args:
        meta: VideoMetadata for the source video
        output_dir: folder where the extracted .wav will be saved

    Returns:
        path to the extracted .wav file
    """
    video_path = Path(meta.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    audio_path = Path(output_dir) / f"{meta.doc_id}.mp3"

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-ar", "16000",   # Whisper wants 16kHz
        "-ac", "1",        # mono
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",  # normalize volume, helps quiet speech
        "-codec:a", "libmp3lame", "-b:a", "64k",  # compressed -- ~5x smaller than WAV, same speech quality
        "-vn",              # strip video stream
        "-y",                # overwrite if exists
        str(audio_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path.name}:\n{result.stderr}")

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("Audio extraction produced an empty/missing file.")

    print(f"[audio_extractor] OK: {audio_path} "
          f"({audio_path.stat().st_size / 1024:.1f} KB)")
    return str(audio_path)


def get_audio_duration(audio_path: str) -> float:
    """Returns the duration of an audio file in seconds, via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed to read duration:\n{result.stderr}")
    return float(result.stdout.strip())


def split_audio_into_chunks(
    audio_path: str,
    output_dir: str = "output/audio_chunks",
    chunk_seconds: int = 60,
    overlap_seconds: int = 5,
) -> List[Tuple[str, float]]:
    """
    Splits a WAV into smaller overlapping chunks for more reliable
    transcription. Sending a long, continuous stretch of audio to Whisper
    in one request risks small stretches being silently dropped, or the
    model spiraling into a repetition loop -- both confirmed failure modes
    observed in practice. Shorter chunks give the model far less room for
    either. The overlap ensures a sentence sitting on a chunk boundary is
    captured by at least one of the two overlapping chunks.

    Args:
        audio_path: path to the full WAV file
        output_dir: folder to save chunk files
        chunk_seconds: length of each chunk (default 60s)
        overlap_seconds: overlap between consecutive chunks (default 5 sec)

    Returns:
        list of (chunk_file_path, chunk_start_offset_seconds) tuples, in order
    """
    audio_path = Path(audio_path)
    os.makedirs(output_dir, exist_ok=True)

    duration = get_audio_duration(audio_path)
    step = chunk_seconds - overlap_seconds
    if step <= 0:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    chunks: List[Tuple[str, float]] = []
    start = 0.0
    idx = 0
    while start < duration:
        chunk_path = Path(output_dir) / f"{audio_path.stem}_chunk{idx:03d}{audio_path.suffix}"
        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(chunk_seconds),
            "-c", "copy",
            "-y",
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to create chunk {idx}:\n{result.stderr}")

        chunks.append((str(chunk_path), start))
        print(f"[audio_extractor] Chunk {idx}: {chunk_path.name} "
              f"(starts at {start:.1f}s)")

        start += step
        idx += 1

    print(f"[audio_extractor] Split into {len(chunks)} chunks "
          f"({chunk_seconds}s each, {overlap_seconds}s overlap)")
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.audio_extractor <path_to_video.mp4>")
        sys.exit(1)

    test_meta = VideoMetadata(video_path=sys.argv[1])
    extract_audio(test_meta)