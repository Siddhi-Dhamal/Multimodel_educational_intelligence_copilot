"""
transcriber.py

STEP 3 of the pipeline: Audio -> Timestamped Transcript

Uses Groq's Whisper API for cloud-based transcription. Single request per
video -- fast, no chunking overhead. Accuracy improvements are applied via
request parameters (temperature, language hint, subject prompt) rather than
splitting audio, since splitting/retry logic added latency without being
needed for typical lecture-length videos.

Requires GROQ_API_KEY to be set as an environment variable.
"""

import json
import os
import re
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
load_dotenv()
from src.video_metadata import TranscriptSegment, VideoMetadata

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Matches sentence-ending punctuation for both English and Hindi (Devanagari "।")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def _build_prompt(meta: VideoMetadata) -> str:
    """
    Builds a transcription prompt ONLY if a subject was explicitly provided.
    With no subject, returns an empty string -- meaning no bias is applied
    at all, and Whisper transcribes purely based on the audio itself.
    """
    if meta.subject:
        return f"This is an educational lecture on {meta.subject}. Use correct terminology and punctuation."
    return ""


def transcribe_audio(
    audio_path: str,
    meta: VideoMetadata,
    output_dir: str = "output/transcripts",
    model: str = "whisper-large-v3",
    prompt: str = "",
    language: str = None,
    temperature: float = 0.2,
) -> List[TranscriptSegment]:
    """
    Sends a WAV/MP3 file to Groq's Whisper API in a single request and
    returns timestamped, sentence-reconstructed segments.

    Args:
        audio_path: path to the audio file from audio_extractor.py
        meta: VideoMetadata for the source video (used for doc_id/output naming)
        output_dir: folder to save the raw transcript JSON
        model: "whisper-large-v3-turbo" (fast) or "whisper-large-v3" (most accurate)
        prompt: optional context hint; auto-built from meta.subject if empty
        language: optional ISO language code (e.g. "hi" for Hindi). Giving an
                  explicit language for code-switched lecture audio (Hindi
                  narration with English technical terms) often improves
                  accuracy over auto-detection, since the model isn't
                  re-guessing the primary language segment by segment.
        temperature: decoding temperature. A small non-zero value (default
                     0.2) tends to reduce repetition-loop artifacts compared
                     to temperature=0's fully greedy decoding, at effectively
                     no cost to speed.

    Returns:
        list of TranscriptSegment objects, sorted by start time
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable not set. "
            "Run: $env:GROQ_API_KEY = 'your_key_here'"
        )

    if not prompt:
        prompt = _build_prompt(meta)

    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    size_mb = audio_file.stat().st_size / (1024 * 1024)
    if size_mb > 25:
        raise ValueError(
            f"Audio file is {size_mb:.1f}MB, Groq's limit is 25MB. "
            f"Use a shorter video or compress the audio further."
        )

    print(f"[transcriber] Sending {audio_file.name} to Groq ({model})...")

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
        "temperature": temperature,
    }
    if prompt:
        data["prompt"] = prompt
    if language:
        data["language"] = language

    mime_type = "audio/mpeg" if audio_file.suffix == ".mp3" else "audio/wav"

    with open(audio_file, "rb") as f:
        files = {"file": (audio_file.name, f, mime_type)}
        response = requests.post(GROQ_URL, headers=headers, data=data, files=files, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(f"Groq request failed ({response.status_code}): {response.text}")

    result = response.json()
    raw_segments = _parse_groq_response(result)
    segments = _merge_into_sentences(raw_segments)

    print(f"[transcriber] Produced {len(segments)} segments")
    for s in segments:
        print(f"  [{s.start:6.1f}s -> {s.end:6.1f}s] {s.text}")

    _save_transcript(segments, meta, output_dir)
    return segments


def _merge_into_sentences(raw_segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """
    Whisper-family models decode audio in fixed ~30s internal chunks, so a
    sentence that starts near the end of one chunk continues in the next
    segment's text. This stitches all segment text into one stream, then
    re-splits on real sentence boundaries (., !, ?, Hindi ।), assigning each
    reconstructed sentence a start/end time interpolated from the original
    segment timings based on character position.
    """
    if not raw_segments:
        return []

    full_text = ""
    char_times: List[float] = []

    for seg in raw_segments:
        piece = seg.text.strip()
        if not piece:
            continue
        if full_text:
            full_text += " "
            char_times.append(seg.start)
        seg_start_index = len(full_text)
        full_text += piece
        duration = max(seg.end - seg.start, 0.01)
        for i in range(len(piece)):
            frac = i / max(len(piece) - 1, 1)
            char_times.append(seg.start + frac * duration)

    raw_sentences = _SENTENCE_END.split(full_text)

    merged: List[TranscriptSegment] = []
    cursor = 0
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        start_idx = full_text.find(sentence, cursor)
        if start_idx == -1:
            start_idx = cursor
        end_idx = start_idx + len(sentence) - 1
        cursor = end_idx + 1

        start_idx = min(start_idx, len(char_times) - 1)
        end_idx = min(end_idx, len(char_times) - 1)

        merged.append(
            TranscriptSegment(
                text=sentence,
                start=char_times[start_idx],
                end=char_times[end_idx],
                source="audio",
            )
        )

    return merged


def _parse_groq_response(result: dict) -> List[TranscriptSegment]:
    segments: List[TranscriptSegment] = []
    raw_segments = result.get("segments", [])
    if raw_segments:
        for seg in raw_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(text=text, start=seg["start"], end=seg["end"], source="audio")
            )
    else:
        text = result.get("text", "").strip()
        if text:
            segments.append(TranscriptSegment(text=text, start=0.0, end=0.0, source="audio"))
    return segments


def _save_transcript(
    segments: List[TranscriptSegment], meta: VideoMetadata, output_dir: str
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{meta.doc_id}.json"
    data = [
        {"text": s.text, "start": s.start, "end": s.end, "source": s.source}
        for s in segments
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[transcriber] Saved transcript: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.transcriber <path_to_audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    test_meta = VideoMetadata(video_path=audio_file)
    transcribe_audio(audio_file, test_meta)