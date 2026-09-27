"""
main.py

Entry point. Run:
    python main.py input/lecture.mp4
    python main.py input/lecture.mp4 --subject Biology
"""

import argparse
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Process a lecture video into a timestamped transcript.")
    parser.add_argument("video_path", help="Path to the input MP4 file")
    parser.add_argument(
        "--subject", default=None,
        help="Optional subject label (e.g. 'Biology', 'Physics') to improve transcription accuracy"
    )
    args = parser.parse_args()

    result = run_pipeline(args.video_path, subject=args.subject)

    print("\n--- Summary ---")
    print(f"doc_id: {result['doc_id']}")
    print(f"audio_path: {result['audio_path']}")
    print(f"segments produced: {result['num_segments']}")
    if result["segments"]:
        print("\nFirst segment:")
        s = result["segments"][0]
        print(f"  [{s.start:.1f}s - {s.end:.1f}s] {s.text}")


if __name__ == "__main__":
    main()