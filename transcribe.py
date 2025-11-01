#!/usr/bin/env python3
"""
Audio/Video Transcription CLI Tool
Transcribes media files using OpenAI Whisper and outputs to Markdown files.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple
import json

import torch
import whisper


# Constants
DEFAULT_MODEL = "base"
DEFAULT_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma",
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"
}
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


@dataclass
class TranscriptionResult:
    """Result of a single file transcription."""
    file_path: Path
    success: bool
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class TranscriptionStats:
    """Summary statistics for the transcription session."""
    total_files: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration: float = 0.0


def setup_logging() -> None:
    """Configure logging with a clean format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def detect_device() -> str:
    """Detect if CUDA is available and return appropriate device."""
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"CUDA detected. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logging.info("CUDA not available. Using CPU")
    return device


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s interview.mp4
  %(prog)s /path/to/interviews/ --model small --timestamps
  %(prog)s file1.mp3 file2.wav dir/ --overwrite --language en
        """
    )
    
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more files or directories to process (directories are searched recursively)"
    )
    
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=WHISPER_MODELS,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g., 'en', 'es', 'fr'). Auto-detect if not specified"
    )
    
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Prefix each line with segment start time in [HH:MM:SS] format"
    )
    
    # Paragraph formatting options
    parser.add_argument(
        "--paragraphs",
        dest="paragraphs",
        action="store_true",
        default=True,
        help="Group segments into paragraphs separated by silence gaps (default: on)"
    )
    parser.add_argument(
        "--no-paragraphs",
        dest="paragraphs",
        action="store_false",
        help="Disable paragraph grouping; output continuous text"
    )
    parser.add_argument(
        "--paragraph-gap-seconds",
        type=float,
        default=2.0,
        help="Silence duration (in seconds) that starts a new paragraph when grouping segments (default: 2.0)"
    )
    
    # Default behavior is to overwrite existing outputs; this flag lets users opt out
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have a corresponding .md output (default behavior overwrites)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where .md files will be written. "
            "If provided, directory structure under provided input folders is preserved. "
            "If not provided, .md files are written next to their source files."
        )
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (currently processes sequentially)"
    )
    
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help=f"File extensions to process (default: common audio/video formats)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and planned output paths without performing transcription"
    )
    
    return parser.parse_args()


def find_media_files(paths: List[Path], extensions: Set[str]) -> List[Path]:
    """
    Recursively find all media files from given paths.
    
    Args:
        paths: List of file or directory paths
        extensions: Set of file extensions to include (with leading dot)
    
    Returns:
        List of Path objects for all discovered media files
    """
    media_files: List[Path] = []
    extensions_lower = {ext.lower() for ext in extensions}
    
    for path in paths:
        if not path.exists():
            logging.warning(f"Path does not exist: {path}")
            continue
        
        if path.is_file():
            if path.suffix.lower() in extensions_lower:
                media_files.append(path)
            else:
                logging.warning(f"Skipping file with unsupported extension: {path}")
        elif path.is_dir():
            # Recursively search directory
            for ext in extensions_lower:
                media_files.extend(path.rglob(f"*{ext}"))
                media_files.extend(path.rglob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    media_files = sorted(set(media_files))
    return media_files


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to [HH:MM:SS] format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def generate_markdown(
    file_path: Path,
    transcription: dict,
    model_name: str,
    include_timestamps: bool,
    enable_paragraphs: bool,
    paragraph_gap_seconds: float
) -> str:
    """
    Generate Markdown content with YAML frontmatter.
    
    Args:
        file_path: Path to the original media file
        transcription: Whisper transcription result
        model_name: Name of the Whisper model used
        include_timestamps: Whether to include timestamps in output
    
    Returns:
        Formatted Markdown string
    """
    # Calculate duration from segments
    duration = 0.0
    if transcription.get("segments"):
        last_segment = transcription["segments"][-1]
        duration = last_segment.get("end", 0.0)
    
    # Detected language
    language = transcription.get("language", "unknown")
    
    # Current timestamp
    created_at = datetime.now().isoformat()
    
    # Build YAML frontmatter
    frontmatter = f"""---
source: {file_path.absolute()}
model: {model_name}
language: {language}
duration_seconds: {duration:.2f}
created: {created_at}
---

"""
    
    # Build transcription text
    segments = transcription.get("segments") or []
    if not segments:
        # Fallback to flat text when segments are missing
        transcription_text = transcription.get("text", "").strip()
    else:
        # Optionally group segments into paragraphs based on silence gaps
        if enable_paragraphs:
            paragraphs = []  # list[list[dict]]
            current_group = []
            last_end = None
            for seg in segments:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                if last_end is not None and (start - last_end) >= paragraph_gap_seconds:
                    if current_group:
                        paragraphs.append(current_group)
                        current_group = []
                current_group.append(seg)
                last_end = end
            if current_group:
                paragraphs.append(current_group)

            # Render paragraphs
            lines = []
            if include_timestamps:
                # Keep per-segment timestamp lines; insert blank line between paragraphs
                for i, group in enumerate(paragraphs):
                    if i > 0:
                        lines.append("")  # blank line between paragraphs
                    for seg in group:
                        timestamp = format_timestamp(float(seg.get("start", 0.0)))
                        text = str(seg.get("text", "")).strip()
                        if text:
                            lines.append(f"{timestamp} {text}")
                transcription_text = "\n".join(lines)
            else:
                # Join segment texts within a paragraph with spaces, blank line between paragraphs
                for i, group in enumerate(paragraphs):
                    if i > 0:
                        lines.append("")
                    paragraph_text = " ".join(str(s.get("text", "")).strip() for s in group if str(s.get("text", "")).strip())
                    if paragraph_text:
                        lines.append(paragraph_text)
                transcription_text = "\n".join(lines)
        else:
            # No paragraph grouping: either lines with timestamps or flat text
            if include_timestamps:
                lines = []
                for segment in segments:
                    timestamp = format_timestamp(float(segment.get("start", 0.0)))
                    text = str(segment.get("text", "")).strip()
                    if text:
                        lines.append(f"{timestamp} {text}")
                transcription_text = "\n".join(lines)
            else:
                # Join all segments into one block of text
                transcription_text = " ".join(str(s.get("text", "")).strip() for s in segments if str(s.get("text", "")).strip())
    
    return frontmatter + transcription_text + "\n"


def transcribe_file(
    file_path: Path,
    model: whisper.Whisper,
    model_name: str,
    language: Optional[str],
    include_timestamps: bool,
    skip_existing: bool,
    output_path: Optional[Path] = None,
    enable_paragraphs: bool = True,
    paragraph_gap_seconds: float = 2.0
) -> TranscriptionResult:
    """
    Transcribe a single media file.
    
    Args:
        file_path: Path to the media file
        model: Loaded Whisper model
        model_name: Name of the model
        language: Target language code or None for auto-detect
        include_timestamps: Whether to include timestamps
        overwrite: Whether to overwrite existing output files
    
    Returns:
        TranscriptionResult object
    """
    # Determine output path
    output_path = output_path or file_path.with_suffix(".md")
    # Ensure parent directory exists when writing outside of source directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output already exists
    if output_path.exists() and skip_existing:
        logging.info(f"Skipping {file_path.name} (output already exists)")
        return TranscriptionResult(file_path=file_path, success=False, error="already_exists")
    
    try:
        logging.info(f"Transcribing: {file_path}")

        # Transcribe with Whisper
        transcribe_options = {"verbose": False}
        if language:
            transcribe_options["language"] = language

        result = model.transcribe(str(file_path), **transcribe_options)

        # Generate Markdown content
        markdown_content = generate_markdown(
            file_path=file_path,
            transcription=result,
            model_name=model_name,
            include_timestamps=include_timestamps,
            enable_paragraphs=enable_paragraphs,
            paragraph_gap_seconds=paragraph_gap_seconds
        )

        # Write output file immediately
        output_path.write_text(markdown_content, encoding="utf-8")

        # Calculate duration
        duration = 0.0
        if result.get("segments"):
            last_segment = result["segments"][-1]
            duration = last_segment.get("end", 0.0)

        # Log and print a concise line showing where the transcription was saved
        logging.info(f"✓ Saved: {output_path} ({duration:.1f}s)")
        print(f"SAVED: {output_path}", flush=True)

        return TranscriptionResult(
            file_path=file_path,
            success=True,
            duration=duration
        )

    except Exception as e:
        logging.error(f"✗ Failed: {file_path} - {str(e)}")
        return TranscriptionResult(
            file_path=file_path,
            success=False,
            error=str(e)
        )


def compute_output_path(
    file_path: Path,
    output_dir: Optional[Path],
    input_roots: List[Path]
) -> Path:
    """
    Compute the output .md path for a given media file.
    
    - If output_dir is None, place the .md next to the source file.
    - If output_dir is provided, attempt to preserve the relative path
      under the most specific input root directory provided by the user.
      For files that don't reside under any input root (e.g., files passed
      directly), place them at the top level of output_dir.
    """
    if output_dir is None:
        return file_path.with_suffix(".md")

    src_resolved = file_path.resolve()
    # Pick the deepest matching root (most specific)
    best_root: Optional[Tuple[int, Path]] = None
    for root in input_roots:
        try:
            rel = src_resolved.relative_to(root)
            depth = len(rel.parts)
            if best_root is None or depth < best_root[0]:
                best_root = (depth, root)
        except Exception:
            continue

    if best_root is not None:
        _, root = best_root
        rel = src_resolved.relative_to(root)
        return (output_dir / rel).with_suffix(".md")

    # Fallback for direct file inputs not under any provided folder
    return (output_dir / src_resolved.name).with_suffix(".md")


def print_summary(stats: TranscriptionStats) -> None:
    """
    Print a formatted summary table of transcription results.
    
    Args:
        stats: TranscriptionStats object with session statistics
    """
    print("\n" + "=" * 60)
    print("TRANSCRIPTION SUMMARY")
    print("=" * 60)
    
    # Format duration
    hours = int(stats.total_duration // 3600)
    minutes = int((stats.total_duration % 3600) // 60)
    seconds = int(stats.total_duration % 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Print table
    print(f"{'Files Processed:':<25} {stats.total_files}")
    print(f"{'Succeeded:':<25} {stats.succeeded}")
    print(f"{'Failed:':<25} {stats.failed}")
    print(f"{'Skipped:':<25} {stats.skipped}")
    print(f"{'Total Audio Duration:':<25} {duration_str} ({stats.total_duration:.1f}s)")
    print("=" * 60)


def main() -> None:
    """Main entry point for the CLI tool."""
    setup_logging()
    args = parse_arguments()
    
    # Detect device (GPU/CPU)
    device = detect_device()
    
    # Normalize extensions
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions}
    
    # Find all media files
    logging.info("Searching for media files...")
    media_files = find_media_files(args.paths, extensions)
    
    if not media_files:
        logging.error("No media files found")
        sys.exit(1)
    
    logging.info(f"Found {len(media_files)} media file(s)")
    
    # Prepare output directory and input roots if requested
    output_dir: Optional[Path] = args.output_dir.resolve() if args.output_dir else None
    input_roots: List[Path] = []
    for p in args.paths:
        try:
            if p.exists() and p.is_dir():
                input_roots.append(p.resolve())
        except Exception:
            # Ignore any problematic paths; they were already warned about earlier
            pass
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output root: {output_dir}")

    # If dry-run, only list planned operations
    if args.dry_run:
        for f in media_files:
            planned = compute_output_path(f, output_dir, input_roots)
            print(f"PLAN: {f} -> {planned}")
        print(f"Total files planned: {len(media_files)}")
        return
    
    # Load Whisper model
    logging.info(f"Loading Whisper model: {args.model}")
    try:
        model = whisper.load_model(args.model, device=device)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Process each file
    stats = TranscriptionStats()
    
    for file_path in media_files:
        planned_output = compute_output_path(file_path, output_dir, input_roots)
        result = transcribe_file(
            file_path=file_path,
            model=model,
            model_name=args.model,
            language=args.language,
            include_timestamps=args.timestamps,
            skip_existing=args.skip_existing,
            output_path=planned_output,
            enable_paragraphs=args.paragraphs,
            paragraph_gap_seconds=args.paragraph_gap_seconds
        )
        
        if result.error == "already_exists":
            stats.skipped += 1
        elif result.success:
            stats.succeeded += 1
            stats.total_duration += result.duration
        else:
            stats.failed += 1
        
        stats.total_files += 1
    
    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()
