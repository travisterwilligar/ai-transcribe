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
from typing import List, Optional, Set
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
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .md files"
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
    include_timestamps: bool
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
    if include_timestamps and transcription.get("segments"):
        lines = []
        for segment in transcription["segments"]:
            timestamp = format_timestamp(segment["start"])
            text = segment["text"].strip()
            lines.append(f"{timestamp} {text}")
        transcription_text = "\n".join(lines)
    else:
        transcription_text = transcription.get("text", "").strip()
    
    return frontmatter + transcription_text + "\n"


def transcribe_file(
    file_path: Path,
    model: whisper.Whisper,
    model_name: str,
    language: Optional[str],
    include_timestamps: bool,
    overwrite: bool
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
    output_path = file_path.with_suffix(".md")
    
    # Check if output already exists
    if output_path.exists() and not overwrite:
        logging.info(f"Skipping {file_path.name} (output already exists)")
        return TranscriptionResult(file_path=file_path, success=False, error="already_exists")
    
    try:
        logging.info(f"Transcribing: {file_path.name}")
        
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
            include_timestamps=include_timestamps
        )
        
        # Write output file
        output_path.write_text(markdown_content, encoding="utf-8")
        
        # Calculate duration
        duration = 0.0
        if result.get("segments"):
            last_segment = result["segments"][-1]
            duration = last_segment.get("end", 0.0)
        
        logging.info(f"✓ Completed: {output_path.name} ({duration:.1f}s)")
        
        return TranscriptionResult(
            file_path=file_path,
            success=True,
            duration=duration
        )
    
    except Exception as e:
        logging.error(f"✗ Failed: {file_path.name} - {str(e)}")
        return TranscriptionResult(
            file_path=file_path,
            success=False,
            error=str(e)
        )


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
        result = transcribe_file(
            file_path=file_path,
            model=model,
            model_name=args.model,
            language=args.language,
            include_timestamps=args.timestamps,
            overwrite=args.overwrite
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
