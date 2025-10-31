#!/usr/bin/env python3
"""
Audio/Video Transcription CLI Tool using OpenAI Whisper

This tool transcribes audio and video files using OpenAI's Whisper model.
It automatically detects GPU/CPU availability and can process individual files
or entire directories recursively.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import whisper
    import torch
except ImportError as e:
    print(f"Error: Required dependencies not installed. Run: pip install -r requirements.txt")
    print(f"Details: {e}")
    sys.exit(1)


# Supported audio and video file extensions
SUPPORTED_EXTENSIONS = {
    # Audio formats
    '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.opus',
    # Video formats
    '.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.m4v'
}


def detect_device() -> str:
    """
    Auto-detect the best available device (GPU/CPU) for processing.
    
    Returns:
        str: 'cuda' if GPU is available, otherwise 'cpu'
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("✓ Using CPU (no GPU detected)")
    return device


def find_media_files(path: Path, recursive: bool = True) -> List[Path]:
    """
    Find all supported media files in the given path.
    
    Args:
        path: File or directory path
        recursive: Whether to search directories recursively
        
    Returns:
        List of Path objects for supported media files
    """
    files = []
    
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        else:
            print(f"Warning: Unsupported file format: {path}")
    elif path.is_dir():
        if recursive:
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(path.rglob(f"*{ext}"))
        else:
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(path.glob(f"*{ext}"))
    else:
        print(f"Error: Path not found: {path}")
    
    return sorted(files)


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS.mmm format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def transcribe_file(
    file_path: Path,
    model: whisper.Whisper,
    include_timestamps: bool = False,
    language: Optional[str] = None
) -> Dict:
    """
    Transcribe a single audio/video file.
    
    Args:
        file_path: Path to the media file
        model: Loaded Whisper model
        include_timestamps: Whether to include timestamps in output
        language: Optional language code (e.g., 'en', 'es')
        
    Returns:
        Dictionary containing transcription results
    """
    print(f"Processing: {file_path.name}")
    
    try:
        # Transcribe the audio
        result = model.transcribe(
            str(file_path),
            language=language,
            verbose=False
        )
        
        return {
            'success': True,
            'text': result['text'].strip(),
            'segments': result.get('segments', []) if include_timestamps else [],
            'language': result.get('language', 'unknown')
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def generate_markdown_output(
    file_path: Path,
    transcription: Dict,
    include_timestamps: bool = False
) -> str:
    """
    Generate Markdown content with YAML frontmatter.
    
    Args:
        file_path: Original media file path
        transcription: Transcription results dictionary
        include_timestamps: Whether to include timestamps
        
    Returns:
        Markdown formatted string with YAML metadata
    """
    # Get file statistics
    file_stats = file_path.stat()
    file_size_mb = file_stats.st_size / (1024 * 1024)
    
    # Build YAML frontmatter
    yaml_lines = [
        "---",
        f"source_file: {file_path.name}",
        f"transcription_date: {datetime.now().isoformat()}",
        f"file_size_mb: {file_size_mb:.2f}",
        f"language: {transcription.get('language', 'unknown')}",
        f"model: whisper",
        "---",
        ""
    ]
    
    # Build content
    content_lines = [
        f"# Transcription: {file_path.stem}",
        "",
        "## Full Text",
        "",
        transcription['text'],
        ""
    ]
    
    # Add timestamps section if requested
    if include_timestamps and transcription.get('segments'):
        content_lines.extend([
            "## Timestamped Segments",
            ""
        ])
        
        for segment in transcription['segments']:
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            content_lines.append(f"**[{start} - {end}]** {text}")
            content_lines.append("")
    
    return "\n".join(yaml_lines + content_lines)


def save_transcription(
    file_path: Path,
    transcription: Dict,
    include_timestamps: bool = False,
    overwrite: bool = False
) -> Tuple[bool, str]:
    """
    Save transcription to a Markdown file next to the original.
    
    Args:
        file_path: Original media file path
        transcription: Transcription results
        include_timestamps: Whether to include timestamps
        overwrite: Whether to overwrite existing files
        
    Returns:
        Tuple of (success: bool, output_path: str)
    """
    # Create output filename
    output_path = file_path.with_suffix('.md')
    
    # Check if file exists
    if output_path.exists() and not overwrite:
        return False, f"File exists (use --overwrite): {output_path}"
    
    try:
        # Generate and write content
        content = generate_markdown_output(file_path, transcription, include_timestamps)
        output_path.write_text(content, encoding='utf-8')
        return True, str(output_path)
    except Exception as e:
        return False, f"Error writing file: {e}"


def print_summary_table(results: List[Dict]) -> None:
    """
    Print a summary table of all transcription results.
    
    Args:
        results: List of result dictionaries
    """
    print("\n" + "=" * 80)
    print("TRANSCRIPTION SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\nTotal files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if results:
        print("\nDetailed Results:")
        print("-" * 80)
        
        for result in results:
            status = "✓" if result['success'] else "✗"
            file_name = result['file']
            
            if result['success']:
                output = result.get('output', 'N/A')
                print(f"{status} {file_name}")
                print(f"  → {output}")
            else:
                error = result.get('error', 'Unknown error')
                print(f"{status} {file_name}")
                print(f"  → Error: {error}")
    
    print("=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a single file
  python transcribe_whisper.py interview.mp4

  # Transcribe all files in a directory (recursive)
  python transcribe_whisper.py /path/to/interviews/

  # Include timestamps in output
  python transcribe_whisper.py --timestamps interview.mp4

  # Overwrite existing transcriptions
  python transcribe_whisper.py --overwrite *.mp3

  # Use specific model size
  python transcribe_whisper.py --model medium interview.wav
        """
    )
    
    parser.add_argument(
        'paths',
        nargs='+',
        type=str,
        help='Path(s) to audio/video file(s) or directory/directories'
    )
    
    parser.add_argument(
        '--timestamps',
        action='store_true',
        help='Include timestamps for each segment in output'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing transcription files'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default=None,
        help='Language code (e.g., en, es, fr). Auto-detected if not specified.'
    )
    
    parser.add_argument(
        '--non-recursive',
        action='store_true',
        help='Do not process directories recursively'
    )
    
    args = parser.parse_args()
    
    # Collect all media files
    all_files = []
    for path_str in args.paths:
        path = Path(path_str).resolve()
        files = find_media_files(path, recursive=not args.non_recursive)
        all_files.extend(files)
    
    if not all_files:
        print("Error: No supported media files found.")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)
    
    print(f"\nFound {len(all_files)} file(s) to process")
    
    # Detect device
    device = detect_device()
    
    # Load Whisper model
    print(f"Loading Whisper model '{args.model}'...")
    try:
        model = whisper.load_model(args.model, device=device)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Process each file
    results = []
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] {file_path.name}")
        print("-" * 60)
        
        # Transcribe
        transcription = transcribe_file(
            file_path,
            model,
            include_timestamps=args.timestamps,
            language=args.language
        )
        
        if transcription['success']:
            # Save transcription
            success, message = save_transcription(
                file_path,
                transcription,
                include_timestamps=args.timestamps,
                overwrite=args.overwrite
            )
            
            results.append({
                'file': file_path.name,
                'success': success,
                'output': message if success else None,
                'error': None if success else message
            })
            
            if success:
                print(f"✓ Saved to: {message}")
            else:
                print(f"✗ {message}")
        else:
            results.append({
                'file': file_path.name,
                'success': False,
                'error': transcription['error']
            })
            print(f"✗ Transcription failed: {transcription['error']}")
    
    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
