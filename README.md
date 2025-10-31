# ai-transcribe

A local Python CLI tool for transcribing customer interview audio and video files using OpenAI Whisper.

## Features

- **GPU Acceleration**: Automatically detects and uses CUDA GPU if available, falls back to CPU
- **Flexible Input**: Process single files, multiple files, or entire directories (recursive search)
- **Multiple Formats**: Supports common audio (mp3, wav, m4a, flac, aac, ogg, wma) and video formats (mp4, mkv, avi, mov, wmv, flv, webm)
- **Markdown Output**: Generates `.md` files next to original files with YAML frontmatter metadata
- **Timestamps**: Optional segment timestamps in `[HH:MM:SS]` format
- **Smart Skipping**: Avoids re-processing files unless `--overwrite` is specified
- **Summary Statistics**: Displays detailed summary table with success/failure counts and total duration

## Installation

### Prerequisites

- Python 3.8 or higher
- ffmpeg (required by Whisper for audio processing)

### Install ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `choco install ffmpeg`

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

**For GPU support (CUDA):**
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Whisper Model Downloads

**Important:** Whisper models are NOT included in the pip installation. They are automatically downloaded on first use:

- Models are downloaded from OpenAI's servers when you first run the tool
- Cached locally in `~/.cache/whisper/` (macOS/Linux) or `C:\Users\<username>\.cache\whisper\` (Windows)
- Only the models you use are downloaded (not all 5 models)
- Subsequent runs use the cached models (no re-download)

**Model sizes:**
- `tiny`: ~72 MB
- `base`: ~142 MB (default)
- `small`: ~466 MB
- `medium`: ~1.5 GB
- `large`: ~2.9 GB

**First run example:**
```bash
python transcribe.py interview.mp4
# Downloads base model (~142 MB) on first run, then transcribes
# Future runs skip the download and start transcribing immediately
```

## Usage

### Basic Usage

Transcribe a single file:
```bash
python transcribe.py interview.mp4
```

Transcribe multiple files:
```bash
python transcribe.py file1.mp3 file2.wav file3.mp4
```

Transcribe all media files in a directory (recursive):
```bash
python transcribe.py /path/to/interviews/
```

### Command-Line Options

```bash
python transcribe.py [paths...] [options]
```

**Positional Arguments:**
- `paths`: One or more files or directories to process

**Optional Arguments:**
- `--model {tiny,base,small,medium,large}`: Whisper model to use (default: `base`)
  - `tiny`: Fastest, least accurate (~1GB RAM)
  - `base`: Good balance (default, ~1GB RAM)
  - `small`: Better accuracy (~2GB RAM)
  - `medium`: High accuracy (~5GB RAM)
  - `large`: Best accuracy (~10GB RAM)

- `--language LANG`: Language code (e.g., `en`, `es`, `fr`). Auto-detects if not specified

- `--timestamps`: Prefix each line with segment start time in `[HH:MM:SS]` format

- `--overwrite`: Overwrite existing `.md` files (by default, existing files are skipped)

- `--workers N`: Number of parallel workers (currently processes sequentially)

- `--extensions EXT [EXT ...]`: File extensions to process (default: common audio/video formats)

### Examples

Transcribe with timestamps and specific model:
```bash
python transcribe.py interviews/ --model small --timestamps
```

Transcribe in Spanish, overwriting existing files:
```bash
python transcribe.py *.mp3 --language es --overwrite
```

Process only MP4 files:
```bash
python transcribe.py videos/ --extensions .mp4
```

## Output Format

Each transcribed file generates a Markdown file with the same name and location:

**Input:** `/path/to/interview.mp4`  
**Output:** `/path/to/interview.md`

### Example Output

```markdown
---
source: /absolute/path/to/interview.mp4
model: base
language: en
duration_seconds: 245.60
created: 2025-10-31T10:30:45.123456
---

This is the transcribed text from the audio file.
Multiple paragraphs are preserved as spoken.
```

### With Timestamps

```markdown
---
source: /absolute/path/to/interview.mp4
model: base
language: en
duration_seconds: 245.60
created: 2025-10-31T10:30:45.123456
---

[00:00:00] This is the first segment of transcribed text.
[00:00:05] This is the second segment that starts at 5 seconds.
[00:00:12] And so on for each segment detected by Whisper.
```

## Summary Output

After processing, a summary table is displayed:

```
============================================================
TRANSCRIPTION SUMMARY
============================================================
Files Processed:          15
Succeeded:                12
Failed:                   1
Skipped:                  2
Total Audio Duration:     01:23:45 (5025.0s)
============================================================
```

## Troubleshooting

### CUDA Out of Memory
If you encounter GPU memory errors, try:
1. Use a smaller model: `--model tiny` or `--model base`
2. Force CPU usage by setting: `export CUDA_VISIBLE_DEVICES=""`

### ffmpeg Not Found
Ensure ffmpeg is installed and in your PATH:
```bash
ffmpeg -version
```

### Module Not Found
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## License

MIT License - See LICENSE file for details
