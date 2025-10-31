# AI Transcribe

A local Python CLI tool for transcribing audio and video files from customer interviews using OpenAI Whisper.

## Features

- 🎯 **Automatic GPU/CPU Detection**: Automatically uses GPU if available, falls back to CPU
- 📁 **Flexible Input**: Process individual files or entire directories recursively
- 🎬 **Multi-Format Support**: Works with common audio (MP3, WAV, M4A, FLAC, AAC, OGG, etc.) and video formats (MP4, MOV, AVI, MKV, etc.)
- 📝 **Markdown Output**: Generates clean Markdown files with YAML metadata next to original files
- ⏱️ **Timestamps**: Optional timestamped segments for detailed transcriptions
- 🔄 **Overwrite Control**: Safe mode prevents accidental overwrites unless specified
- 📊 **Summary Table**: Displays processing results summary after completion
- 🌍 **Multi-Language**: Supports automatic language detection or manual specification

## Installation

### Prerequisites

- Python 3.8 or higher
- ffmpeg (required for audio processing)

#### Install ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use chocolatey:
```bash
choco install ffmpeg
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Transcribe a single file:
```bash
python transcribe_whisper.py interview.mp4
```

Transcribe all files in a directory (recursive):
```bash
python transcribe_whisper.py /path/to/interviews/
```

Transcribe multiple files:
```bash
python transcribe_whisper.py file1.mp3 file2.wav interview.mp4
```

### Options

```
positional arguments:
  paths                 Path(s) to audio/video file(s) or directory/directories

optional arguments:
  -h, --help           show this help message and exit
  --timestamps         Include timestamps for each segment in output
  --overwrite          Overwrite existing transcription files
  --model {tiny,base,small,medium,large}
                       Whisper model size (default: base)
  --language LANGUAGE  Language code (e.g., en, es, fr). Auto-detected if not specified.
  --non-recursive      Do not process directories recursively
```

### Examples

**Include timestamps in transcription:**
```bash
python transcribe_whisper.py --timestamps interview.mp4
```

**Use a larger model for better accuracy:**
```bash
python transcribe_whisper.py --model medium interview.mp4
```

**Specify language (skip auto-detection):**
```bash
python transcribe_whisper.py --language en interview.mp4
```

**Overwrite existing transcriptions:**
```bash
python transcribe_whisper.py --overwrite /path/to/interviews/
```

**Process directory without recursion:**
```bash
python transcribe_whisper.py --non-recursive /path/to/interviews/
```

## Output Format

Transcriptions are saved as Markdown files (`.md`) in the same directory as the source file.

### Example Output

For a file named `interview.mp4`, the output `interview.md` will look like:

```markdown
---
source_file: interview.mp4
transcription_date: 2025-10-31T19:30:00.123456
file_size_mb: 45.23
language: en
model: whisper
---

# Transcription: interview

## Full Text

[Full transcription text here...]

## Timestamped Segments

**[00:00:00.000 - 00:00:05.120]** Hello and welcome to today's interview.

**[00:00:05.120 - 00:00:12.450]** We're here to discuss...
```

## Model Sizes

| Model  | Parameters | English-only | Multilingual | Required VRAM | Relative Speed |
|--------|------------|--------------|--------------|---------------|----------------|
| tiny   | 39 M       | ✓            | ✓            | ~1 GB         | ~32x           |
| base   | 74 M       | ✓            | ✓            | ~1 GB         | ~16x           |
| small  | 244 M      | ✓            | ✓            | ~2 GB         | ~6x            |
| medium | 769 M      | ✓            | ✓            | ~5 GB         | ~2x            |
| large  | 1550 M     | ✗            | ✓            | ~10 GB        | 1x             |

**Recommendation**: Start with `base` for a good balance of speed and accuracy. Use `small` or `medium` for better accuracy with longer files.

## Supported Formats

### Audio Formats
- MP3, WAV, M4A, FLAC, AAC, OGG, WMA, OPUS

### Video Formats
- MP4, MOV, AVI, MKV, FLV, WMV, WEBM, M4V

## Troubleshooting

**Issue**: "No module named 'whisper'"
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: "ffmpeg not found"
- **Solution**: Install ffmpeg using the instructions in the Installation section

**Issue**: Out of memory errors
- **Solution**: Use a smaller model (e.g., `--model tiny` or `--model base`)

**Issue**: Slow processing on CPU
- **Solution**: This is expected. Consider using a smaller model or upgrading to a GPU

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
