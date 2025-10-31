# Usage Examples

This document provides detailed examples of using the transcription CLI tool.

## Prerequisites

Before using the tool, ensure you have:

1. Python 3.8 or higher installed
2. ffmpeg installed on your system
3. Python dependencies installed: `pip install -r requirements.txt`

## Basic Examples

### 1. Transcribe a Single Audio File

```bash
python transcribe_whisper.py interview.mp3
```

Output:
- Creates `interview.md` in the same directory as `interview.mp3`
- Uses the default `base` model
- Auto-detects language
- Uses GPU if available, otherwise CPU

### 2. Transcribe a Single Video File

```bash
python transcribe_whisper.py customer_feedback.mp4
```

Works the same as audio files - automatically extracts and transcribes the audio track.

### 3. Transcribe All Files in a Directory (Recursive)

```bash
python transcribe_whisper.py /path/to/interviews/
```

- Processes all supported audio/video files in the directory and subdirectories
- Creates `.md` files next to each source file
- Shows progress for each file
- Displays summary table at the end

### 4. Include Timestamps

```bash
python transcribe_whisper.py --timestamps interview.mp4
```

Output includes timestamped segments:

```markdown
## Timestamped Segments

**[00:00:00.000 - 00:00:05.120]** Hello and welcome to today's interview.
**[00:00:05.120 - 00:00:12.450]** We're here to discuss your experience with our product.
```

### 5. Overwrite Existing Transcriptions

```bash
python transcribe_whisper.py --overwrite interview.mp4
```

By default, the tool won't overwrite existing `.md` files. Use `--overwrite` to force overwrite.

### 6. Use a Larger Model for Better Accuracy

```bash
python transcribe_whisper.py --model medium interview.mp4
```

Available models:
- `tiny` - Fastest, least accurate (~1GB VRAM)
- `base` - Good balance (default) (~1GB VRAM)
- `small` - Better accuracy (~2GB VRAM)
- `medium` - High accuracy (~5GB VRAM)
- `large` - Best accuracy (~10GB VRAM)

### 7. Specify Language (Skip Auto-Detection)

```bash
python transcribe_whisper.py --language en interview.mp4
```

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `zh` - Chinese
- `ja` - Japanese

### 8. Process Directory Without Recursion

```bash
python transcribe_whisper.py --non-recursive /path/to/interviews/
```

Only processes files directly in the specified directory, ignoring subdirectories.

### 9. Batch Process Multiple Files

```bash
python transcribe_whisper.py interview1.mp3 interview2.mp4 feedback.wav
```

Processes all specified files in sequence.

### 10. Complex Example - Full Featured

```bash
python transcribe_whisper.py \
  --timestamps \
  --model medium \
  --language en \
  --overwrite \
  /path/to/customer_interviews/
```

This command:
- Processes all files in `/path/to/customer_interviews/` recursively
- Uses the `medium` model for better accuracy
- Forces English language (faster than auto-detection)
- Includes timestamps in output
- Overwrites any existing transcription files

## Expected Output Format

For a file named `interview.mp4`, the tool creates `interview.md`:

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

[Complete transcription of the audio content goes here...]

## Timestamped Segments

**[00:00:00.000 - 00:00:05.120]** Hello and welcome to today's interview.

**[00:00:05.120 - 00:00:12.450]** We're here to discuss your experience...
```

## Summary Table Example

After processing multiple files, you'll see a summary:

```
================================================================================
TRANSCRIPTION SUMMARY
================================================================================

Total files processed: 5
Successful: 4
Failed: 1

Detailed Results:
--------------------------------------------------------------------------------
✓ interview1.mp3
  → /path/to/interview1.md
✓ interview2.mp4
  → /path/to/interview2.md
✓ feedback.wav
  → /path/to/feedback.md
✗ corrupted.mp4
  → Error: Failed to decode audio
✓ discussion.m4a
  → /path/to/discussion.md
================================================================================
```

## Performance Tips

1. **Start with smaller models**: Use `base` or `small` for initial testing
2. **Use GPU**: Significantly faster than CPU (10-100x depending on hardware)
3. **Specify language**: If you know the language, use `--language` to skip detection
4. **Batch processing**: Process multiple files in one command for efficiency
5. **Watch disk space**: Transcription files are small, but processing uses temporary space

## Troubleshooting

### Issue: "No module named 'whisper'"
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: "ffmpeg not found"
**Solution**: Install ffmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Windows: Download from ffmpeg.org

### Issue: Out of memory
**Solution**: Use a smaller model (`--model tiny` or `--model base`)

### Issue: Very slow on CPU
**Solution**: Expected behavior. Consider:
- Using a smaller model
- Processing fewer/shorter files
- Upgrading to a system with GPU

### Issue: Wrong language detected
**Solution**: Use `--language` flag to specify the correct language

## Supported File Formats

### Audio
MP3, WAV, M4A, FLAC, AAC, OGG, WMA, OPUS

### Video
MP4, MOV, AVI, MKV, FLV, WMV, WEBM, M4V
