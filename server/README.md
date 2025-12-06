# Audio Autotune and Enhancement Tool

A Python tool that takes MP3 audio files and applies autotuning/pitch correction along with optional audio enhancements to improve sound quality.

## Features

- **Pitch Correction/Autotune**: Detects pitch and corrects it to the nearest note in a musical scale
- **Multiple Scales**: Supports chromatic, major, and minor scales
- **3-Band EQ**: Optional equalization for low, mid, and high frequencies
- **Audio Compression**: Optional dynamic range compression
- **Adjustable Strength**: Control how much pitch correction is applied

## Installation

1. Install Python 3.7 or higher
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: On some systems, you may need to install additional audio libraries:
- **Windows**: Usually works out of the box
- **Linux**: May need `sudo apt-get install libsndfile1`
- **macOS**: May need `brew install libsndfile`

## Usage

### Basic Autotune

```bash
python autotune.py input.mp3 output.wav
```

### Advanced Options

```bash
# Strong autotune with major scale
python autotune.py input.mp3 output.wav --strength 0.9 --scale major

# Autotune with EQ enhancement
python autotune.py input.mp3 output.wav --eq --low-gain 2 --high-gain 1

# Full enhancement (autotune + EQ + compression)
python autotune.py input.mp3 output.wav --strength 0.7 --eq --low-gain 1 --compress --threshold 0.6
```

### Command Line Arguments

**Autotune Parameters:**
- `--strength`: Autotune strength (0.0-1.0, default: 0.7)
  - 0.0 = no correction
  - 1.0 = full correction to nearest note
- `--scale`: Musical scale (`chromatic`, `major`, `minor`, default: `chromatic`)
- `--root-note`: Root note frequency in Hz (default: 440.0 for A4)

**EQ Parameters:**
- `--eq`: Enable 3-band equalization
- `--low-gain`: Low frequency gain in dB (default: 0.0)
- `--mid-gain`: Mid frequency gain in dB (default: 0.0)
- `--high-gain`: High frequency gain in dB (default: 0.0)

**Compression Parameters:**
- `--compress`: Enable audio compression
- `--threshold`: Compression threshold (0-1, default: 0.7)
- `--ratio`: Compression ratio (default: 4.0)

## How It Works

1. **Pitch Detection**: Uses autocorrelation to detect the fundamental frequency of the audio
2. **Pitch Correction**: Snaps detected pitches to the nearest note in the selected musical scale
3. **Pitch Shifting**: Uses phase vocoder technique to shift pitch while preserving audio quality
4. **Optional Enhancements**: Applies EQ and compression if enabled

## Tips

- For subtle correction, use `--strength 0.3-0.5`
- For strong autotune effect (like T-Pain style), use `--strength 0.8-1.0`
- Use `--scale major` or `--scale minor` for more musical-sounding results
- Output format: WAV is recommended for best quality, but other formats supported by soundfile will work

## Limitations

- Works best on monophonic audio (single voice/instrument)
- Processing time depends on audio length (longer files take more time)
- Very low or very high pitches may not be detected accurately

## Example Use Cases

- Vocal pitch correction
- Instrument tuning
- Creating autotune effects
- General audio enhancement

