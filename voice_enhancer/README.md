# AI Music Producer Tool

This tool processes raw vocal recordings (Audio or Video) and generates a complete music track.

## Setup

1.  Ensure you have Python 3.10+ installed.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    (Note: You may need to install `ffmpeg` on your system if it's not automatically handled by `imageio-ffmpeg`)

## Usage

Run the script pointing to your input file:

```bash
python main.py
```

By default, it looks for `IMG_2090.MOV` in the current directory. You can modify the `input_file` variable in `main.py` to process other files.

## Output

The tool generates the following files in the `output/` directory:

- `extracted_audio.wav`: Raw audio from video.
- `1_denoised.wav`: Audio after noise reduction.
- `2_enhanced_vocals.wav`: Vocals with EQ, Compression, and Reverb.
- `3_drums.wav`: Generated drum track.
- `3_bass.wav`: Generated bass track.
- `arrangement.mid`: MIDI file containing the generated Drums and Bass patterns.
- `final_mix.wav`: The final song.

## Features

- **Noise Reduction**: Spectral gating to remove background noise.
- **Analysis**: Detects BPM and Key.
- **Vocal Chain**: High-pass filter, Noise Gate, Compressor, Reverb, Limiter.
- **Generation**: Procedural generation of Drums and Bass based on detected tempo and key.
