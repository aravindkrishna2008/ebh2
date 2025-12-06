"""
Video to Audio Extractor
Takes a MOV (or other video) file and extracts the audio track to an audio file.
"""

import argparse
import os
import sys
from pathlib import Path


def extract_audio_from_video(input_file, output_file=None, audio_format='mp3', 
                             audio_codec=None, bitrate='192k', sample_rate=None):
    """
    Extract audio from a video file.
    
    Args:
        input_file: Path to input video file (MOV, MP4, etc.)
        output_file: Path to output audio file (if None, auto-generates from input)
        audio_format: Output audio format ('mp3', 'wav', 'm4a', 'aac', etc.)
        audio_codec: Audio codec to use (None = auto-detect based on format)
        bitrate: Audio bitrate (for compressed formats like MP3)
        sample_rate: Sample rate in Hz (None = keep original)
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        print("Error: moviepy is not installed.")
        print("Please install it with: pip install moviepy")
        sys.exit(1)
    
    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    # Auto-generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix(f'.{audio_format}'))
    
    print(f"Loading video file: {input_file}")
    
    try:
        # Load video file
        video = VideoFileClip(input_file)
        
        # Get video info
        duration = video.duration
        fps = video.fps
        print(f"Video info: {duration:.2f} seconds, {fps:.2f} fps")
        
        # Extract audio
        print("Extracting audio...")
        audio = video.audio
        
        if audio is None:
            print("Error: No audio track found in the video file.")
            video.close()
            return False
        
        # Get audio info
        audio_duration = audio.duration
        audio_fps = audio.fps
        print(f"Audio info: {audio_duration:.2f} seconds, {audio_fps:.0f} Hz sample rate")
        
        # Set codec based on format if not specified
        if audio_codec is None:
            codec_map = {
                'mp3': 'mp3',
                'wav': 'pcm_s16le',
                'm4a': 'aac',
                'aac': 'aac',
                'ogg': 'libvorbis',
                'flac': 'flac'
            }
            audio_codec = codec_map.get(audio_format.lower(), 'mp3')
        
        # Prepare write parameters
        write_params = {
            'codec': audio_codec,
        }
        
        # Add bitrate for compressed formats
        if audio_format.lower() in ['mp3', 'm4a', 'aac']:
            write_params['bitrate'] = bitrate
        
        # Resample if requested
        if sample_rate is not None and sample_rate != audio_fps:
            print(f"Resampling audio to {sample_rate} Hz...")
            audio = audio.set_fps(sample_rate)
            write_params['fps'] = sample_rate
        
        # Write audio file
        print(f"Writing audio to: {output_file}")
        audio.write_audiofile(
            output_file,
            **write_params
        )
        
        # Clean up
        audio.close()
        video.close()
        
        print(f"Successfully extracted audio to: {output_file}")
        
        # Show file size
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"Output file size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_audio_tracks(input_file):
    """
    List available audio tracks in a video file.
    
    Args:
        input_file: Path to input video file
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        print("Error: moviepy is not installed.")
        print("Please install it with: pip install moviepy")
        return
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    try:
        video = VideoFileClip(input_file)
        audio = video.audio
        
        if audio is None:
            print("No audio tracks found in the video file.")
        else:
            print(f"Audio track found:")
            print(f"  Duration: {audio.duration:.2f} seconds")
            print(f"  Sample rate: {audio.fps:.0f} Hz")
            print(f"  Channels: {audio.nchannels if hasattr(audio, 'nchannels') else 'Unknown'}")
        
        video.close()
        if audio:
            audio.close()
            
    except Exception as e:
        print(f"Error reading video: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio from MOV and other video files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract audio to MP3 (default)
  python extract_audio.py video.mov
  
  # Extract to specific output file
  python extract_audio.py video.mov -o output.mp3
  
  # Extract to WAV format
  python extract_audio.py video.mov -f wav
  
  # Extract with custom bitrate
  python extract_audio.py video.mov -b 320k
  
  # Extract with resampling
  python extract_audio.py video.mov -r 44100
  
  # List audio track info
  python extract_audio.py video.mov --info
        """
    )
    
    parser.add_argument('input', help='Input video file path (MOV, MP4, etc.)')
    parser.add_argument('-o', '--output', default=None,
                       help='Output audio file path (auto-generated if not specified)')
    parser.add_argument('-f', '--format', default='mp3',
                       choices=['mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'],
                       help='Output audio format (default: mp3)')
    parser.add_argument('-b', '--bitrate', default='192k',
                       help='Audio bitrate for compressed formats (default: 192k)')
    parser.add_argument('-r', '--sample-rate', type=int, default=None,
                       help='Sample rate in Hz (default: keep original)')
    parser.add_argument('-c', '--codec', default=None,
                       help='Audio codec (default: auto-detect based on format)')
    parser.add_argument('--info', action='store_true',
                       help='Show audio track information without extracting')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    # Show info only
    if args.info:
        list_audio_tracks(args.input)
        return
    
    # Extract audio
    success = extract_audio_from_video(
        args.input,
        output_file=args.output,
        audio_format=args.format,
        audio_codec=args.codec,
        bitrate=args.bitrate,
        sample_rate=args.sample_rate
    )
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

