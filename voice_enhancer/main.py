import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import noisereduce as nr
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError:
    from moviepy import VideoFileClip, AudioFileClip
from pedalboard import Pedalboard, NoiseGate, Compressor, HighpassFilter, Reverb, Limiter
from pedalboard.io import AudioFile
import scipy.signal
from midiutil import MIDIFile
import whisper
import torch
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv("voice_enhancer/.env")

class Synthesizer:
    def __init__(self, sr=44100):
        self.sr = sr

    def generate_sine(self, freq, duration, envelope=None):
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        wave = np.sin(2 * np.pi * freq * t)
        if envelope is not None:
            wave *= envelope
        return wave

    def generate_saw(self, freq, duration, envelope=None):
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        wave = scipy.signal.sawtooth(2 * np.pi * freq * t)
        if envelope is not None:
            wave *= envelope
        return wave

    def generate_square(self, freq, duration, envelope=None):
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        wave = scipy.signal.square(2 * np.pi * freq * t)
        if envelope is not None:
            wave *= envelope
        return wave

    def generate_noise(self, duration, envelope=None):
        wave = np.random.uniform(-1, 1, int(self.sr * duration))
        if envelope is not None:
            wave *= envelope
        return wave

    def adsr_envelope(self, duration, attack=0.01, decay=0.1, sustain=0.7, release=0.1):
        total_samples = int(self.sr * duration)
        attack_samples = int(self.sr * attack)
        decay_samples = int(self.sr * decay)
        release_samples = int(self.sr * release)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        if sustain_samples < 0:
            # Adjust if duration is too short
            factor = total_samples / (attack_samples + decay_samples + release_samples)
            attack_samples = int(attack_samples * factor)
            decay_samples = int(decay_samples * factor)
            release_samples = int(release_samples * factor)
            sustain_samples = 0

        envelope = np.concatenate([
            np.linspace(0, 1, attack_samples),
            np.linspace(1, sustain, decay_samples),
            np.full(sustain_samples, sustain),
            np.linspace(sustain, 0, release_samples)
        ])
        
        # Pad or trim to exact length due to rounding
        if len(envelope) < total_samples:
            envelope = np.pad(envelope, (0, total_samples - len(envelope)))
        else:
            envelope = envelope[:total_samples]
            
        return envelope

    def riser(self, duration=2.0):
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        noise = np.random.uniform(-1, 1, len(t))
        
        # Rising filter sweep (simulated with bandpass moving up)
        # Actually, simpler to just do amplitude rise + highpass sweep
        env = np.linspace(0, 1, len(t)) ** 2
        
        # Simple highpass sweep simulation
        # We'll just use white noise with rising amplitude for now as a placeholder
        # A real sweep needs time-varying filter which is complex in pure numpy without loop
        # Let's do a sine sweep + noise
        
        freq_sweep = np.linspace(200, 2000, len(t))
        sine = np.sin(2 * np.pi * freq_sweep * t) * 0.3
        
        return (noise * 0.7 + sine) * env

    def impact(self, duration=1.5):
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        # Sine drop
        freq = np.linspace(150, 40, len(t))
        sine = np.sin(2 * np.pi * freq * t)
        env = np.exp(-5 * t)
        
        # Noise burst
        noise = np.random.uniform(-1, 1, len(t)) * np.exp(-10 * t)
        
        return (sine + noise * 0.5) * env

    def bass_drop(self, duration=2.0):
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        # Sub bass sine sweep - go really low
        freq = np.linspace(120, 20, len(t)) 
        sine = np.sin(2 * np.pi * freq * t)
        
        # Add harmonics (square wave layer for "buzz" and texture)
        square = scipy.signal.square(2 * np.pi * freq * t) * 0.3
        
        # Envelope - sustain longer then fade
        attack_len = int(0.05 * self.sr)
        hold_len = int(0.5 * self.sr)
        decay_len = len(t) - attack_len - hold_len
        
        if decay_len < 0: # Safety for short durations
            env = np.linspace(1, 0, len(t))
        else:
            env = np.concatenate([
                np.linspace(0, 1, attack_len), # Fast attack
                np.linspace(1, 0.8, hold_len), # Hold
                np.linspace(0.8, 0, decay_len) # Long decay
            ])
        
        # Ensure lengths match due to rounding
        if len(env) < len(t):
            env = np.pad(env, (0, len(t) - len(env)))
        else:
            env = env[:len(t)]
            
        wave = (sine + square) * env
        
        # Heavy Distortion/Saturation
        wave = np.clip(wave * 5.0, -1, 1) 
        
        return wave

    def scratch(self, duration=0.5):
        # Simulate vinyl scratch (chirp)
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        
        # Forward-Backward motion simulation
        # Frequency sweep up and down rapidly
        freq_lfo = 5 + 10 * np.sin(2 * np.pi * (4/duration) * t) # Speed of scratch
        base_freq = 800
        
        # FM Synthesis for scratch texture
        carrier = np.sin(2 * np.pi * base_freq * t + 5 * np.sin(2 * np.pi * freq_lfo * t))
        
        # Noise layer for vinyl friction
        noise = np.random.uniform(-1, 1, len(t))
        b, a = scipy.signal.butter(2, [1000/(self.sr/2), 4000/(self.sr/2)], btype='band')
        noise = scipy.signal.lfilter(b, a, noise)
        
        # Envelope
        env = self.adsr_envelope(duration, 0.01, 0.1, 0.5, 0.1)
        
        return (carrier * 0.7 + noise * 0.4) * env

    def kick(self, style='pop', variation=0):
        duration = 0.4
        if style == 'rap':
            if variation == 0: # Deep 808 (TR-808 Style)
                duration = 0.6 
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                # 808 Kick Pitch Envelope: Fast drop from ~150Hz to ~50Hz
                freq = 50 + 100 * np.exp(-15 * t)
                wave = np.sin(2 * np.pi * freq * t)
                # 808 Amp Envelope: Fast attack, long decay
                env = np.exp(-3 * t) 
                # Saturation for punch
                wave = np.clip(wave * 2.0, -1, 1) 
            elif variation == 1: # Punchy Boom Bap (Sampled Style)
                duration = 0.3
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                freq = np.linspace(120, 50, len(t))
                wave = np.sin(2 * np.pi * freq * t)
                # Add some noise for "sampled" grit
                noise = self.generate_noise(duration) * 0.1
                b, a = scipy.signal.butter(2, 1000/(self.sr/2), btype='low')
                noise = scipy.signal.lfilter(b, a, noise)
                wave += noise
                env = np.exp(-10 * t)
                wave = np.clip(wave * 1.5, -1, 1)
            elif variation == 2: # Distorted 808
                duration = 0.5
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                freq = 50 + 80 * np.exp(-10 * t)
                wave = scipy.signal.square(2 * np.pi * freq * t) # Square for grit
                b, a = scipy.signal.butter(2, 150/(self.sr/2), btype='low')
                wave = scipy.signal.lfilter(b, a, wave)
                env = np.exp(-4 * t)
                wave = wave * env
            elif variation == 3: # Soft/Round (Jazz/Soul Kick)
                duration = 0.4
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                freq = np.linspace(80, 40, len(t))
                wave = np.sin(2 * np.pi * freq * t)
                env = np.exp(-5 * t)
            else: # Clicky
                duration = 0.3
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                freq = np.linspace(200, 50, len(t))
                wave = np.sin(2 * np.pi * freq * t)
                env = np.exp(-12 * t)
                click = self.generate_noise(0.01, self.adsr_envelope(0.01, 0.001, 0.005, 0, 0.004))
                wave[:len(click)] += click * 0.5
                
            return wave
            
        elif style == 'chill':
            duration = 0.3
            t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
            freq = np.linspace(120, 60, len(t))
            wave = np.sin(2 * np.pi * freq * t)
            env = np.exp(-8 * t)
            return wave * env
        else: # Pop
            t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
            freq = np.linspace(150, 50, len(t))
            wave = np.sin(2 * np.pi * freq * t)
            env = np.exp(-10 * t)
            return wave * env

    def snare(self, style='pop', variation=0):
        duration = 0.25
        if style == 'rap':
            if variation == 0: # TR-808 Snare (Tone + Noise)
                # Tone: Sine with pitch drop
                t_tone = np.linspace(0, 0.15, int(self.sr * 0.15), endpoint=False)
                freq = np.linspace(300, 150, len(t_tone))
                tone = np.sin(2 * np.pi * freq * t_tone) * np.exp(-15 * t_tone)
                
                # Noise: White noise with bandpass
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.001, 0.15, 0.0, 0.05))
                b, a = scipy.signal.butter(2, [1000/(self.sr/2), 4000/(self.sr/2)], btype='band')
                noise = scipy.signal.lfilter(b, a, noise)
                
                # Mix
                wave = np.zeros(int(self.sr * duration))
                wave[:len(tone)] += tone * 0.6
                wave[:len(noise)] += noise * 0.8
                return wave
                
            elif variation == 1: # Boom Bap Snare (Sampled Style)
                tone = self.generate_sine(180, duration, self.adsr_envelope(duration, 0.005, 0.1, 0, 0.05))
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.005, 0.2, 0, 0.05))
                b, a = scipy.signal.butter(2, 1000/(self.sr/2), btype='low') # Darker noise
                # Add bitcrush for lo-fi feel
                noise = noise[::2].repeat(2)[:len(noise)]
                return (tone * 0.6 + noise * 0.7)
            elif variation == 2: # Rimshot (808 Style)
                tone = self.generate_sine(500, 0.05, self.adsr_envelope(0.05, 0.001, 0.02, 0, 0.02))
                # Highpass click
                click = self.generate_noise(0.01)
                b, a = scipy.signal.butter(2, 5000/(self.sr/2), btype='high')
                click = scipy.signal.lfilter(b, a, click)
                tone[:len(click)] += click * 0.5
                return tone * 0.8
            elif variation == 3: # Snap
                noise = self.generate_noise(0.05, self.adsr_envelope(0.05, 0.001, 0.03, 0, 0.01))
                b, a = scipy.signal.butter(2, 2000/(self.sr/2), btype='high')
                noise = scipy.signal.lfilter(b, a, noise)
                return noise * 0.6
            else: # Industrial
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.001, 0.2, 0, 0.05))
                # Comb filter for metallic sound
                delay = int(0.005 * self.sr)
                comb = np.zeros_like(noise)
                for i in range(len(noise)):
                    if i >= delay:
                        comb[i] = noise[i] + 0.8 * comb[i-delay]
                    else:
                        comb[i] = noise[i]
                return comb * 0.5

        elif style == 'chill':
            # Rimshot / Soft snare
            tone = self.generate_sine(400, duration, self.adsr_envelope(duration, 0.001, 0.05, 0.0, 0.05))
            return tone * 0.4
        else: # Pop
            tone = self.generate_sine(180, duration, self.adsr_envelope(duration, 0.005, 0.1, 0.0, 0.1))
            noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.005, 0.15, 0.0, 0.05))
            return (tone * 0.5 + noise * 0.8)

    def hihat(self, style='pop', variation=0):
        duration = 0.1
        if style == 'rap':
            if variation == 0: # TR-808 Closed Hat
                duration = 0.05
                # 6 Square waves at specific frequencies
                freqs = [200, 300, 400, 600, 800, 1000] # Simplified metallic spectrum
                wave = np.zeros(int(self.sr * duration))
                for f in freqs:
                    wave += self.generate_square(f, duration)
                
                # Highpass filter
                b, a = scipy.signal.butter(2, 5000/(self.sr/2), btype='high')
                wave = scipy.signal.lfilter(b, a, wave)
                
                # Envelope
                env = self.adsr_envelope(duration, 0.001, 0.03, 0.0, 0.01)
                return wave * env * 0.6
                
            elif variation == 1: # Shaker (Sampled Style)
                duration = 0.1
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.02, 0.05, 0, 0.03))
                b, a = scipy.signal.butter(2, 3000/(self.sr/2), btype='high')
                noise = scipy.signal.lfilter(b, a, noise)
                return noise * 0.5
            elif variation == 2: # TR-808 Open Hat
                duration = 0.3
                # Similar to closed but longer decay
                freqs = [200, 300, 400, 600, 800, 1000]
                wave = np.zeros(int(self.sr * duration))
                for f in freqs:
                    wave += self.generate_square(f, duration)
                
                b, a = scipy.signal.butter(2, 4000/(self.sr/2), btype='high')
                wave = scipy.signal.lfilter(b, a, wave)
                
                env = self.adsr_envelope(duration, 0.01, 0.2, 0, 0.05)
                return wave * env * 0.5
            elif variation == 3: # Lo-fi Hat
                duration = 0.08
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.005, 0.05, 0, 0.02))
                # Bitcrush simulation (downsample)
                noise = noise[::4].repeat(4)[:len(noise)]
                return noise * 0.6
            else: # Metallic
                duration = 0.1
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.001, 0.08, 0, 0.01))
                # Ring mod
                t = np.linspace(0, duration, len(noise))
                mod = np.sin(2 * np.pi * 2000 * t)
                return noise * mod * 0.8

        elif style == 'chill':
            # Shaker-like
            noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.02, 0.05, 0.0, 0.03))
            b, a = scipy.signal.butter(2, 4000/(self.sr/2), btype='high')
            noise = scipy.signal.lfilter(b, a, noise)
            return noise * 0.3
        else: # Pop
            noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.001, 0.05, 0.0, 0.04))
            noise = np.diff(noise, prepend=0)
            return noise * 0.6

    def bass(self, freq, duration, style='pop', variation=0):
        if style == 'rap':
            if variation == 0: # 808 Sine Saturation
                wave = self.generate_sine(freq, duration, self.adsr_envelope(duration, 0.05, 0.4, 0.6, 0.2))
                wave = np.clip(wave * 2.0, -1, 1) 
                return wave * 0.7
            elif variation == 1: # Square Bass
                wave = self.generate_square(freq, duration, self.adsr_envelope(duration, 0.05, 0.3, 0.5, 0.1))
                b, a = scipy.signal.butter(2, 400/(self.sr/2), btype='low')
                wave = scipy.signal.lfilter(b, a, wave)
                return wave * 0.6
            elif variation == 2: # Saw Bass
                wave = self.generate_saw(freq, duration, self.adsr_envelope(duration, 0.1, 0.4, 0.7, 0.2))
                b, a = scipy.signal.butter(2, 600/(self.sr/2), btype='low')
                wave = scipy.signal.lfilter(b, a, wave)
                return wave * 0.6
            elif variation == 3: # Plucky
                wave = self.generate_sine(freq, duration, self.adsr_envelope(duration, 0.01, 0.2, 0.0, 0.1))
                wave += self.generate_sine(freq*2, duration, self.adsr_envelope(duration, 0.01, 0.1, 0.0, 0.1)) * 0.5
                return wave * 0.7
            else: # Sub
                wave = self.generate_sine(freq, duration, self.adsr_envelope(duration, 0.1, 0.5, 0.8, 0.2))
                return wave * 0.8

        elif style == 'chill':
            # Smooth Sine
            wave = self.generate_sine(freq, duration, self.adsr_envelope(duration, 0.1, 0.2, 0.8, 0.2))
            return wave * 0.6
        else: # Pop
            wave = self.generate_saw(freq, duration, self.adsr_envelope(duration, 0.05, 0.2, 0.8, 0.1))
            wave = np.convolve(wave, np.ones(5)/5, mode='same')
            return wave * 0.6

    def chord(self, freqs, duration, style='pop', variation=0):
        wave = np.zeros(int(self.sr * duration))
        env = self.adsr_envelope(duration, 0.1, 0.2, 0.6, 0.5)
        
        if style == 'rap':
            if variation == 0: # Dark Pad (Synth)
                for f in freqs:
                    wave += self.generate_saw(f, duration, env) * 0.3
                b, a = scipy.signal.butter(2, 800/(self.sr/2), btype='low')
                wave = scipy.signal.lfilter(b, a, wave)
            elif variation == 1: # Jazz/Soul E-Piano (Sampled Style)
                # Sine + Harmonics + Tremolo + Saturation
                for f in freqs:
                    wave += self.generate_sine(f, duration, env) * 0.4
                    wave += self.generate_sine(f*2, duration, env) * 0.1
                    wave += self.generate_sine(f*3, duration, env) * 0.05
                
                # Tremolo
                t = np.linspace(0, duration, len(wave))
                tremolo = 1 + 0.2 * np.sin(2 * np.pi * 6 * t)
                wave *= tremolo
                
                # Saturation (Warmth)
                wave = np.tanh(wave * 1.5)
                
            elif variation == 2: # Orchestral Strings (Sampled Style)
                env_str = self.adsr_envelope(duration, 0.5, 0.2, 0.8, 0.5)
                for f in freqs:
                    wave += self.generate_saw(f, duration, env_str) * 0.2
                    wave += self.generate_saw(f*1.01, duration, env_str) * 0.15
                b, a = scipy.signal.butter(2, 1500/(self.sr/2), btype='low')
                wave = scipy.signal.lfilter(b, a, wave)
            elif variation == 3: # Funk Brass Stab (Sampled Style)
                env_stab = self.adsr_envelope(duration, 0.05, 0.1, 0.4, 0.1)
                for f in freqs:
                    wave += self.generate_saw(f, duration, env_stab) * 0.3
                    wave += self.generate_square(f, duration, env_stab) * 0.2
                
                # Filter Envelope (Wah effect)
                t = np.linspace(0, duration, len(wave))
                cutoff = 500 + 2000 * np.exp(-10 * t)
                # Note: Time-varying filter is hard in pure numpy/scipy without loop
                # We'll approximate with static filter for now
                b, a = scipy.signal.butter(2, 2000/(self.sr/2), btype='low')
                wave = scipy.signal.lfilter(b, a, wave)
                
            else: # Wobble Bass/Synth
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                lfo = 1 + 0.01 * np.sin(2 * np.pi * 3 * t)
                for f in freqs:
                    wave += self.generate_saw(f, duration, env) * 0.3
                wave = wave * lfo
                
            return wave * 0.5
        elif style == 'chill':
            # E-Piano (Sine + Harmonics)
            for f in freqs:
                wave += self.generate_sine(f, duration, env) * 0.4
                wave += self.generate_sine(f*2, duration, env) * 0.1
                wave += self.generate_sine(f*3, duration, env) * 0.05
            # Tremolo
            t = np.linspace(0, duration, len(wave))
            tremolo = 1 + 0.2 * np.sin(2 * np.pi * 5 * t)
            return wave * tremolo * 0.5
        else: # Pop
            for f in freqs:
                wave += self.generate_saw(f, duration, env) * 0.3
                wave += self.generate_saw(f * 1.01, duration, env) * 0.2
                wave += self.generate_saw(f * 0.99, duration, env) * 0.2
            wave = np.convolve(wave, np.ones(10)/10, mode='same')
            return wave * 0.4

class CreativeDirector:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ.get("gh_token"),
        )

    def analyze_structure(self, transcript_text):
        print("Phase 1.5: Creative Direction (AI Analysis)...")
        prompt = f"""
        You are a music producer. Analyze the following lyrics/transcript with timestamps.
        Identify moments for dramatic musical effects to enhance the emotion.
        
        Transcript:
        {transcript_text}
        
        Return a JSON object with a key "events" containing a list of objects.
        Each object must have:
        - "time": float (timestamp in seconds)
        - "type": string (one of: "drop", "riser", "impact")
        - "duration": float (duration of effect in seconds, default 1.0)
        - "reason": string (brief explanation)
        
        Rules:
        - "drop": MASSIVE, EARTH-SHAKING BASS DROP. Use at the absolute peak moments or punchlines.
        - "silence": Stop the music completely for a second (or specified duration) for dramatic effect.
        - ONLY use "drop" or "silence". Do not use other effects.
        - Keep it tasteful. Max 2-3 events for this short clip.
        - IMPORTANT: Do not place effects too close to each other. Avoid consecutive effects within 5-10 seconds of each other.
        - Less is more. Only add an effect if it is absolutely necessary for the impact.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o",
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            # Clean up markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            data = json.loads(content.strip())
            print("Creative Director suggested events:")
            for event in data['events']:
                print(f"- {event['time']}s: {event['type']} ({event['reason']})")
            return data['events']
        except Exception as e:
            print(f"Creative Director failed: {e}")
            return []

class AudioProcessor:
    def __init__(self, input_path, output_dir="output"):
        self.input_path = input_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sr = 44100
        self.creative_events = []
        
    def load_audio(self):
        print(f"Loading audio from {self.input_path}...")
        if self.input_path.lower().endswith('.mov'):
            video = VideoFileClip(self.input_path)
            audio_path = os.path.join(self.output_dir, "extracted_audio.wav")
            video.audio.write_audiofile(audio_path, logger=None)
            self.audio_path = audio_path
        else:
            self.audio_path = self.input_path
            
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr)
        return self.y, self.sr

    def preprocess(self):
        print("Phase 1: Preprocessing (Noise Reduction)...")
        # Noise reduction
        reduced_noise = nr.reduce_noise(y=self.y, sr=self.sr, prop_decrease=0.75)
        
        # Save intermediate
        sf.write(os.path.join(self.output_dir, "1_denoised.wav"), reduced_noise, self.sr)
        self.y = reduced_noise
        return self.y

    def transcribe_and_analyze_wpm(self):
        print("Phase 1.4: Transcription & WPM Analysis (Word-Level)...")
        try:
            model = whisper.load_model("base")
            
            # Use in-memory audio to avoid ffmpeg file reading issues
            print("Resampling for Whisper (16kHz)...")
            audio_16k = librosa.resample(self.y, orig_sr=self.sr, target_sr=16000)
            audio_16k = audio_16k.astype(np.float32)
            
            # Request word timestamps for instantaneous WPM
            result = model.transcribe(audio_16k, word_timestamps=True)
            self.transcription_segments = result['segments']
            
            # Collect all words
            all_words = []
            for segment in self.transcription_segments:
                if 'words' in segment:
                    all_words.extend(segment['words'])
            
            # Fallback if word timestamps missing
            if not all_words and self.transcription_segments:
                print("Word timestamps not available, falling back to segment averaging.")
                for segment in self.transcription_segments:
                    text = segment['text']
                    words = text.split()
                    seg_dur = segment['end'] - segment['start']
                    if len(words) > 0:
                        per_word = seg_dur / len(words)
                        for i, w_text in enumerate(words):
                            all_words.append({
                                'word': w_text,
                                'start': segment['start'] + i * per_word,
                                'end': segment['start'] + (i + 1) * per_word
                            })

            print(f"Detected {len(all_words)} words.")
            
            # Calculate WPM profile using sliding window
            duration = len(self.y) / self.sr
            window_size = 2.0 # 2 seconds window for local density
            step_size = 0.25 # 0.25s resolution
            
            self.wpm_points = []
            times = np.arange(0, duration, step_size)
            
            # Calculate global average WPM first
            total_words = len(all_words)
            if total_words > 1:
                speaking_duration = all_words[-1]['end'] - all_words[0]['start']
                self.avg_wpm = (total_words / speaking_duration) * 60 if speaking_duration > 0 else 100
            else:
                self.avg_wpm = 100
                
            print(f"Global Average WPM: {self.avg_wpm:.2f}")

            raw_wpms = []
            for t in times:
                # Count words centered around t
                start_window = t - window_size/2
                end_window = t + window_size/2
                
                count = 0
                for w in all_words:
                    # Use word center to check inclusion
                    w_center = (w['start'] + w['end']) / 2
                    if start_window <= w_center <= end_window:
                        count += 1
                
                wpm = (count / window_size) * 60
                raw_wpms.append(wpm)
            
            # Apply additional smoothing to prevent rapid changes (Savgol filter)
            # Window length 11 * 0.25 = 2.75s
            if len(raw_wpms) > 11:
                smoothed_wpms = scipy.signal.savgol_filter(raw_wpms, window_length=11, polyorder=2)
            else:
                smoothed_wpms = raw_wpms
                
            smoothed_wpms = np.maximum(smoothed_wpms, 0)

            for i, t in enumerate(times):
                self.wpm_points.append({'time': t, 'wpm': smoothed_wpms[i]})
                
            # Save transcript with word timings
            transcript_path = os.path.join(self.output_dir, "transcript_detailed.txt")
            full_text_transcript = ""
            with open(transcript_path, "w") as f:
                f.write(f"Global Average WPM: {self.avg_wpm:.2f}\n\n")
                for w in all_words:
                    line = f"[{w['start']:.2f} - {w['end']:.2f}] {w['word']}\n"
                    f.write(line)
                    full_text_transcript += line
            print(f"Detailed transcript saved to {transcript_path}")
            
            # Call Creative Director
            director = CreativeDirector()
            self.creative_events = director.analyze_structure(full_text_transcript)
            
        except Exception as e:
            print(f"Warning: Transcription failed ({e}). Using default WPM.")
            import traceback
            traceback.print_exc()
            self.avg_wpm = 100
            self.wpm_points = []
            self.creative_events = []

    def analyze(self):
        print("Phase 1.2 & 1.3: Analysis (Pitch & Rhythm)...")
        
        # Run Transcription
        self.transcribe_and_analyze_wpm()

        # Tempo
        tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.tempo = float(tempo)
        self.beat_frames = beat_frames
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        print(f"Detected Tempo: {self.tempo:.2f} BPM")
        print(f"Detected {len(self.beat_times)} beats.")
        
        # Pitch / Key (Simple estimation)
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        key_index = np.argmax(np.sum(chroma, axis=1))
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.key = notes[key_index]
        print(f"Estimated Key: {self.key}")
        
        return self.tempo, self.key

    def enhance_vocals(self):
        print("Phase 2: Vocal Enhancement...")
        # Using Pedalboard for effects chain
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80),
            NoiseGate(threshold_db=-40, ratio=4, release_ms=250),
            Compressor(threshold_db=-16, ratio=3, attack_ms=10, release_ms=100),
            Reverb(room_size=0.3, wet_level=0.2),
            Limiter(threshold_db=-1.0)
        ])
        
        # Pedalboard works with audio file reading/writing usually, or numpy arrays
        # We need to ensure shape is (channels, samples)
        if len(self.y.shape) == 1:
            audio_to_process = self.y.reshape(1, -1)
        else:
            audio_to_process = self.y

        processed = board(audio_to_process, self.sr)
        
        # Flatten back if mono
        if processed.shape[0] == 1:
            self.y_processed = processed.flatten()
        else:
            self.y_processed = processed
            
        sf.write(os.path.join(self.output_dir, "2_enhanced_vocals.wav"), self.y_processed.T if len(self.y_processed.shape) > 1 else self.y_processed, self.sr)
        return self.y_processed

    def generate_music(self, genre='pop', variation=0):
        print(f"Phase 3: Music Generation ({genre.capitalize()} - Var {variation})...")
        
        synth = Synthesizer(self.sr)
        
        duration = len(self.y) / self.sr
        
        # Construct beat grid for variable tempo
        if hasattr(self, 'wpm_points') and len(self.wpm_points) > 0:
            print("Using WPM to modulate tempo...")
            times = [p['time'] for p in self.wpm_points]
            wpms = [p['wpm'] for p in self.wpm_points]
            
            # Interpolation function
            def get_wpm_factor(t):
                if not times: return 1.0
                wpm = np.interp(t, times, wpms)
                # Normalize against average
                if self.avg_wpm == 0: return 1.0
                ratio = wpm / self.avg_wpm
                # Dampen extreme values (0.5x to 2.0x)
                ratio = np.clip(ratio, 0.5, 2.0)
                return ratio

            # Start from the first detected beat by librosa to align phase
            start_time = self.beat_times[0] if (hasattr(self, 'beat_times') and len(self.beat_times) > 0) else 0.0
            
            new_beats = [start_time]
            
            # Forward generation
            current_t = start_time
            base_ibi = 60 / self.tempo
            while current_t < duration:
                factor = get_wpm_factor(current_t)
                # Higher WPM -> Faster Tempo -> Shorter Interval
                # Direct modulation for tighter sync
                interval = base_ibi / factor 
                current_t += interval
                new_beats.append(current_t)
                
            # Backward generation
            current_t = start_time
            while current_t > 0:
                factor = get_wpm_factor(current_t)
                interval = base_ibi / factor
                current_t -= interval
                new_beats.insert(0, current_t)
                
            full_beat_times = np.array(new_beats)
            full_beat_times = full_beat_times[(full_beat_times >= -1.0) & (full_beat_times <= duration + 1.0)]

        elif hasattr(self, 'beat_times') and len(self.beat_times) > 1:
            # Use detected beats
            avg_ibi = np.mean(np.diff(self.beat_times))
            if avg_ibi == 0: avg_ibi = 60 / self.tempo # Fallback
            
            # Extrapolate start
            current_beat = self.beat_times[0]
            pre_beats = []
            while current_beat > 0:
                current_beat -= avg_ibi
                pre_beats.insert(0, current_beat)
            
            # Extrapolate end
            current_beat = self.beat_times[-1]
            post_beats = []
            while current_beat < duration:
                current_beat += avg_ibi
                post_beats.append(current_beat)
                
            full_beat_times = np.concatenate([pre_beats, self.beat_times, post_beats])
            # Filter to be within [0, duration] roughly
            full_beat_times = full_beat_times[(full_beat_times >= -avg_ibi) & (full_beat_times <= duration + avg_ibi)]
        else:
            # Fallback to constant tempo
            beat_duration = 60 / self.tempo
            full_beat_times = np.arange(0, duration + beat_duration, beat_duration)

        total_beats = len(full_beat_times)
        
        def get_beat_time(beat_idx):
            # Handle fractional beats
            idx_int = int(beat_idx)
            frac = beat_idx - idx_int
            
            if idx_int < 0: return 0.0
            if idx_int >= len(full_beat_times) - 1:
                # Extrapolate from last known interval
                last_interval = full_beat_times[-1] - full_beat_times[-2] if len(full_beat_times) > 1 else 60/self.tempo
                return full_beat_times[-1] + (beat_idx - (len(full_beat_times)-1)) * last_interval
                
            t1 = full_beat_times[idx_int]
            t2 = full_beat_times[idx_int+1]
            return t1 + frac * (t2 - t1)
        
        # Initialize tracks
        total_samples = len(self.y)
        self.drums = np.zeros(total_samples)
        self.bass = np.zeros(total_samples)
        self.chords = np.zeros(total_samples)
        
        def add_sample(track, sample, start_time):
            start_sample = int(start_time * self.sr)
            
            # Handle negative start time (partial sample at beginning)
            if start_sample < 0:
                if start_sample + len(sample) <= 0: return # Completely before start
                offset = -start_sample
                sample = sample[offset:]
                start_sample = 0
            
            if start_sample >= len(track): return
            
            end_sample = start_sample + len(sample)
            if end_sample > len(track):
                sample = sample[:len(track)-start_sample]
                end_sample = len(track)
            
            if len(sample) == 0: return
            
            track[start_sample:end_sample] += sample

        # 1. Drums Pattern (One-shots, pre-generated)
        kick_sample = synth.kick(style=genre, variation=variation)
        snare_sample = synth.snare(style=genre, variation=variation)
        hihat_sample = synth.hihat(style=genre, variation=variation)
        
        # 2. Harmony & Bass Logic
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        base_freqs = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88]
        try:
            key_idx = note_names.index(self.key)
        except:
            key_idx = 0 # Default C
        
        # Scale degrees
        if genre == 'rap':
            # Minor Scale: W H W W H W W -> 0, 2, 3, 5, 7, 8, 10
            scale_intervals = [0, 2, 3, 5, 7, 8, 10]
        else:
            # Major Scale
            scale_intervals = [0, 2, 4, 5, 7, 9, 11]
            
        scale_freqs = []
        for interval in scale_intervals:
            idx = (key_idx + interval) % 12
            freq = base_freqs[idx]
            if (key_idx + interval) >= 12: freq *= 2
            scale_freqs.append(freq)
            
        # Progressions
        if genre == 'pop':
            progs = [
                [[0, 2, 4], [4, 6, 1], [5, 0, 2], [3, 5, 0]], # I V vi IV
                [[5, 0, 2], [3, 5, 0], [0, 2, 4], [4, 6, 1]], # vi IV I V
                [[0, 2, 4], [3, 5, 0], [0, 2, 4], [4, 6, 1]], # I IV I V
                [[0, 2, 4], [4, 6, 1], [3, 5, 0], [4, 6, 1]], # I V IV V
                [[5, 0, 2], [2, 4, 6], [3, 5, 0], [4, 6, 1]]  # vi iii IV V
            ]
        elif genre == 'rap':
            # Minor loops
            progs = [
                [[0, 2, 4], [0, 2, 4], [5, 0, 2], [4, 6, 1]], # i i VI V
                [[0, 2, 4], [5, 0, 2], [0, 2, 4], [5, 0, 2]], # i VI i VI
                [[0, 2, 4], [0, 2, 4], [3, 5, 0], [4, 6, 1]], # i i iv V
                [[0, 2, 4], [2, 4, 6], [0, 2, 4], [4, 6, 1]], # i III i V
                [[5, 0, 2], [4, 6, 1], [5, 0, 2], [0, 2, 4]]  # VI V VI i
            ]
        elif genre == 'chill':
            # 7th chords logic (simplified)
            progs = [
                [[0, 2, 4, 6], [3, 5, 0, 2], [4, 6, 1, 3], [0, 2, 4, 6]], # Imaj7 IVmaj7 V7 Imaj7
                [[1, 3, 5, 0], [4, 6, 1, 3], [0, 2, 4, 6], [5, 0, 2, 4]], # ii7 V7 Imaj7 vi7
                [[3, 5, 0, 2], [3, 5, 0, 2], [0, 2, 4, 6], [0, 2, 4, 6]], # IVmaj7 Imaj7
                [[5, 0, 2, 4], [3, 5, 0, 2], [0, 2, 4, 6], [4, 6, 1, 3]], # vi7 IVmaj7 Imaj7 V7
                [[0, 2, 4, 6], [1, 3, 5, 0], [3, 5, 0, 2], [4, 6, 1, 3]]  # I ii IV V
            ]

        progression = progs[variation % len(progs)]
        
        # Main Loop
        for i in range(total_beats):
            # Calculate local beat duration for synthesis
            current_time = get_beat_time(i)
            next_time = get_beat_time(i+1)
            local_beat_dur = next_time - current_time
            if local_beat_dur <= 0: local_beat_dur = 0.5 # Safety
            
            # Drums
            if genre == 'pop':
                if i % 4 == 0 or i % 4 == 2:
                    add_sample(self.drums, kick_sample, get_beat_time(i))
                elif i % 8 == 7 and variation > 2: 
                     add_sample(self.drums, kick_sample, get_beat_time(i + 0.5))
                if i % 4 == 1 or i % 4 == 3:
                    add_sample(self.drums, snare_sample, get_beat_time(i))
                add_sample(self.drums, hihat_sample, get_beat_time(i))
                add_sample(self.drums, hihat_sample, get_beat_time(i + 0.5))
                
            elif genre == 'rap':
                is_halftime = self.tempo > 100
                
                if variation == 0: # Standard Trap
                    if i % 4 == 0: add_sample(self.drums, kick_sample, get_beat_time(i))
                    if i % 4 == 2: add_sample(self.drums, snare_sample, get_beat_time(i))
                    if i % 2 == 0:
                        add_sample(self.drums, hihat_sample, get_beat_time(i))
                        add_sample(self.drums, hihat_sample, get_beat_time(i + 0.5))
                    else:
                        add_sample(self.drums, hihat_sample, get_beat_time(i))
                        add_sample(self.drums, hihat_sample, get_beat_time(i + 0.25))
                        add_sample(self.drums, hihat_sample, get_beat_time(i + 0.5))
                        add_sample(self.drums, hihat_sample, get_beat_time(i + 0.75))
                        
                elif variation == 1: # Boom Bap
                    if i % 4 == 0: add_sample(self.drums, kick_sample, get_beat_time(i))
                    if i % 4 == 2: add_sample(self.drums, snare_sample, get_beat_time(i))
                    if i % 4 == 1: add_sample(self.drums, kick_sample, get_beat_time(i + 0.75))
                    if i % 4 == 3: add_sample(self.drums, kick_sample, get_beat_time(i + 0.25))
                    add_sample(self.drums, hihat_sample, get_beat_time(i))
                    add_sample(self.drums, hihat_sample, get_beat_time(i + 0.66))
                    
                elif variation == 2: # Drill-ish
                    if i % 4 == 0: add_sample(self.drums, kick_sample, get_beat_time(i))
                    if i % 4 == 2: add_sample(self.drums, snare_sample, get_beat_time(i))
                    if i % 4 == 3: add_sample(self.drums, kick_sample, get_beat_time(i + 0.5))
                    add_sample(self.drums, hihat_sample, get_beat_time(i))
                    add_sample(self.drums, hihat_sample, get_beat_time(i + 0.33))
                    add_sample(self.drums, hihat_sample, get_beat_time(i + 0.66))
                    
                elif variation == 3: # Minimal
                    if i % 4 == 0: add_sample(self.drums, kick_sample, get_beat_time(i))
                    if i % 4 == 2: add_sample(self.drums, snare_sample, get_beat_time(i))
                    if i % 2 == 0: add_sample(self.drums, hihat_sample, get_beat_time(i))
                    
                else: # Industrial
                    if i % 4 == 0: add_sample(self.drums, kick_sample, get_beat_time(i))
                    if i % 4 == 1: add_sample(self.drums, kick_sample, get_beat_time(i))
                    if i % 4 == 2: add_sample(self.drums, snare_sample, get_beat_time(i))
                    if i % 4 == 3: add_sample(self.drums, kick_sample, get_beat_time(i))
                    add_sample(self.drums, hihat_sample, get_beat_time(i))
                    add_sample(self.drums, hihat_sample, get_beat_time(i + 0.25))
                    add_sample(self.drums, hihat_sample, get_beat_time(i + 0.5))
                    add_sample(self.drums, hihat_sample, get_beat_time(i + 0.75))

            elif genre == 'chill':
                if i % 4 == 0:
                    add_sample(self.drums, kick_sample, get_beat_time(i))
                if i % 4 == 2 and variation % 2 == 1:
                    add_sample(self.drums, kick_sample, get_beat_time(i + 0.5))
                if i % 4 == 1 or i % 4 == 3:
                    add_sample(self.drums, snare_sample, get_beat_time(i))
                if i % 2 == 0:
                    add_sample(self.drums, hihat_sample, get_beat_time(i))

            # Bass & Chords (Generated dynamically for local tempo)
            bar = i // 4
            chord_idx = bar % 4
            chord_indices = progression[chord_idx]
            
            # Bass
            root_scale_idx = chord_indices[0]
            root_freq = scale_freqs[root_scale_idx] / 2
            
            if genre == 'rap':
                if i % 4 == 0: # Beat 1
                    dur = get_beat_time(i+2) - get_beat_time(i)
                    add_sample(self.bass, synth.bass(root_freq, dur, style=genre, variation=variation), get_beat_time(i))
                if i % 4 == 2 and variation % 2 == 1: # Beat 3
                    dur = get_beat_time(i+2) - get_beat_time(i)
                    add_sample(self.bass, synth.bass(root_freq, dur, style=genre, variation=variation), get_beat_time(i + 0.5))
            elif genre == 'chill':
                if i % 4 == 0:
                    dur = get_beat_time(i+4) - get_beat_time(i)
                    add_sample(self.bass, synth.bass(root_freq, dur, style=genre), get_beat_time(i))
            else: # Pop
                if i % 4 == 0:
                    dur = get_beat_time(i+1) - get_beat_time(i)
                    add_sample(self.bass, synth.bass(root_freq, dur, style=genre), get_beat_time(i))
                if i % 4 == 2:
                    dur = get_beat_time(i+1) - get_beat_time(i)
                    add_sample(self.bass, synth.bass(root_freq, dur/2, style=genre), get_beat_time(i + 0.5))
                if i % 4 == 3:
                    dur = get_beat_time(i+1) - get_beat_time(i)
                    add_sample(self.bass, synth.bass(root_freq, dur/2, style=genre), get_beat_time(i + 0.5))

            # Chords (Once per bar)
            if i % 4 == 0:
                chord_freqs = [scale_freqs[idx % 7] for idx in chord_indices]
                dur = get_beat_time(i+4) - get_beat_time(i)
                chord_sample = synth.chord(chord_freqs, dur, style=genre, variation=variation)
                add_sample(self.chords, chord_sample, get_beat_time(i))

        # Save stems (optional, maybe skip for batch to save space/time)
        # sf.write(os.path.join(self.output_dir, f"3_drums_{genre}_{variation}.wav"), self.drums, self.sr)

        # Apply Creative Effects
        if self.creative_events:
            print("Applying creative effects...")
            for event in self.creative_events:
                t = event['time']
                etype = event['type']
                dur = event.get('duration', 1.0)
                
                start_sample = int(t * self.sr)
                
                if start_sample >= total_samples: continue
                
                if etype == "drop":
                    # Use the new heavy bass drop
                    drop_sample = synth.bass_drop(duration=dur)
                    # Add to bass track with high volume
                    add_sample(self.bass, drop_sample * 1.0, t)
                    # Also add a kick impact for punch
                    impact = synth.impact(duration=0.5)
                    add_sample(self.drums, impact * 0.8, t)

                elif etype == "silence":
                    # Fade out before silence
                    fade_out_len = int(0.1 * self.sr) # 100ms fade out
                    fade_start = max(0, start_sample - fade_out_len)
                    end_sample = int((t + dur) * self.sr)
                    
                    # Fade in after silence
                    fade_in_len = int(0.1 * self.sr) # 100ms fade in
                    fade_in_end = min(total_samples, end_sample + fade_in_len)
                    
                    # Apply to tracks
                    for track in [self.drums, self.bass, self.chords]:
                        # Fade Out
                        if fade_start < len(track) and start_sample < len(track):
                            actual_fade_len = start_sample - fade_start
                            if actual_fade_len > 0:
                                track[fade_start:start_sample] *= np.linspace(1, 0, actual_fade_len)
                        
                        # Silence
                        if start_sample < len(track):
                            silence_end = min(end_sample, len(track))
                            track[start_sample:silence_end] = 0
                            
                        # Fade In
                        if end_sample < len(track) and fade_in_end <= len(track):
                            actual_fade_in_len = fade_in_end - end_sample
                            if actual_fade_in_len > 0:
                                track[end_sample:fade_in_end] *= np.linspace(0, 1, actual_fade_in_len)

    def apply_sidechain(self, source, target, amount=0.5):
        # Simple envelope follower on source (Drums)
        env = np.abs(source)
        # Lowpass to get the "thump" envelope
        b, a = scipy.signal.butter(2, 50/(self.sr/2), btype='low')
        env = scipy.signal.lfilter(b, a, env)
        # Normalize envelope
        if np.max(env) > 0:
            env = env / np.max(env)
        
        # Invert for gain reduction (Ducking)
        # When kick hits (env=1), gain goes to (1-amount)
        gain_reduction = 1.0 - (env * amount)
        gain_reduction = np.clip(gain_reduction, 0, 1)
        
        return target * gain_reduction

    def mix(self, output_filename):
        print(f"Phase 4: Mixing & Mastering ({output_filename})...")
        def normalize(arr):
            if np.max(np.abs(arr)) == 0: return arr
            return arr / (np.max(np.abs(arr)) + 1e-6)

        # Ensure all tracks are same length
        max_len = max(len(self.y_processed), len(self.drums), len(self.bass), len(self.chords))
        
        def pad(arr, length):
            if len(arr) < length:
                return np.pad(arr, (0, length - len(arr)))
            return arr[:length]
            
        # Handle potential stereo vocals (mix down to mono for consistency in this script)
        vocals = self.y_processed
        if len(vocals.shape) > 1:
            vocals = np.mean(vocals, axis=0)
            
        vocals = pad(vocals, max_len)
        drums = pad(self.drums, max_len)
        bass = pad(self.bass, max_len)
        chords = pad(self.chords, max_len)

        # Apply Sidechain Compression (Ducking)
        # Bass and Chords duck when Drums hit
        print("Applying Sidechain Compression...")
        bass = self.apply_sidechain(drums, bass, amount=0.6)
        chords = self.apply_sidechain(drums, chords, amount=0.4)

        vocal_level = 1.0
        drum_level = 0.5 # Reduced drums
        bass_level = 0.45 # Reduced bass
        chord_level = 0.3 # Reduced chords
        
        # Summing
        final_mix = (
            normalize(vocals) * vocal_level +
            normalize(drums) * drum_level +
            normalize(bass) * bass_level +
            normalize(chords) * chord_level
        )
        
        # Mastering Chain
        print("Applying Mastering Chain...")
        master_board = Pedalboard([
            # Glue Compressor
            Compressor(threshold_db=-12, ratio=2.5, attack_ms=30, release_ms=100),
            # Final Limiter to boost loudness without clipping
            Limiter(threshold_db=-0.1)
        ])
        
        # Pedalboard expects (channels, samples) or (samples)
        # It returns same shape
        final_mix = master_board(final_mix, self.sr)
        
        # Clip safety
        final_mix = np.clip(final_mix, -1.0, 1.0)
        
        output_path = os.path.join(self.output_dir, output_filename)
        sf.write(output_path, final_mix, self.sr)
        print(f"Done! Saved audio to {output_path}")
        
        # Export Video if input was video
        if self.input_path.lower().endswith('.mov') or self.input_path.lower().endswith('.mp4'):
            try:
                video_output_path = output_path.replace('.wav', '.mp4')
                print(f"Merging with video -> {video_output_path}")
                
                video = VideoFileClip(self.input_path)
                audio = AudioFileClip(output_path)
                
                # Trim to shorter length
                min_dur = min(video.duration, audio.duration)
                try:
                    video = video.subclip(0, min_dur)
                    audio = audio.subclip(0, min_dur)
                except AttributeError:
                    print("Warning: subclip not available, skipping trim.")
                
                final_video = video.set_audio(audio)
                # Use a fast preset for testing, 'medium' or 'slow' for quality
                final_video.write_videofile(video_output_path, codec='libx264', audio_codec='aac', logger=None)
                print(f"Video saved to {video_output_path}")
            except Exception as e:
                print(f"Video export failed: {e}")

if __name__ == "__main__":
    # Check for input file
    input_file = "voice_enhancer/IMG_2091.MOV"
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
    else:
        processor = AudioProcessor(input_file)
        processor.load_audio()
        processor.preprocess()
        processor.analyze()
        processor.enhance_vocals()
        
        # Generate 5 Pop, 5 Rap, 5 Chill
        genres = ['pop', 'rap', 'chill']
        for genre in genres:
            for i in range(5):
                processor.generate_music(genre=genre, variation=i)
                processor.mix(f"{genre}_{i+1}.wav")

