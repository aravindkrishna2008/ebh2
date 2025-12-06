import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from scipy.interpolate import interp1d
import scipy.signal

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
        
        if len(envelope) < total_samples:
            envelope = np.pad(envelope, (0, total_samples - len(envelope)))
        else:
            envelope = envelope[:total_samples]
            
        return envelope

    def kick(self, style='pop', variation=0):
        duration = 0.4
        if style == 'rap':
            if variation == 0: # Deep 808
                duration = 0.6 
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                freq = np.linspace(100, 40, len(t)) 
                wave = np.sin(2 * np.pi * freq * t)
                env = np.exp(-4 * t) 
                wave = np.clip(wave * 1.5, -1, 1) 
            elif variation == 1: # Punchy Boom Bap
                duration = 0.3
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                freq = np.linspace(150, 60, len(t))
                wave = np.sin(2 * np.pi * freq * t)
                env = np.exp(-15 * t) # Fast decay
                wave = np.clip(wave * 1.2, -1, 1)
            elif variation == 2: # Distorted
                duration = 0.5
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                freq = np.linspace(120, 40, len(t))
                wave = scipy.signal.square(2 * np.pi * freq * t) # Square for grit
                # Lowpass to tame it
                b, a = scipy.signal.butter(2, 200/(self.sr/2), btype='low')
                wave = scipy.signal.lfilter(b, a, wave)
                env = np.exp(-5 * t)
                wave = wave * env
            elif variation == 3: # Soft/Round
                duration = 0.4
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                freq = np.linspace(90, 45, len(t))
                wave = np.sin(2 * np.pi * freq * t)
                env = np.exp(-6 * t)
            else: # Clicky
                duration = 0.3
                t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
                freq = np.linspace(200, 50, len(t))
                wave = np.sin(2 * np.pi * freq * t)
                env = np.exp(-12 * t)
                # Add click
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
            if variation == 0: # Trap Clap
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.01, 0.15, 0.0, 0.05))
                b, a = scipy.signal.butter(2, [800/(self.sr/2), 2000/(self.sr/2)], btype='band')
                noise = scipy.signal.lfilter(b, a, noise)
                return noise * 0.9
            elif variation == 1: # Boom Bap Snare
                tone = self.generate_sine(180, duration, self.adsr_envelope(duration, 0.005, 0.1, 0, 0.05))
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.005, 0.2, 0, 0.05))
                b, a = scipy.signal.butter(2, 1000/(self.sr/2), btype='low') # Darker noise
                noise = scipy.signal.lfilter(b, a, noise)
                return (tone * 0.6 + noise * 0.7)
            elif variation == 2: # Rimshot
                tone = self.generate_sine(500, 0.05, self.adsr_envelope(0.05, 0.001, 0.02, 0, 0.02))
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
            if variation == 0: # Trap Tick
                duration = 0.05
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.001, 0.03, 0.0, 0.01))
                noise = np.diff(noise, prepend=0)
                return noise * 0.7
            elif variation == 1: # Shaker
                duration = 0.1
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.02, 0.05, 0, 0.03))
                b, a = scipy.signal.butter(2, 3000/(self.sr/2), btype='high')
                noise = scipy.signal.lfilter(b, a, noise)
                return noise * 0.5
            elif variation == 2: # Open Hat
                duration = 0.3
                noise = self.generate_noise(duration, self.adsr_envelope(duration, 0.01, 0.2, 0, 0.05))
                b, a = scipy.signal.butter(2, 4000/(self.sr/2), btype='high')
                noise = scipy.signal.lfilter(b, a, noise)
                return noise * 0.6
            elif variation == 3: # Lo-fi
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
            if variation == 0: # Dark Pad
                for f in freqs:
                    wave += self.generate_saw(f, duration, env) * 0.3
                b, a = scipy.signal.butter(2, 800/(self.sr/2), btype='low')
                wave = scipy.signal.lfilter(b, a, wave)
            elif variation == 1: # Plucky Bells
                env_bell = self.adsr_envelope(duration, 0.01, 0.3, 0.0, 0.1)
                for f in freqs:
                    wave += self.generate_sine(f, duration, env_bell) * 0.4
                    wave += self.generate_sine(f*2, duration, env_bell) * 0.1
            elif variation == 2: # Strings
                env_str = self.adsr_envelope(duration, 0.5, 0.2, 0.8, 0.5)
                for f in freqs:
                    wave += self.generate_saw(f, duration, env_str) * 0.2
                    wave += self.generate_saw(f*1.01, duration, env_str) * 0.15
                b, a = scipy.signal.butter(2, 1500/(self.sr/2), btype='low')
                wave = scipy.signal.lfilter(b, a, wave)
            elif variation == 3: # Organ
                for f in freqs:
                    wave += self.generate_sine(f, duration, env) * 0.3
                    wave += self.generate_sine(f*2, duration, env) * 0.2
                    wave += self.generate_sine(f*4, duration, env) * 0.1
            else: # Wobble
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


class BeatGenerator:
    def __init__(self, vocal_path, style='rap', intensity=3):
        self.vocal_path = vocal_path
        self.style = style
        self.intensity = intensity
        self.tempo = None
        self.duration = None
        self.beat_times = []
        
    def generate_beat_map(self):
        try:
            print("Analyzing dynamic tempo...")
            y, sr = librosa.load(self.vocal_path, sr=44100)
            self.duration = librosa.get_duration(y=y, sr=sr)
            
            try:
                global_tempo_raw = librosa.beat.beat_track(y=y, sr=sr)[0]
                global_tempo = float(global_tempo_raw)
            except:
                global_tempo = 120.0
                
            if global_tempo <= 0: global_tempo = 120.0
            print(f"Global tempo: {global_tempo:.1f} BPM")
            
            num_windows = 8
            window_size = self.duration / num_windows
            
            min_window_samples = sr * 0.5 
            
            times = []
            tempos = []
            
            for i in range(num_windows):
                start_time = i * window_size
                end_time = min((i + 2) * window_size, self.duration) 
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                if end_sample - start_sample < min_window_samples: continue 
                
                y_slice = y[start_sample:end_sample]
                try:
                    if len(y_slice) == 0: continue
                    
                    local_tempo_raw = librosa.beat.beat_track(y=y_slice, sr=sr)[0]
                    local_tempo = float(local_tempo_raw)
                    
                    if local_tempo <= 0: continue
                    
                    if local_tempo < global_tempo * 0.6: local_tempo *= 2
                    elif local_tempo > global_tempo * 1.8: local_tempo /= 2
                    
                    mid_time = (start_time + end_time) / 2
                    times.append(mid_time)
                    tempos.append(local_tempo)
                except Exception as e:
                    print(f"Window {i} analysis failed: {e}")
                    pass
                    
            if not times or len(times) < 2:
                print("Dynamic analysis failed or insufficient data, falling back to global")
                times = [0, max(self.duration, 0.1)] 
                tempos = [global_tempo, global_tempo]
            else:
                if times[0] > 0:
                    times.insert(0, 0)
                    tempos.insert(0, tempos[0])
                if times[-1] < self.duration:
                    times.append(self.duration)
                    tempos.append(tempos[-1])
                    
            tempo_func = interp1d(times, tempos, kind='linear', bounds_error=False, fill_value="extrapolate")
            
            beat_times = [0.0]
            curr_time = 0.0
            safety_limit = 10000 
            count = 0
            
            while curr_time < self.duration and count < safety_limit:
                try:
                    local_bpm = float(tempo_func(curr_time))
                except:
                    local_bpm = global_tempo
                    
                step = 60.0 / max(local_bpm, 30) 
                curr_time += step
                if curr_time < self.duration:
                    beat_times.append(curr_time)
                count += 1
                    
            self.beat_times = beat_times
            self.tempo = global_tempo 
            print(f"Generated {len(beat_times)} beats with variable tempo")
            
        except Exception as e:
            print(f"CRITICAL: generate_beat_map failed: {e}")
            self.duration = self.duration if self.duration else 10.0
            self.beat_times = [0.0, 0.5, 1.0, 1.5] 
            self.tempo = 120.0

    def detect_tempo(self):
        return self.generate_beat_map()
    
    def generate_kick(self, sr=44100):
        duration = 0.15
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        freq = 150 * np.exp(-15 * t)
        kick = np.sin(2 * np.pi * freq * t)
        envelope = np.exp(-10 * t)
        kick = kick * envelope
        return (kick * 32767).astype(np.int16)
    
    def generate_snare(self, sr=44100):
        duration = 0.1
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        noise = np.random.uniform(-1, 1, samples)
        tone = np.sin(2 * np.pi * 200 * t)
        snare = 0.6 * noise + 0.4 * tone
        envelope = np.exp(-25 * t)
        snare = snare * envelope
        return (snare * 32767).astype(np.int16)
    
    def generate_hihat(self, sr=44100, closed=True):
        duration = 0.05 if closed else 0.15
        samples = int(sr * duration)
        hihat = np.random.uniform(-1, 1, samples)
        hihat = np.diff(hihat, prepend=hihat[0])
        t = np.linspace(0, duration, samples)
        envelope = np.exp(-30 * t if closed else -10 * t)
        hihat = hihat * envelope
        return (hihat * 32767 * 0.6).astype(np.int16)
    
    def generate_bass(self, sr=44100, note_duration=0.5, frequency=55):
        samples = int(sr * note_duration)
        t = np.linspace(0, note_duration, samples)
        bass = np.sin(2 * np.pi * frequency * t)
        bass += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
        envelope = np.exp(-2 * t)
        bass = bass * envelope * 0.7
        return (bass * 32767).astype(np.int16)
    
    def create_drum_pattern(self):
        sr = 44100
        total_samples = int(sr * self.duration)
        drums = np.zeros(total_samples, dtype=np.int16)
        
        kick = self.generate_kick(sr)
        snare = self.generate_snare(sr)
        hihat_closed = self.generate_hihat(sr, closed=True)
        hihat_open = self.generate_hihat(sr, closed=False)
        
        style_mode = 'rap' if self.style in ['trap', 'rap'] else 'chill'
        
        for i, t in enumerate(self.beat_times):
            beat_pos = int(t * sr)
            if beat_pos >= total_samples: break
            
            beat_in_bar = i % 4
            
            if style_mode == 'rap':
                if beat_in_bar in [0, 2]:
                    end_pos = min(beat_pos + len(kick), total_samples)
                    drums[beat_pos:end_pos] += kick[:end_pos - beat_pos]
                
                if beat_in_bar in [1, 3]:
                    end_pos = min(beat_pos + len(snare), total_samples)
                    drums[beat_pos:end_pos] += snare[:end_pos - beat_pos]
                
                next_t = self.beat_times[i+1] if i + 1 < len(self.beat_times) else t + (t - self.beat_times[i-1])
                half_t = (t + next_t) / 2
                half_pos = int(half_t * sr)
                
                if half_pos < total_samples:
                    end_pos = min(beat_pos + len(hihat_closed), total_samples)
                    drums[beat_pos:end_pos] += hihat_closed[:end_pos - beat_pos]
                    
                    hihat = hihat_open if (beat_in_bar % 2 == 1) else hihat_closed
                    end_pos = min(half_pos + len(hihat), total_samples)
                    drums[half_pos:end_pos] += hihat[:end_pos - half_pos]
            
            else: 
                if beat_in_bar in [0, 2]:
                    end_pos = min(beat_pos + len(kick), total_samples)
                    drums[beat_pos:end_pos] += (kick * 0.6).astype(np.int16)[:end_pos - beat_pos]
                
                if beat_in_bar in [1, 3]:
                    end_pos = min(beat_pos + len(snare), total_samples)
                    drums[beat_pos:end_pos] += (snare * 0.5).astype(np.int16)[:end_pos - beat_pos]
                
                end_pos = min(beat_pos + len(hihat_closed), total_samples)
                drums[beat_pos:end_pos] += (hihat_closed * 0.4).astype(np.int16)[:end_pos - beat_pos]
        
        return drums
    
    def create_bassline(self):
        sr = 44100
        total_samples = int(sr * self.duration)
        bassline = np.zeros(total_samples, dtype=np.int16)
        
        style_mode = 'rap' if self.style in ['trap', 'rap'] else 'chill'

        if style_mode == 'rap':
            notes = [55, 55, 44, 55]
            note_freq_multiplier = 1
        else:
            notes = [55, 49, 44, 49]
            note_freq_multiplier = 2 
        
        note_idx = 0
        
        for i, t in enumerate(self.beat_times):
            if style_mode == 'chill' and i % 2 != 0: continue
            
            start_pos = int(t * sr)
            if start_pos >= total_samples: break
            
            steps = 1 if style_mode == 'rap' else 2
            target_idx = min(i + steps, len(self.beat_times) - 1)
            
            if target_idx > i:
                end_t = self.beat_times[target_idx]
                note_dur = end_t - t
            else:
                note_dur = 0.5 
            
            frequency = notes[note_idx % len(notes)]
            bass_note = self.generate_bass(sr, note_dur, frequency)
            
            end_pos = min(start_pos + len(bass_note), total_samples)
            bassline[start_pos:end_pos] += bass_note[:end_pos - start_pos]
            
            if style_mode == 'rap' or (style_mode == 'chill' and i % 2 == 0):
                note_idx += 1
        
        return bassline
    
    def calculate_optimal_beat_volume(self, vocal):
        print(f"Vocal loudness: {vocal.dBFS:.1f} dBFS")
        
        base_adjustment = -12
        
        intensity_adjustment = (self.intensity - 3) * 3
        final_adjustment = base_adjustment + intensity_adjustment
        
        print(f"Beat adjustment: {final_adjustment} dB (Base: {base_adjustment}, Intensity: {self.intensity})")
        return final_adjustment
    
    def generate_crazy_beat(self, output_path):
        print("Generating Crazy (Enhanced Rap) beat...")
        
        # 1. Setup Synthesizer and basics
        sr = 44100
        synth = Synthesizer(sr)
        
        # Use existing beat map
        if not self.beat_times:
            self.generate_beat_map()
            
        variation = max(0, min(4, self.intensity - 1)) # Map 1-5 to 0-4
        genre = 'rap'
        
        total_samples = int(sr * self.duration)
        drums = np.zeros(total_samples)
        bass = np.zeros(total_samples)
        chords = np.zeros(total_samples)
        
        full_beat_times = np.array(self.beat_times)
        
        def get_beat_time(beat_idx):
            idx_int = int(beat_idx)
            frac = beat_idx - idx_int
            
            if idx_int < 0: return 0.0
            if idx_int >= len(full_beat_times) - 1:
                last_interval = full_beat_times[-1] - full_beat_times[-2] if len(full_beat_times) > 1 else 60/120
                return full_beat_times[-1] + (beat_idx - (len(full_beat_times)-1)) * last_interval
                
            t1 = full_beat_times[idx_int]
            t2 = full_beat_times[idx_int+1]
            return t1 + frac * (t2 - t1)
            
        def add_sample(track, sample, start_time):
            start_sample = int(start_time * sr)
            
            if start_sample < 0:
                if start_sample + len(sample) <= 0: return
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

        # 2. Logic from voice_enhancer/main.py
        kick_sample = synth.kick(style=genre, variation=variation)
        snare_sample = synth.snare(style=genre, variation=variation)
        hihat_sample = synth.hihat(style=genre, variation=variation)
        
        # Harmony setup
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        base_freqs = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88]
        key_idx = 0 # Default C, could try to detect or use param
        
        # Minor scale for rap
        scale_intervals = [0, 2, 3, 5, 7, 8, 10]
        scale_freqs = []
        for interval in scale_intervals:
            idx = (key_idx + interval) % 12
            freq = base_freqs[idx]
            if (key_idx + interval) >= 12: freq *= 2
            scale_freqs.append(freq)
            
        # Rap Progressions
        progs = [
            [[0, 2, 4], [0, 2, 4], [5, 0, 2], [4, 6, 1]], # i i VI V
            [[0, 2, 4], [5, 0, 2], [0, 2, 4], [5, 0, 2]], # i VI i VI
            [[0, 2, 4], [0, 2, 4], [3, 5, 0], [4, 6, 1]], # i i iv V
            [[0, 2, 4], [2, 4, 6], [0, 2, 4], [4, 6, 1]], # i III i V
            [[5, 0, 2], [4, 6, 1], [5, 0, 2], [0, 2, 4]]  # VI V VI i
        ]
        progression = progs[variation % len(progs)]
        
        total_beats = len(full_beat_times)
        
        for i in range(total_beats):
            current_time = get_beat_time(i)
            
            # Drums
            if variation == 0: # Standard Trap
                if i % 4 == 0: add_sample(drums, kick_sample, get_beat_time(i))
                if i % 4 == 2: add_sample(drums, snare_sample, get_beat_time(i))
                if i % 2 == 0:
                    add_sample(drums, hihat_sample, get_beat_time(i))
                    add_sample(drums, hihat_sample, get_beat_time(i + 0.5))
                else:
                    add_sample(drums, hihat_sample, get_beat_time(i))
                    add_sample(drums, hihat_sample, get_beat_time(i + 0.25))
                    add_sample(drums, hihat_sample, get_beat_time(i + 0.5))
                    add_sample(drums, hihat_sample, get_beat_time(i + 0.75))
            elif variation == 1: # Boom Bap
                if i % 4 == 0: add_sample(drums, kick_sample, get_beat_time(i))
                if i % 4 == 2: add_sample(drums, snare_sample, get_beat_time(i))
                if i % 4 == 1: add_sample(drums, kick_sample, get_beat_time(i + 0.75))
                if i % 4 == 3: add_sample(drums, kick_sample, get_beat_time(i + 0.25))
                add_sample(drums, hihat_sample, get_beat_time(i))
                add_sample(drums, hihat_sample, get_beat_time(i + 0.66))
            elif variation == 2: # Drill-ish
                if i % 4 == 0: add_sample(drums, kick_sample, get_beat_time(i))
                if i % 4 == 2: add_sample(drums, snare_sample, get_beat_time(i))
                if i % 4 == 3: add_sample(drums, kick_sample, get_beat_time(i + 0.5))
                add_sample(drums, hihat_sample, get_beat_time(i))
                add_sample(drums, hihat_sample, get_beat_time(i + 0.33))
                add_sample(drums, hihat_sample, get_beat_time(i + 0.66))
            elif variation == 3: # Minimal
                if i % 4 == 0: add_sample(drums, kick_sample, get_beat_time(i))
                if i % 4 == 2: add_sample(drums, snare_sample, get_beat_time(i))
                if i % 2 == 0: add_sample(drums, hihat_sample, get_beat_time(i))
            else: # Industrial
                if i % 4 == 0: add_sample(drums, kick_sample, get_beat_time(i))
                if i % 4 == 1: add_sample(drums, kick_sample, get_beat_time(i))
                if i % 4 == 2: add_sample(drums, snare_sample, get_beat_time(i))
                if i % 4 == 3: add_sample(drums, kick_sample, get_beat_time(i))
                add_sample(drums, hihat_sample, get_beat_time(i))
                add_sample(drums, hihat_sample, get_beat_time(i + 0.25))
                add_sample(drums, hihat_sample, get_beat_time(i + 0.5))
                add_sample(drums, hihat_sample, get_beat_time(i + 0.75))
            
            # Bass & Chords
            bar = i // 4
            chord_idx = bar % 4
            chord_indices = progression[chord_idx]
            
            root_scale_idx = chord_indices[0]
            root_freq = scale_freqs[root_scale_idx] / 2
            
            # Bass pattern
            if i % 4 == 0: # Beat 1
                dur = get_beat_time(i+2) - get_beat_time(i)
                add_sample(bass, synth.bass(root_freq, dur, style=genre, variation=variation), get_beat_time(i))
            if i % 4 == 2 and variation % 2 == 1: # Beat 3
                dur = get_beat_time(i+2) - get_beat_time(i)
                add_sample(bass, synth.bass(root_freq, dur, style=genre, variation=variation), get_beat_time(i + 0.5))
                
            # Chords (Once per bar)
            if i % 4 == 0:
                chord_freqs = [scale_freqs[idx % 7] for idx in chord_indices]
                dur = get_beat_time(i+4) - get_beat_time(i)
                chord_sample = synth.chord(chord_freqs, dur, style=genre, variation=variation)
                add_sample(chords, chord_sample, get_beat_time(i))
                
        # 3. Mixing
        def normalize(arr):
            m = np.max(np.abs(arr))
            if m > 0: return arr / m
            return arr

        vocal_level = 1.0 # Not used here, this function generates beat only
        drum_level = 0.5 # Boost drums a bit
        bass_level = 0.4
        chord_level = 0.3
        
        final_mix = (
            normalize(drums) * drum_level +
            normalize(bass) * bass_level +
            normalize(chords) * chord_level
        )
        
        final_mix = np.clip(final_mix, -1.0, 1.0)
        
        # Save temp beat
        beat_path = self.vocal_path.rsplit('.', 1)[0] + '_crazy_beat.wav'
        sf.write(beat_path, final_mix, sr)
        
        return beat_path

    def mix_tracks(self, output_path):
        print(f"Generating {self.style} beat...")
        
        if self.style == 'crazy':
            beat_path = self.generate_crazy_beat(output_path)
            
            print("Mixing with vocals (Crazy mode)...")
            vocal = AudioSegment.from_file(self.vocal_path)
            beat_audio = AudioSegment.from_wav(beat_path)
            
            if len(beat_audio) > len(vocal):
                beat_audio = beat_audio[:len(vocal)]
            else:
                beat_audio = beat_audio + AudioSegment.silent(duration=len(vocal) - len(beat_audio))
            
            # Crazy volume adjustment
            beat_adjustment = -8 + (self.intensity - 3) * 2
            beat_audio = beat_audio + beat_adjustment
            
            mixed = vocal.overlay(beat_audio)
            mixed.export(output_path, format='wav')
            
            if os.path.exists(beat_path):
                os.remove(beat_path)
                
            return output_path

        # Standard mode
        self.generate_beat_map()
        
        drums = self.create_drum_pattern()
        bass = self.create_bassline()
        
        min_length = min(len(drums), len(bass))
        drums = drums[:min_length]
        bass = bass[:min_length]
        
        beat = (drums.astype(np.int32) + bass.astype(np.int32))
        beat = np.clip(beat, -32768, 32767).astype(np.int16)
        
        beat_path = self.vocal_path.rsplit('.', 1)[0] + '_temp_beat.wav'
        print(f"Saving temp beat to {beat_path}, shape: {beat.shape}, max: {np.max(np.abs(beat))}")
        sf.write(beat_path, beat, 44100)
        
        print("Mixing with vocals...")
        vocal = AudioSegment.from_file(self.vocal_path)
        beat_audio = AudioSegment.from_wav(beat_path)
        
        print(f"Vocal duration: {len(vocal)}ms, Beat duration: {len(beat_audio)}ms")
        
        if len(beat_audio) > len(vocal):
            beat_audio = beat_audio[:len(vocal)]
        else:
            beat_audio = beat_audio + AudioSegment.silent(duration=len(vocal) - len(beat_audio))
        
        beat_adjustment = self.calculate_optimal_beat_volume(vocal)
        beat_audio = beat_audio + beat_adjustment
        print(f"Applying beat adjustment: {beat_adjustment}dB")
        
        mixed = vocal.overlay(beat_audio)
        print(f"Exporting mixed audio to {output_path}")
        mixed.export(output_path, format='wav')
        
        if os.path.exists(beat_path):
            os.remove(beat_path)
        
        print(f"Done! Saved to: {output_path}")
        return output_path
