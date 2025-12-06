import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from scipy.interpolate import interp1d

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
    
    def mix_tracks(self, output_path):
        print(f"Generating {self.style} beat...")
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