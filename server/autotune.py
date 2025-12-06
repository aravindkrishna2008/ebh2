import numpy as np
import librosa
import soundfile as sf
import argparse
import os
import sys
import types
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
try:
    backend = types.ModuleType('torchaudio.backend')
    backend_common = types.ModuleType('torchaudio.backend.common')
    if 'torchaudio.backend' not in sys.modules:
        sys.modules['torchaudio.backend'] = backend
    if 'torchaudio.backend.common' not in sys.modules:
        sys.modules['torchaudio.backend.common'] = backend_common
    class AudioMetaData:
        def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding
    backend_common.AudioMetaData = AudioMetaData
    backend.common = backend_common
except Exception:
    pass
try:
    import pyworld as pw
    PYWORLD_AVAILABLE = True
except ImportError:
    PYWORLD_AVAILABLE = False
    print("Note: pyworld not available. Using librosa for pitch shifting.")
SPEECHBRAIN_AVAILABLE = False
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        def list_audio_backends():
            return ['soundfile']
        torchaudio.list_audio_backends = list_audio_backends
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from speechbrain.inference import enhancement
    SPEECHBRAIN_AVAILABLE = True
except Exception:
    SPEECHBRAIN_AVAILABLE = False
    pass
NOISEREDUCE_AVAILABLE = False
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    pass
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    pass
OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    pass
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
DEEPFILTERNET_AVAILABLE = False
DEEPFILTERNET_USE_API = False
try:
    from df.enhance import enhance, init_df
    import torch
    DEEPFILTERNET_AVAILABLE = True
    DEEPFILTERNET_USE_API = True
except (ImportError, ModuleNotFoundError, Exception):
    try:
        import subprocess
        result = subprocess.run(['deepFilter', '--help'], 
                              capture_output=True, 
                              timeout=2,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode == 0 or 'deepFilter' in result.stderr.decode('utf-8', errors='ignore'):
            DEEPFILTERNET_AVAILABLE = True
    except (ImportError, ModuleNotFoundError, FileNotFoundError, subprocess.TimeoutExpired):
        DEEPFILTERNET_AVAILABLE = False
        pass
BASIC_PITCH_AVAILABLE = False
try:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    import basic_pitch.note_creation as note_creation
    import tensorflow as tf
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    pass
def detect_pitch(y, sr, frame_length=2048, hop_length=512, use_pyworld=False):
    if use_pyworld and PYWORLD_AVAILABLE:
        try:
            frame_period = (hop_length / sr) * 1000.0
            f0_dio, timeaxis = pw.dio(y.astype(np.float64), sr, frame_period=frame_period)
            f0 = pw.stonemask(y.astype(np.float64), f0_dio, timeaxis, sr)
            expected_frames = int(len(y) / hop_length) + 1
            if len(f0) != expected_frames:
                original_indices = np.linspace(0, len(f0)-1, expected_frames)
                interp_func = interp1d(np.arange(len(f0)), f0, kind='linear',
                                      bounds_error=False, fill_value=0.0)
                f0 = interp_func(original_indices)
            f0[f0 < 0] = 0
            pitches = f0
            return pitches
        except Exception as e:
            print(f"PyWorld pitch detection failed, using librosa: {e}")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )
    pitches = np.nan_to_num(f0, nan=0.0)
    return pitches
def snap_to_scale(pitch, scale='chromatic', root_note=440, correction_threshold=0.2):
    if pitch <= 0:
        return pitch, 0.0
    semitones = 12 * np.log2(pitch / root_note)
    if scale == 'chromatic':
        semitones_target = np.round(semitones)
    elif scale == 'major':
        major_scale = [0, 2, 4, 5, 7, 9, 11]
        octave = int(semitones // 12)
        note_in_octave = semitones % 12
        distances = [abs(note_in_octave - note) for note in major_scale]
        nearest_idx = np.argmin(distances)
        semitones_target = octave * 12 + major_scale[nearest_idx]
    elif scale == 'minor':
        minor_scale = [0, 2, 3, 5, 7, 8, 10]
        octave = int(semitones // 12)
        note_in_octave = semitones % 12
        distances = [abs(note_in_octave - note) for note in minor_scale]
        nearest_idx = np.argmin(distances)
        semitones_target = octave * 12 + minor_scale[nearest_idx]
    else:
        semitones_target = np.round(semitones)
    deviation = abs(semitones - semitones_target)
    if deviation < correction_threshold:
        correction_amount = (deviation / correction_threshold) * 0.15
        blend = correction_amount
        corrected_semitones = semitones * (1 - blend) + semitones_target * blend
        corrected_pitch = root_note * (2 ** (corrected_semitones / 12))
    else:
        excess_deviation = deviation - correction_threshold
        correction_amount = 0.2 + min(0.4, excess_deviation / 0.5)
        corrected_pitch = root_note * (2 ** (semitones_target / 12))
    return corrected_pitch, correction_amount
def pitch_shift_pyworld(y, sr, pitch_ratio):
    if pitch_ratio == 1.0:
        return y
    y = y.astype(np.float64)
    frame_period = 5.0
    f0, timeaxis = pw.harvest(y, sr, frame_period=frame_period)
    sp = pw.cheaptrick(y, f0, timeaxis, sr)
    ap = pw.d4c(y, f0, timeaxis, sr)
    f0_shifted = f0 * pitch_ratio
    f0_shifted[f0 == 0] = 0
    y_shifted = pw.synthesize(f0_shifted, sp, ap, sr, frame_period=frame_period)
    if len(y_shifted) != len(y):
        original_indices = np.linspace(0, len(y_shifted)-1, len(y))
        interp_func = interp1d(np.arange(len(y_shifted)), y_shifted, kind='linear', 
                              bounds_error=False, fill_value=0.0)
        y_shifted = interp_func(original_indices)
    return y_shifted.astype(np.float32)
def pitch_shift_librosa(y, sr, pitch_ratio):
    if pitch_ratio == 1.0:
        return y
    y_shifted = librosa.effects.pitch_shift(
        y, 
        sr=sr, 
        n_steps=12 * np.log2(pitch_ratio),
        bins_per_octave=12
    )
    return y_shifted
def pitch_shift_audio(y, sr, pitch_ratio, use_pyworld=True):
    if pitch_ratio == 1.0:
        return y
    if use_pyworld and PYWORLD_AVAILABLE:
        try:
            return pitch_shift_pyworld(y, sr, pitch_ratio)
        except Exception as e:
            print(f"PyWorld pitch shift failed, using librosa: {e}")
            return pitch_shift_librosa(y, sr, pitch_ratio)
    else:
        return pitch_shift_librosa(y, sr, pitch_ratio)
def smooth_pitch_curve(pitches, alpha=0.6, preserve_variation=True):
    if len(pitches) == 0:
        return pitches
    smoothed = np.zeros_like(pitches)
    smoothed[0] = pitches[0] if pitches[0] > 0 else 0
    for i in range(1, len(pitches)):
        if pitches[i] > 0:
            if smoothed[i-1] > 0:
                pitch_change = abs(pitches[i] - smoothed[i-1]) / smoothed[i-1] if smoothed[i-1] > 0 else 0
                if preserve_variation:
                    if i > 2:
                        recent_changes = [abs(pitches[j] - pitches[j-1]) / pitches[j-1] 
                                        for j in range(max(0, i-3), i) if pitches[j] > 0 and pitches[j-1] > 0]
                        if recent_changes and np.std(recent_changes) < 0.02:
                            adaptive_alpha = min(0.8, alpha + 0.2)
                        else:
                            adaptive_alpha = max(0.3, alpha - 0.1)
                    else:
                        adaptive_alpha = alpha
                else:
                    adaptive_alpha = alpha
                smoothed[i] = adaptive_alpha * pitches[i] + (1 - adaptive_alpha) * smoothed[i-1]
            else:
                smoothed[i] = pitches[i]
        else:
            smoothed[i] = smoothed[i-1] if smoothed[i-1] > 0 else 0
    return smoothed
def apply_autotune(y, sr, strength=0.4, scale='chromatic', root_note=440, correction_threshold=0.2, use_pyworld_pitch=False):
    if use_pyworld_pitch and PYWORLD_AVAILABLE:
        print("Detecting pitch with PyWorld DIO/STONEMASK...")
    else:
        print("Detecting pitch with PYIN algorithm...")
    hop_length = 512
    frame_length = 2048
    pitches = detect_pitch(y, sr, frame_length=frame_length, hop_length=hop_length, use_pyworld=use_pyworld_pitch)
    frame_times = np.arange(len(pitches)) * hop_length / sr
    valid_mask = pitches > 0
    if not np.any(valid_mask):
        print("Warning: No valid pitch detected. Returning original audio.")
        return y
    print("Smoothing pitch curve (preserving natural variations)...")
    smoothed_pitches = smooth_pitch_curve(pitches, alpha=0.6, preserve_variation=True)
    valid_mask_smooth = smoothed_pitches > 0
    if not np.any(valid_mask_smooth):
        print("Warning: No valid pitch after smoothing. Returning original audio.")
        return y
    print("Correcting pitch (adaptive, musical correction)...")
    valid_smoothed = smoothed_pitches[valid_mask_smooth]
    valid_times = frame_times[valid_mask_smooth]
    corrected_data = [snap_to_scale(p, scale, root_note, correction_threshold) 
                      for p in valid_smoothed]
    if len(corrected_data) == 0:
        print("Warning: No valid corrections. Returning original audio.")
        return y
    corrected_pitches = np.array([d[0] for d in corrected_data])
    correction_amounts = np.array([d[1] for d in corrected_data])
    valid_pitches_smooth = valid_smoothed
    pitch_ratios = corrected_pitches / valid_pitches_smooth
    adaptive_strength = strength * correction_amounts
    adaptive_strength_smooth = gaussian_filter1d(adaptive_strength, sigma=2.0)
    pitch_ratios_smooth = gaussian_filter1d(pitch_ratios, sigma=1.5)
    pitch_ratio_func = interp1d(
        valid_times, 
        pitch_ratios_smooth,
        kind='cubic',
        fill_value='extrapolate',
        bounds_error=False
    )
    strength_func = interp1d(
        valid_times,
        adaptive_strength_smooth,
        kind='cubic',
        fill_value=0.0,
        bounds_error=False
    )
    print("Applying smooth pitch correction...")
    segment_length = int(sr * 2.0)
    overlap = int(sr * 0.3)
    output = np.zeros_like(y)
    window = np.hanning(overlap * 2)
    for start_idx in range(0, len(y), segment_length - overlap):
        end_idx = min(start_idx + segment_length, len(y))
        segment = y[start_idx:end_idx]
        segment_times = np.linspace(start_idx/sr, end_idx/sr, len(segment))
        segment_ratios = pitch_ratio_func(segment_times)
        segment_strengths = strength_func(segment_times)
        valid_ratios = segment_ratios[(segment_ratios > 0) & (segment_ratios != 1.0)]
        if len(valid_ratios) == 0:
            if start_idx > 0:
                fade_in = window[:overlap]
                fade_out = window[overlap:]
                output[start_idx:start_idx+overlap] += segment[:overlap] * fade_in
                output[start_idx+overlap:end_idx] += segment[overlap:]
            else:
                output[start_idx:end_idx] += segment
            continue
        median_ratio = np.median(valid_ratios)
        if median_ratio <= 0 or median_ratio == 1.0:
            if start_idx > 0:
                fade_in = window[:overlap]
                fade_out = window[overlap:]
                output[start_idx:start_idx+overlap] += segment[:overlap] * fade_in
                output[start_idx+overlap:end_idx] += segment[overlap:]
            else:
                output[start_idx:end_idx] += segment
            continue
        valid_strengths = segment_strengths[segment_strengths > 0]
        if len(valid_strengths) == 0:
            if start_idx > 0:
                fade_in = window[:overlap]
                fade_out = window[overlap:]
                output[start_idx:start_idx+overlap] += segment[:overlap] * fade_in
                output[start_idx+overlap:end_idx] += segment[overlap:]
            else:
                output[start_idx:end_idx] += segment
            continue
        median_strength = np.median(valid_strengths)
        if median_strength < 0.12:
            if start_idx > 0:
                fade_in = window[:overlap]
                fade_out = window[overlap:]
                output[start_idx:start_idx+overlap] += segment[:overlap] * fade_in
                output[start_idx+overlap:end_idx] += segment[overlap:]
            else:
                output[start_idx:end_idx] += segment
            continue
        target_ratio = 1.0 + (median_ratio - 1.0) * median_strength * 0.7
        if abs(target_ratio - 1.0) > 0.012:
            corrected_segment = pitch_shift_audio(
                segment,
                sr=sr,
                pitch_ratio=target_ratio,
                use_pyworld=True
            )
            blend_factor = min(0.65, median_strength * 0.75)
            segment = segment * (1 - blend_factor) + corrected_segment * blend_factor
        if start_idx > 0:
            fade_in = window[:overlap]
            fade_out = window[overlap:]
            output[start_idx:start_idx+overlap] += segment[:overlap] * fade_in
            output[start_idx+overlap:end_idx] += segment[overlap:]
        else:
            output[start_idx:end_idx] += segment
    print("Adding subtle natural variation...")
    variation_amount = 0.02
    variation_freq = 5.0
    t = np.arange(len(output)) / sr
    pitch_variation = 1.0 + variation_amount * np.sin(2 * np.pi * variation_freq * t)
    if PYWORLD_AVAILABLE:
        try:
            frame_period = 5.0
            f0, timeaxis = pw.harvest(output.astype(np.float64), sr, frame_period=frame_period)
            sp = pw.cheaptrick(output.astype(np.float64), f0, timeaxis, sr)
            ap = pw.d4c(output.astype(np.float64), f0, timeaxis, sr)
            f0_varied = f0.copy()
            for i, freq in enumerate(f0):
                if freq > 0:
                    frame_time = timeaxis[i] / 1000.0
                    if frame_time < len(pitch_variation) / sr:
                        idx = int(frame_time * sr)
                        if idx < len(pitch_variation):
                            f0_varied[i] = freq * pitch_variation[idx]
            output = pw.synthesize(f0_varied, sp, ap, sr, frame_period=frame_period)
            if len(output) != len(t):
                original_indices = np.linspace(0, len(output)-1, len(t))
                interp_func = interp1d(np.arange(len(output)), output, kind='linear',
                                      bounds_error=False, fill_value=0.0)
                output = interp_func(original_indices)
            output = output.astype(np.float32)
        except:
            pass
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * 0.95
    return output
def apply_eq(y, sr, low_gain=0, mid_gain=0, high_gain=0):
    nyquist = sr / 2
    low_cutoff = 200 / nyquist
    if low_gain != 0:
        b, a = signal.iirfilter(4, low_cutoff, btype='low', ftype='butter')
        y_low = signal.filtfilt(b, a, y)
        y = y + y_low * (10**(low_gain/20) - 1)
    high_cutoff = 5000 / nyquist
    if high_gain != 0:
        b, a = signal.iirfilter(4, high_cutoff, btype='high', ftype='butter')
        y_high = signal.filtfilt(b, a, y)
        y = y + y_high * (10**(high_gain/20) - 1)
    if mid_gain != 0:
        low_mid = 200 / nyquist
        high_mid = 5000 / nyquist
        b, a = signal.iirfilter(4, [low_mid, high_mid], btype='band', ftype='butter')
        y_mid = signal.filtfilt(b, a, y)
        y = y + y_mid * (10**(mid_gain/20) - 1)
    return y
def apply_compression(y, threshold=0.7, ratio=4.0, attack=0.003, release=0.1, sr=44100):
    envelope = np.abs(y)
    attack_samples = max(1, int(attack * sr))
    release_samples = max(1, int(release * sr))
    attack_alpha = 1 - np.exp(-1 / attack_samples)
    release_alpha = 1 - np.exp(-1 / release_samples)
    smoothed = np.zeros_like(envelope)
    smoothed[0] = envelope[0]
    prev = envelope[0]
    for i in range(1, len(envelope)):
        if envelope[i] > prev:
            prev = attack_alpha * envelope[i] + (1 - attack_alpha) * prev
        else:
            prev = release_alpha * envelope[i] + (1 - release_alpha) * prev
        smoothed[i] = prev
    over_threshold = smoothed > threshold
    gain_reduction = np.ones_like(smoothed)
    if np.any(over_threshold):
        gain_reduction[over_threshold] = (threshold + (smoothed[over_threshold] - threshold) / ratio) / smoothed[over_threshold]
    output = y * gain_reduction
    return output
def enhance_clarity(y, sr, preserve_details=True, intensity=0.8):
    nyquist = sr / 2
    if preserve_details:
        presence_low = 3000 / nyquist
        presence_high = 5000 / nyquist
        if presence_high < 0.95:
            b, a = signal.iirfilter(4, [presence_low, presence_high], btype='band', ftype='butter')
            y_presence = signal.filtfilt(b, a, y)
            y = y + y_presence * (0.08 * intensity)
        brilliance_low = 5000 / nyquist
        brilliance_high = min(8000 / nyquist, 0.95)
        if brilliance_high < 0.95:
            b, a = signal.iirfilter(3, brilliance_low, btype='high', ftype='butter')
            y_brilliance = signal.filtfilt(b, a, y)
            brilliance_mask = np.fft.rfftfreq(len(y), 1/sr) >= 5000
            if np.any(brilliance_mask):
                Y = np.fft.rfft(y)
                Y[brilliance_mask] *= (1 + 0.05 * intensity)
                y = np.fft.irfft(Y)[:len(y)]
        formant_low = 1500 / nyquist
        formant_high = 3000 / nyquist
        if formant_high < 0.95:
            b, a = signal.iirfilter(4, [formant_low, formant_high], btype='band', ftype='butter')
            y_formant = signal.filtfilt(b, a, y)
            y = y + y_formant * (0.06 * intensity)
        low_cutoff = 60 / nyquist
        b, a = signal.iirfilter(3, low_cutoff, btype='high', ftype='butter')
        y_low_clean = signal.filtfilt(b, a, y)
        y = y_low_clean * 0.97 + y * 0.03
        if intensity > 0.5:
            deess_low = 4000 / nyquist
            deess_high = min(8000 / nyquist, 0.95)
            if deess_high < 0.95:
                b, a = signal.iirfilter(4, [deess_low, deess_high], btype='band', ftype='butter')
                y_sibilants = signal.filtfilt(b, a, y)
                sibilant_envelope = np.abs(y_sibilants)
                harsh_threshold = np.percentile(sibilant_envelope, 85)
                harsh_mask = sibilant_envelope > harsh_threshold
                if np.any(harsh_mask):
                    reduction = 0.15 * intensity
                    y = y - y_sibilants * harsh_mask * reduction
    else:
        high_cutoff = 3000 / nyquist
        b, a = signal.iirfilter(4, high_cutoff, btype='high', ftype='butter')
        y_high = signal.filtfilt(b, a, y)
        y = y + y_high * 0.12
        mid_low = 2000 / nyquist
        mid_high = 4000 / nyquist
        b, a = signal.iirfilter(4, [mid_low, mid_high], btype='band', ftype='butter')
        y_mid = signal.filtfilt(b, a, y)
        y = y + y_mid * 0.08
        low_cutoff = 100 / nyquist
        b, a = signal.iirfilter(2, low_cutoff, btype='high', ftype='butter')
        y_low_clean = signal.filtfilt(b, a, y)
        y = y_low_clean * 0.95 + y * 0.05
    return y
def enhance_with_elevenlabs_api(y, sr, api_key=None, intensity=0.7):
    if not REQUESTS_AVAILABLE:
        return y
    if intensity <= 0:
        return y
    if api_key is None:
        api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        return y
    print("Attempting ElevenLabs API (may not be available for voice enhancement)...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, y, sr)
        try:
            url = "https://api.elevenlabs.io/v1/audio/enhancement"
            with open(tmp_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                headers = {
                    'xi-api-key': api_key
                }
                data = {
                    'output_format': 'wav',
                    'output_quality': 'high'
                }
                response = requests.post(url, files=files, headers=headers, data=data, timeout=120)
                if response.status_code == 200:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as enhanced_file:
                        enhanced_path = enhanced_file.name
                        enhanced_file.write(response.content)
                    y_enhanced, sr_enhanced = librosa.load(enhanced_path, sr=sr, mono=True)
                    try:
                        os.unlink(enhanced_path)
                    except:
                        pass
                    min_len = min(len(y), len(y_enhanced))
                    y_result = y[:min_len] * (1 - intensity) + y_enhanced[:min_len] * intensity
                    print(f"  ElevenLabs enhancement successful (blend: {intensity*100:.0f}%)")
                    return y_result
                else:
                    print(f"  ElevenLabs enhancement endpoint not available (status: {response.status_code})")
                    print("  Falling back to local AI models (SpeechBrain, noisereduce)...")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        pass
    return y
def enhance_with_openai_api(y, sr, api_key=None, intensity=0.6):
    if not OPENAI_AVAILABLE or not REQUESTS_AVAILABLE:
        return y
    if intensity <= 0:
        return y
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return y
    print("Using OpenAI Whisper for vocal analysis and enhancement...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, y, sr)
        try:
            client = openai.OpenAI(api_key=api_key)
            with open(tmp_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            if hasattr(transcript, 'language') and transcript.language:
                print(f"  Detected language: {transcript.language}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        print(f"  OpenAI API failed: {e}")
    return y
def enhance_with_deepfilternet(y, sr, intensity=0.7):
    if not DEEPFILTERNET_AVAILABLE:
        return y
    if intensity <= 0:
        return y
    print("Using DeepFilterNet for state-of-the-art speech enhancement...")
    if DEEPFILTERNET_USE_API:
        try:
            model, df_state, _ = init_df()
            target_sr = 48000
            if sr != target_sr:
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            else:
                y_resampled = y
            audio_tensor = torch.from_numpy(y_resampled).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            enhanced_tensor = enhance(model, df_state, audio_tensor)
            enhanced = enhanced_tensor.squeeze().cpu().numpy()
            if sr != target_sr:
                enhanced = librosa.resample(enhanced, orig_sr=target_sr, target_sr=sr)
            min_len = min(len(y), len(enhanced))
            if intensity >= 0.9:
                y_result = enhanced[:min_len]
                print(f"  DeepFilterNet enhancement successful (API mode, full strength)")
            else:
                y_result = y[:min_len] * (1 - intensity) + enhanced[:min_len] * intensity
                print(f"  DeepFilterNet enhancement successful (API mode, blend: {intensity*100:.0f}%)")
            return y_result
        except Exception as e:
            print(f"  DeepFilterNet API failed: {e}, trying command line fallback...")
            pass
    try:
        import tempfile
        import subprocess
        target_sr = 48000
        if sr != target_sr:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        else:
            y_resampled = y
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, y_resampled, target_sr)
        out_path = tmp_path.replace('.wav', '_enhanced.wav')
        try:
            cmd = ['deepFilter', tmp_path, out_path]
            if not os.path.exists(cmd[0]) and 'deep-filter-py' in str(subprocess.run(['where', 'deep-filter-py'], capture_output=True).stdout):
                cmd[0] = 'deep-filter-py'
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            if result.returncode == 0 and os.path.exists(out_path):
                enhanced, sr_enhanced = librosa.load(out_path, sr=target_sr, mono=True)
                min_len = min(len(y_resampled), len(enhanced))
                enhanced = enhanced[:min_len]
                if sr != target_sr:
                    enhanced = librosa.resample(enhanced, orig_sr=target_sr, target_sr=sr)
                min_len = min(len(y), len(enhanced))
                y_result = y[:min_len] * (1 - intensity) + enhanced[:min_len] * intensity
                print(f"  DeepFilterNet enhancement successful (blend: {intensity*100:.0f}%)")
                return y_result
            else:
                print(f"  DeepFilterNet command failed: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
            try:
                if os.path.exists(out_path):
                    os.unlink(out_path)
            except:
                pass
    except Exception as e:
        print(f"  DeepFilterNet failed: {e}")
    return y
def enhance_with_ai_api(y, sr, intensity=0.5, elevenlabs_key=None, openai_key=None):
    if intensity <= 0:
        return y
    if DEEPFILTERNET_AVAILABLE:
        df_intensity = intensity if intensity < 0.8 else 1.0
        y_enhanced = enhance_with_deepfilternet(y, sr, intensity=df_intensity)
        if not np.array_equal(y, y_enhanced):
            y = y_enhanced
            if intensity > 0.5:
                print("  Skipping/Reducing other noise reduction steps (DeepFilterNet was successful)")
                intensity = intensity * 0.3
    if elevenlabs_key or os.getenv('ELEVENLABS_API_KEY'):
        y = enhance_with_elevenlabs_api(y, sr, api_key=elevenlabs_key, intensity=intensity * 0.8)
    if openai_key or os.getenv('OPENAI_API_KEY'):
        y = enhance_with_openai_api(y, sr, api_key=openai_key, intensity=intensity * 0.3)
    if NOISEREDUCE_AVAILABLE:
        print("Using noisereduce AI enhancement (local model, no API keys)...")
        try:
            try:
                y_enhanced = nr.reduce_noise(
                    y=y,
                    sr=sr,
                    stationary=False,
                    prop_decrease=intensity * 0.6,
                    n_fft=2048,
                    win_length=2048,
                    hop_length=512
                )
            except TypeError:
                y_enhanced = nr.reduce_noise(
                    y=y,
                    sr=sr,
                    stationary=False,
                    prop_decrease=intensity * 0.6
                )
            blend_factor = intensity * 0.5
            y = y * (1 - blend_factor) + y_enhanced * blend_factor
        except Exception as e:
            print(f"  noisereduce failed: {e}, falling back to spectral subtraction...")
    if intensity > 0.3:
        print("Using advanced spectral subtraction enhancement...")
        frame_length = 2048
        hop_length = 512
        stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        frame_energies = np.mean(magnitude, axis=0)
        noise_threshold = np.percentile(frame_energies, 10)
        noise_mask = frame_energies < noise_threshold
        if np.any(noise_mask):
            noise_spectrum = np.mean(magnitude[:, noise_mask], axis=1, keepdims=True)
            alpha = 2.5 * intensity
            beta = 0.1 * intensity
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            spectral_floor = beta * magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, spectral_floor)
            strong_signal_mask = magnitude > 2 * noise_spectrum
            enhanced_magnitude[strong_signal_mask] = magnitude[strong_signal_mask] * (1 - intensity * 0.15) + enhanced_magnitude[strong_signal_mask] * (intensity * 0.15)
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            y_enhanced = librosa.istft(enhanced_stft, hop_length=hop_length)
            min_len = min(len(y), len(y_enhanced))
            blend = intensity * 0.4
            y = y[:min_len] * (1 - blend) + y_enhanced[:min_len] * blend
    return y
def polish_with_speechbrain(y, sr, intensity=0.6):
    if not SPEECHBRAIN_AVAILABLE:
        print("SpeechBrain not available, skipping AI polishing...")
        return y
    print(f"Applying SpeechBrain AI polishing (intensity: {intensity})...")
    try:
        import torch
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enhancer = None
            model_sources = [
                "speechbrain/metricgan-plus-voicebank",
                "speechbrain/sepformer-whamr-enhancement",
                "speechbrain/mtl-mimic-voicebank"
            ]
            for model_source in model_sources:
                try:
                    enhancer = enhancement.SpectralMaskEnhancement.from_hparams(
                        source=model_source,
                        savedir=f"pretrained_models/{model_source.split('/')[-1]}",
                        run_opts={"device": "cpu"}
                    )
                    break
                except Exception:
                    continue
            if enhancer is None:
                print("  Could not load SpeechBrain models, skipping AI polish")
                return y
        target_sr = 16000
        if sr != target_sr:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        else:
            y_resampled = y
        if isinstance(y_resampled, np.ndarray):
            y_tensor = torch.from_numpy(y_resampled).float()
        else:
            y_tensor = torch.tensor(y_resampled).float()
        chunk_length = int(target_sr * 10)
        enhanced_chunks = []
        failed_chunks = 0
        total_chunks = (len(y_resampled) + chunk_length - 1) // chunk_length
        for start_idx in range(0, len(y_resampled), chunk_length):
            end_idx = min(start_idx + chunk_length, len(y_resampled))
            chunk = y_resampled[start_idx:end_idx]
            if isinstance(chunk, np.ndarray):
                chunk_tensor = torch.from_numpy(chunk).float()
            else:
                chunk_tensor = torch.tensor(chunk).float()
            if chunk_tensor.dim() == 1:
                chunk_tensor = chunk_tensor.unsqueeze(0)
            with torch.no_grad():
                try:
                    if chunk_tensor.shape[0] != 1:
                        chunk_tensor = chunk_tensor.unsqueeze(0) if chunk_tensor.dim() == 1 else chunk_tensor
                    enhanced_chunk_tensor = enhancer.enhance_batch(chunk_tensor)
                    if isinstance(enhanced_chunk_tensor, torch.Tensor):
                        enhanced_chunk = enhanced_chunk_tensor.squeeze().cpu().numpy()
                    else:
                        enhanced_chunk = enhanced_chunk_tensor
                except Exception as e:
                    import tempfile
                    import os as os_module
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                            sf.write(tmp_path, chunk, target_sr)
                        if hasattr(enhancer, 'enhance_file'):
                            enhanced_chunk_tensor = enhancer.enhance_file(tmp_path)
                            if isinstance(enhanced_chunk_tensor, torch.Tensor):
                                enhanced_chunk = enhanced_chunk_tensor.squeeze().cpu().numpy()
                            else:
                                enhanced_chunk = enhanced_chunk_tensor
                        else:
                            raise AttributeError("enhance_file not available")
                    except Exception as e2:
                        failed_chunks += 1
                        enhanced_chunk = chunk
                    finally:
                        if tmp_path is not None:
                            try:
                                os_module.unlink(tmp_path)
                            except:
                                pass
            if isinstance(enhanced_chunk, torch.Tensor):
                enhanced_chunk = enhanced_chunk.squeeze().cpu().numpy()
            if enhanced_chunk.ndim > 1:
                enhanced_chunk = enhanced_chunk[0] if enhanced_chunk.shape[0] == 1 else enhanced_chunk.flatten()
            enhanced_chunks.append(enhanced_chunk)
        if failed_chunks >= total_chunks * 0.5:
            print(f"  SpeechBrain failed on {failed_chunks}/{total_chunks} chunks, skipping AI polish")
            return y
        if failed_chunks == total_chunks:
            print(f"  SpeechBrain enhancement not working, skipping AI polish")
            return y
        enhanced = np.concatenate(enhanced_chunks) if len(enhanced_chunks) > 1 else enhanced_chunks[0]
        if sr != target_sr:
            enhanced = librosa.resample(enhanced, orig_sr=target_sr, target_sr=sr)
        min_len = min(len(y), len(enhanced))
        y_polished = y[:min_len] * (1 - intensity) + enhanced[:min_len] * intensity
        return y_polished
    except Exception as e:
        print(f"SpeechBrain enhancement failed: {e}")
        print("Falling back to traditional enhancement methods...")
        import traceback
        traceback.print_exc()
        return y
def remove_artifacts(y, sr):
    from scipy.ndimage import median_filter
    window_size = max(3, int(sr * 0.0001))
    if window_size % 2 == 0:
        window_size += 1
    y_cleaned = median_filter(y, size=window_size, mode='constant')
    envelope = np.abs(y_cleaned)
    envelope_smooth = median_filter(envelope, size=max(3, int(sr * 0.001)), mode='constant')
    diff = np.abs(envelope - envelope_smooth)
    threshold = np.percentile(diff, 98)
    glitch_mask = diff > threshold
    if np.any(glitch_mask):
        from scipy.ndimage import gaussian_filter1d
        y_smooth = gaussian_filter1d(y_cleaned, sigma=2.0)
        blend_factor = 0.5
        y_cleaned = np.where(glitch_mask, y_smooth * blend_factor + y_cleaned * (1 - blend_factor), y_cleaned)
    nyquist = sr / 2
    highpass_cutoff = 20 / nyquist
    if highpass_cutoff < 0.95:
        b, a = signal.butter(2, highpass_cutoff, btype='high')
        y_cleaned = signal.filtfilt(b, a, y_cleaned)
    return y_cleaned
def enhance_enthusiasm(y, sr, intensity=0.5):
    print(f"Applying enthusiasm enhancement (intensity: {intensity})...")
    if intensity <= 0:
        return y
    effective_intensity = intensity * 0.6
    nyquist = sr / 2
    original_max = np.max(np.abs(y))
    if effective_intensity > 0.1:
        fft_size = len(y)
        Y = np.fft.rfft(y, n=fft_size)
        freqs = np.fft.rfftfreq(fft_size, 1/sr)
        boost = np.ones_like(freqs)
        vocal_mask = (freqs >= 2000) & (freqs <= 5000)
        boost[vocal_mask] += 0.08 * effective_intensity
        bright_mask = (freqs >= 5000) & (freqs <= 10000)
        boost[bright_mask] += 0.05 * effective_intensity
        formant_mask = (freqs >= 1000) & (freqs <= 2000)
        boost[formant_mask] += 0.04 * effective_intensity
        from scipy.ndimage import gaussian_filter1d
        boost = gaussian_filter1d(boost, sigma=2.0)
        Y *= boost
        y = np.fft.irfft(Y, n=fft_size)[:len(y)]
    envelope = np.abs(y)
    from scipy.ndimage import uniform_filter1d
    window_size = max(1, int(sr * 0.01))
    envelope_smooth = uniform_filter1d(envelope, size=window_size, mode='constant')
    threshold = np.percentile(envelope_smooth, 30)
    quiet_mask = envelope_smooth < threshold
    if np.any(quiet_mask) and effective_intensity > 0.2:
        expansion_ratio = 1.0 + effective_intensity * 0.15
        gain_map = np.ones_like(y)
        gain_map[quiet_mask] = 1.0 + (threshold - envelope_smooth[quiet_mask]) / (threshold + 1e-10) * expansion_ratio * 0.3
        from scipy.ndimage import gaussian_filter1d
        gain_map = gaussian_filter1d(gain_map, sigma=int(sr * 0.0005))
        y = y * gain_map
    if effective_intensity > 0.3:
        from scipy.ndimage import gaussian_filter1d
        envelope = np.abs(y)
        envelope_smooth = gaussian_filter1d(envelope, sigma=int(sr * 0.002))
        transient = envelope - envelope_smooth
        transient_boost = 1.0 + np.clip(transient * effective_intensity * 0.1, -0.1, 0.15)
        transient_boost = gaussian_filter1d(transient_boost, sigma=int(sr * 0.0003))
        y = y * transient_boost
    if effective_intensity > 0.2:
        saturation_amount = effective_intensity * 0.08
        y_saturated = np.tanh(y * (1 + saturation_amount)) / (1 + saturation_amount)
        y = y * (1 - saturation_amount * 0.2) + y_saturated * (saturation_amount * 0.2)
    if original_max > 0:
        y_max = np.max(np.abs(y))
        if y_max > 0:
            y = y / y_max * original_max * 0.98
    return y
def process_audio(input_file, output_file, autotune_strength=0.4, scale='chromatic', 
                  root_note=440, apply_eq_enabled=False, low_gain=0, mid_gain=0, 
                  high_gain=0, apply_compression_enabled=False, threshold=0.7, 
                  ratio=4.0, auto_enhance=True, use_pyworld=True, use_pyworld_pitch=False,
                  enthusiasm_intensity=0.15, speechbrain_polish=True, polish_intensity=0.6,
                  elevenlabs_key=None, openai_key=None, clarity_intensity=0.8):
    print(f"Loading audio from {input_file}...")
    try:
        y, sr = librosa.load(input_file, sr=None, mono=True)
        print(f"Loaded audio: {len(y)/sr:.2f} seconds, {sr} Hz sample rate")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    if polish_intensity > 0:
        print(f"\nApplying AI-powered enhancement (intensity: {polish_intensity})...")
        y = enhance_with_ai_api(y, sr, intensity=polish_intensity, 
                               elevenlabs_key=elevenlabs_key, openai_key=openai_key)
    if speechbrain_polish and polish_intensity > 0:
        y = polish_with_speechbrain(y, sr, intensity=polish_intensity * 0.4)
    if auto_enhance:
        print(f"\nApplying advanced clarity enhancement (intensity: {clarity_intensity})...")
        y = enhance_clarity(y, sr, preserve_details=True, intensity=clarity_intensity)
    if autotune_strength > 0:
        if use_pyworld and PYWORLD_AVAILABLE:
            print(f"\nApplying musical autotune with PyWorld (strength: {autotune_strength}, scale: {scale})...")
        else:
            print(f"\nApplying musical autotune (strength: {autotune_strength}, scale: {scale})...")
        y = apply_autotune(y, sr, strength=autotune_strength, scale=scale, root_note=root_note, use_pyworld_pitch=use_pyworld_pitch)
    if enthusiasm_intensity > 0:
        y = enhance_enthusiasm(y, sr, intensity=enthusiasm_intensity)
    print("\nRemoving artifacts and glitches (preserving details)...")
    y = remove_artifacts(y, sr)
    print("\nApplying minimal compression for consistency...")
    y = apply_compression(y, threshold=0.9, ratio=2.0, attack=0.01, release=0.2, sr=sr)
    if apply_eq_enabled:
        print(f"\nApplying EQ (low: {low_gain}dB, mid: {mid_gain}dB, high: {high_gain}dB)...")
        y = apply_eq(y, sr, low_gain=low_gain, mid_gain=mid_gain, high_gain=high_gain)
    if apply_compression_enabled:
        print(f"\nApplying additional compression (threshold: {threshold}, ratio: {ratio})...")
        y = apply_compression(y, threshold=threshold, ratio=ratio, sr=sr)
    y_max = np.max(np.abs(y))
    if y_max > 0:
        y = y * (0.95 / y_max)
    print(f"\nSaving to {output_file}...")
    try:
        sf.write(output_file, y, sr)
        print(f"Successfully saved processed audio to {output_file}")
    except Exception as e:
        print(f"Error saving audio: {e}")
def main():
    parser = argparse.ArgumentParser(
        description='Autotune and enhance MP3 audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic autotune
  python autotune.py input.mp3 output.mp3
  # Strong autotune with major scale
  python autotune.py input.mp3 output.mp3 --strength 0.9 --scale major
  # Autotune with EQ and compression
  python autotune.py input.mp3 output.mp3 --eq --low-gain 2 --high-gain 1 --compress
  # Autotune with maximum enthusiasm enhancement
  python autotune.py input.mp3 output.mp3 --enthusiasm 1.0
  # Autotune without enthusiasm enhancement
  python autotune.py input.mp3 output.mp3 --no-enthusiasm
  # Use ElevenLabs API for professional voice enhancement
  python autotune.py input.mp3 output.mp3 --elevenlabs-key YOUR_API_KEY --polish-intensity 0.8
  # Maximum enhancement with all features
  python autotune.py input.mp3 output.mp3 --strength 0.5 --polish-intensity 0.9 --clarity-intensity 0.9 --enthusiasm 0.3
        """
    )
    parser.add_argument('input', help='Input MP3 file path')
    parser.add_argument('output', help='Output file path (WAV format recommended)')
    parser.add_argument('--strength', type=float, default=0.4,
                       help='Autotune strength (0.0-1.0, default: 0.4 for subtle, natural correction)')
    parser.add_argument('--scale', choices=['chromatic', 'major', 'minor'],
                       default='chromatic', help='Musical scale to snap to (default: chromatic)')
    parser.add_argument('--root-note', type=float, default=440.0,
                       help='Root note frequency in Hz (default: 440.0)')
    parser.add_argument('--eq', action='store_true',
                       help='Enable 3-band EQ')
    parser.add_argument('--low-gain', type=float, default=0.0,
                       help='Low frequency gain in dB (default: 0.0)')
    parser.add_argument('--mid-gain', type=float, default=0.0,
                       help='Mid frequency gain in dB (default: 0.0)')
    parser.add_argument('--high-gain', type=float, default=0.0,
                       help='High frequency gain in dB (default: 0.0)')
    parser.add_argument('--compress', action='store_true',
                       help='Enable audio compression')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Compression threshold (0-1, default: 0.7)')
    parser.add_argument('--ratio', type=float, default=4.0,
                       help='Compression ratio (default: 4.0)')
    parser.add_argument('--no-auto-enhance', action='store_true',
                       help='Disable automatic clarity enhancement')
    parser.add_argument('--no-pyworld', action='store_true',
                       help='Disable PyWorld (use librosa instead) for pitch shifting')
    parser.add_argument('--pyworld-pitch', action='store_true',
                       help='Use PyWorld for pitch detection (requires pyworld)')
    parser.add_argument('--enthusiasm', type=float, default=0.15,
                       help='Enthusiasm/vocal enhancement intensity (0.0-1.0, default: 0.15). Makes performance sound more energetic and polished.')
    parser.add_argument('--no-enthusiasm', action='store_true',
                       help='Disable enthusiasm enhancement')
    parser.add_argument('--no-speechbrain', action='store_true',
                       help='Disable SpeechBrain AI polishing')
    parser.add_argument('--polish-intensity', type=float, default=0.7,
                       help='AI enhancement polishing intensity (0.0-1.0, default: 0.7). Higher values provide more enhancement.')
    parser.add_argument('--elevenlabs-key', type=str, default=None,
                       help='ElevenLabs API key for professional voice enhancement (optional). Can also be set via ELEVENLABS_API_KEY env var.')
    parser.add_argument('--openai-key', type=str, default=None,
                       help='OpenAI API key for vocal analysis (optional). Can also be set via OPENAI_API_KEY env var.')
    parser.add_argument('--clarity-intensity', type=float, default=0.8,
                       help='Clarity enhancement intensity (0.0-1.0, default: 0.8). Controls vocal clarity improvements.')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    process_audio(
        args.input,
        args.output,
        autotune_strength=args.strength,
        scale=args.scale,
        root_note=args.root_note,
        apply_eq_enabled=args.eq,
        low_gain=args.low_gain,
        mid_gain=args.mid_gain,
        high_gain=args.high_gain,
        apply_compression_enabled=args.compress,
        threshold=args.threshold,
        ratio=args.ratio,
        auto_enhance=not args.no_auto_enhance,
        use_pyworld=PYWORLD_AVAILABLE and not args.no_pyworld,
        use_pyworld_pitch=args.pyworld_pitch and PYWORLD_AVAILABLE,
        enthusiasm_intensity=0.0 if args.no_enthusiasm else args.enthusiasm,
        speechbrain_polish=not args.no_speechbrain,
        polish_intensity=args.polish_intensity,
        elevenlabs_key=args.elevenlabs_key,
        openai_key=args.openai_key,
        clarity_intensity=args.clarity_intensity
    )
if __name__ == '__main__':
    main()
