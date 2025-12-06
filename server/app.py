import os
import uuid
import json
import subprocess
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import soundfile as sf

from autotune import (
    apply_eq,
    apply_compression,
    enhance_clarity,
    remove_artifacts,
    detect_pitch,
    snap_to_scale,
    smooth_pitch_curve,
    pitch_shift_audio,
    PYWORLD_AVAILABLE,
    NOISEREDUCE_AVAILABLE
)
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from beat_generator import BeatGenerator

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
PROCESSED_FOLDER = os.path.join(os.path.dirname(__file__), 'processed')
METADATA_FILE = os.path.join(os.path.dirname(__file__), 'recordings.json')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

processing_jobs = {}

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_processing_params(music_type, voice_level, beats_level, noise_level):
    voice_level = int(voice_level)
    beats_level = int(beats_level)
    noise_level = int(noise_level)
    
    base_params = {
        'trap': {
            'autotune_strength': 0.7,
            'scale': 'minor',
            'low_gain': 4,
            'mid_gain': -1,
            'high_gain': 2,
            'compression_threshold': 0.5,
            'compression_ratio': 6.0,
            'enthusiasm_intensity': 0.4,
            'clarity_intensity': 0.7
        },
        'rap': {  # Alias for trap
            'autotune_strength': 0.7,
            'scale': 'minor',
            'low_gain': 4,
            'mid_gain': -1,
            'high_gain': 2,
            'compression_threshold': 0.5,
            'compression_ratio': 6.0,
            'enthusiasm_intensity': 0.4,
            'clarity_intensity': 0.7
        },
        'pop': {
            'autotune_strength': 0.5,
            'scale': 'major',
            'low_gain': 1,
            'mid_gain': 2,
            'high_gain': 3,
            'compression_threshold': 0.6,
            'compression_ratio': 4.0,
            'enthusiasm_intensity': 0.5,
            'clarity_intensity': 0.9
        },
        'lofi': {
            'autotune_strength': 0.25,
            'scale': 'chromatic',
            'low_gain': 2,
            'mid_gain': 0,
            'high_gain': -2,
            'compression_threshold': 0.75,
            'compression_ratio': 2.5,
            'enthusiasm_intensity': 0.15,
            'clarity_intensity': 0.5
        },
        'chill': {  # Alias for lofi
            'autotune_strength': 0.25,
            'scale': 'chromatic',
            'low_gain': 2,
            'mid_gain': 0,
            'high_gain': -2,
            'compression_threshold': 0.75,
            'compression_ratio': 2.5,
            'enthusiasm_intensity': 0.15,
            'clarity_intensity': 0.5
        }
    }
    
    params = base_params.get(music_type, base_params['lofi']).copy()
    
    voice_multiplier = voice_level / 3.0
    params['autotune_strength'] = min(1.0, params['autotune_strength'] * voice_multiplier)
    params['clarity_intensity'] = min(1.0, params['clarity_intensity'] * voice_multiplier)
    
    beats_multiplier = beats_level / 3.0
    params['low_gain'] = int(params['low_gain'] * beats_multiplier)
    params['compression_ratio'] = max(2.0, params['compression_ratio'] * beats_multiplier)
    
    noise_reduction = (6 - noise_level) / 5.0
    params['noise_reduction'] = 0.3 + (noise_reduction * 0.4)
    
    # Ensure beats_level is int
    params['beats_level'] = beats_level
    
    return params

def convert_to_wav(input_path):
    wav_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
    
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(44100)
        audio.export(wav_path, format='wav')
        if os.path.exists(wav_path):
            return wav_path
    except Exception as e:
        print(f"Pydub conversion failed: {e}, trying ffmpeg...")
    
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '1',
            wav_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        if result.returncode == 0 and os.path.exists(wav_path):
            return wav_path
        else:
            print(f"FFmpeg error: {result.stderr.decode('utf-8', errors='ignore')}")
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
    
    return None

def update_job_status(job_id, step, progress, message):
    if job_id in processing_jobs:
        processing_jobs[job_id]['step'] = step
        processing_jobs[job_id]['progress'] = progress
        processing_jobs[job_id]['message'] = message

def enhance_enthusiasm_fast(y, sr, intensity=0.5):
    if intensity <= 0:
        return y
    
    effective_intensity = intensity * 0.6
    original_max = np.max(np.abs(y))
    
    if effective_intensity > 0.1:
        from scipy.signal import butter, filtfilt
        
        def boost_band(y, sr, low, high, gain):
            nyq = sr / 2
            low_norm = max(low / nyq, 0.001)
            high_norm = min(high / nyq, 0.999)
            if low_norm >= high_norm:
                return y
            b, a = butter(2, [low_norm, high_norm], btype='band')
            band = filtfilt(b, a, y)
            return y + band * gain
        
        y = boost_band(y, sr, 2000, 5000, 0.08 * effective_intensity)
        y = boost_band(y, sr, 5000, 10000, 0.05 * effective_intensity)
        y = boost_band(y, sr, 1000, 2000, 0.04 * effective_intensity)
    
    if effective_intensity > 0.2:
        saturation_amount = effective_intensity * 0.08
        y_saturated = np.tanh(y * (1 + saturation_amount)) / (1 + saturation_amount)
        y = y * (1 - saturation_amount * 0.3) + y_saturated * (saturation_amount * 0.3)
    
    if original_max > 0:
        y_max = np.max(np.abs(y))
        if y_max > 0:
            y = y / y_max * original_max * 0.98
    
    return y

def fast_noise_reduce(y, sr, intensity=0.5):
    if NOISEREDUCE_AVAILABLE and intensity > 0:
        try:
            import noisereduce as nr
            y_reduced = nr.reduce_noise(
                y=y,
                sr=sr,
                stationary=True,
                prop_decrease=intensity * 0.7
            )
            return y * (1 - intensity * 0.6) + y_reduced * (intensity * 0.6)
        except:
            pass
    return y

def fast_autotune(y, sr, strength=0.4, scale='chromatic', root_note=440, update_fn=None):
    hop_length = 512
    frame_length = 2048
    
    if update_fn:
        update_fn("Detecting pitch contours...")
    
    pitches = detect_pitch(y, sr, frame_length=frame_length, hop_length=hop_length, use_pyworld=PYWORLD_AVAILABLE)
    frame_times = np.arange(len(pitches)) * hop_length / sr
    
    valid_mask = pitches > 0
    if not np.any(valid_mask):
        return y
    
    if update_fn:
        update_fn("Smoothing pitch curve...")
    
    smoothed_pitches = smooth_pitch_curve(pitches, alpha=0.6, preserve_variation=True)
    valid_mask_smooth = smoothed_pitches > 0
    if not np.any(valid_mask_smooth):
        return y
    
    if update_fn:
        update_fn("Calculating corrections...")
    
    valid_smoothed = smoothed_pitches[valid_mask_smooth]
    valid_times = frame_times[valid_mask_smooth]
    
    corrected_data = [snap_to_scale(p, scale, root_note, 0.2) for p in valid_smoothed]
    if len(corrected_data) == 0:
        return y
    
    corrected_pitches = np.array([d[0] for d in corrected_data])
    correction_amounts = np.array([d[1] for d in corrected_data])
    
    pitch_ratios = corrected_pitches / valid_smoothed
    adaptive_strength = strength * correction_amounts
    adaptive_strength_smooth = gaussian_filter1d(adaptive_strength, sigma=2.0)
    pitch_ratios_smooth = gaussian_filter1d(pitch_ratios, sigma=1.5)
    
    pitch_ratio_func = interp1d(valid_times, pitch_ratios_smooth, kind='linear', fill_value='extrapolate', bounds_error=False)
    strength_func = interp1d(valid_times, adaptive_strength_smooth, kind='linear', fill_value=0.0, bounds_error=False)
    
    if update_fn:
        update_fn("Applying pitch correction...")
    
    segment_length = int(sr * 2.0)
    overlap = int(sr * 0.3)
    output = np.zeros_like(y)
    window = np.hanning(overlap * 2)
    
    total_segments = (len(y) + segment_length - overlap - 1) // (segment_length - overlap)
    
    for seg_idx, start_idx in enumerate(range(0, len(y), segment_length - overlap)):
        end_idx = min(start_idx + segment_length, len(y))
        segment = y[start_idx:end_idx]
        segment_times = np.linspace(start_idx/sr, end_idx/sr, len(segment))
        segment_ratios = pitch_ratio_func(segment_times)
        segment_strengths = strength_func(segment_times)
        
        valid_ratios = segment_ratios[(segment_ratios > 0) & (segment_ratios != 1.0)]
        if len(valid_ratios) == 0:
            if start_idx > 0 and len(segment) >= overlap:
                output[start_idx:start_idx+overlap] += segment[:overlap] * window[:overlap]
                output[start_idx+overlap:end_idx] += segment[overlap:] if len(segment) > overlap else np.array([])
            else:
                output[start_idx:end_idx] += segment
            continue
        
        median_ratio = np.median(valid_ratios)
        valid_strengths = segment_strengths[segment_strengths > 0]
        median_strength = np.median(valid_strengths) if len(valid_strengths) > 0 else 0
        
        if median_strength < 0.12 or median_ratio <= 0 or median_ratio == 1.0:
            if start_idx > 0 and len(segment) >= overlap:
                output[start_idx:start_idx+overlap] += segment[:overlap] * window[:overlap]
                output[start_idx+overlap:end_idx] += segment[overlap:] if len(segment) > overlap else np.array([])
            else:
                output[start_idx:end_idx] += segment
            continue
        
        target_ratio = 1.0 + (median_ratio - 1.0) * median_strength * 0.7
        if abs(target_ratio - 1.0) > 0.012:
            corrected_segment = pitch_shift_audio(segment, sr=sr, pitch_ratio=target_ratio, use_pyworld=True)
            blend_factor = min(0.65, median_strength * 0.75)
            segment = segment * (1 - blend_factor) + corrected_segment * blend_factor
        
        if start_idx > 0 and len(segment) >= overlap:
            output[start_idx:start_idx+overlap] += segment[:overlap] * window[:overlap]
            output[start_idx+overlap:end_idx] += segment[overlap:] if len(segment) > overlap else np.array([])
        else:
            output[start_idx:end_idx] += segment
    
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * 0.95
    
    return output

def process_recording_with_progress(job_id, input_path, output_path, params, music_type, input_filename, output_filename):
    try:
        converted_path = None
        file_ext = input_path.rsplit('.', 1)[-1].lower()
        
        update_job_status(job_id, 1, 5, "Converting audio format...")
        
        if file_ext in ['webm', 'ogg', 'opus']:
            converted_path = convert_to_wav(input_path)
            if converted_path:
                input_path = converted_path
        
        update_job_status(job_id, 1, 10, "Audio converted successfully")
        
        try:
            y, sr = librosa.load(input_path, sr=None, mono=True)
        except Exception as e:
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)
            raise e
        
        update_job_status(job_id, 2, 15, "Audio loaded into memory")
        
        noise_intensity = params.get('noise_reduction', 0.5)
        if noise_intensity > 0:
            update_job_status(job_id, 3, 18, "Analyzing background noise...")
            y = fast_noise_reduce(y, sr, intensity=noise_intensity)
            update_job_status(job_id, 3, 25, "Background noise reduced")
        
        update_job_status(job_id, 4, 28, "Analyzing vocal frequencies...")
        y = enhance_clarity(y, sr, preserve_details=True, intensity=params.get('clarity_intensity', 0.8))
        update_job_status(job_id, 4, 35, "Vocal clarity enhanced")
        
        if params.get('autotune_strength', 0) > 0:
            def autotune_progress(msg):
                progress_map = {
                    "Detecting pitch contours...": (5, 40),
                    "Smoothing pitch curve...": (6, 50),
                    "Calculating corrections...": (7, 55),
                    "Applying pitch correction...": (7, 60),
                }
                step, pct = progress_map.get(msg, (7, 55))
                update_job_status(job_id, step, pct, msg)
            
            update_job_status(job_id, 5, 38, "Starting pitch analysis...")
            y = fast_autotune(
                y, sr,
                strength=params['autotune_strength'],
                scale=params.get('scale', 'chromatic'),
                root_note=440,
                update_fn=autotune_progress
            )
            update_job_status(job_id, 8, 70, "Pitch correction complete")
        
        if params.get('enthusiasm_intensity', 0) > 0:
            update_job_status(job_id, 9, 72, "Boosting vocal presence...")
            y = enhance_enthusiasm_fast(y, sr, intensity=params['enthusiasm_intensity'])
            update_job_status(job_id, 9, 76, "Vocal energy enhanced")
        
        update_job_status(job_id, 10, 78, "Cleaning up audio glitches...")
        y = remove_artifacts(y, sr)
        update_job_status(job_id, 10, 82, "Audio artifacts removed")
        
        update_job_status(job_id, 11, 84, "Applying dynamic compression...")
        y = apply_compression(
            y,
            threshold=params.get('compression_threshold', 0.7),
            ratio=params.get('compression_ratio', 4.0),
            sr=sr
        )
        update_job_status(job_id, 11, 88, "Dynamics balanced")
        
        if any([params.get('low_gain', 0), params.get('mid_gain', 0), params.get('high_gain', 0)]):
            update_job_status(job_id, 12, 90, "Applying EQ adjustments...")
            y = apply_eq(
                y, sr,
                low_gain=params.get('low_gain', 0),
                mid_gain=params.get('mid_gain', 0),
                high_gain=params.get('high_gain', 0)
            )
            update_job_status(job_id, 12, 93, "Frequency balance optimized")
        
        update_job_status(job_id, 13, 95, "Normalizing audio levels...")
        y_max = np.max(np.abs(y))
        if y_max > 0:
            y = y * (0.95 / y_max)
        
        update_job_status(job_id, 14, 97, "Encoding final audio...")
        
        beats_level = int(params.get('beats_level', 0))
        print(f"[{job_id}] Beats level: {beats_level}, Music type: {music_type}")
        
        if beats_level > 0:
            update_job_status(job_id, 14, 98, "Generating beat...")
            
            # Save temp processed vocal for mixing
            temp_vocal_path = output_path.rsplit('.', 1)[0] + '_temp_vocal.wav'
            sf.write(temp_vocal_path, y, sr)
            print(f"[{job_id}] Saved temp vocal to {temp_vocal_path}")
            
            try:
                bg = BeatGenerator(
                    vocal_path=temp_vocal_path,
                    style=music_type,
                    intensity=beats_level
                )
                print(f"[{job_id}] Initialized BeatGenerator with style={music_type}")
                bg.mix_tracks(output_path)
                print(f"[{job_id}] Beat generation and mixing successful")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Beat generation failed: {e}")
                # Fallback to just saving vocals
                sf.write(output_path, y, sr)
            finally:
                if os.path.exists(temp_vocal_path):
                    os.remove(temp_vocal_path)
        else:
            sf.write(output_path, y, sr)
        
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)
        
        duration = len(y) / sr
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_str = f"{minutes}:{seconds:02d}"
        
        metadata = load_metadata()
        timestamp = datetime.now()
        new_record = {
            'id': job_id,
            'date': timestamp.strftime('%Y-%m-%d'),
            'time': timestamp.strftime('%H:%M'),
            'type': music_type.upper(),
            'duration': duration_str,
            'original_file': input_filename,
            'processed_file': output_filename,
            'settings': params
        }
        metadata.insert(0, new_record)
        save_metadata(metadata)
        
        update_job_status(job_id, 15, 100, "Ready to play!")
        processing_jobs[job_id]['status'] = 'complete'
        processing_jobs[job_id]['duration'] = duration_str
        print(f"[{job_id}] Processing complete! Duration: {duration_str}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['error'] = str(e)
        update_job_status(job_id, 0, 0, f"Error: {str(e)}")

@app.route('/', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    music_type = request.form.get('musicType', 'lofi')
    voice_level = request.form.get('voiceLevel', '3')
    beats_level = request.form.get('beatsLevel', '3')
    noise_level = request.form.get('noiseLevel', '3')
    
    job_id = str(uuid.uuid4())[:8]
    
    original_filename = secure_filename(audio_file.filename or 'recording.webm')
    input_filename = f"{job_id}_original_{original_filename}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    audio_file.save(input_path)
    
    output_filename = f"{job_id}_processed.wav"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    params = get_processing_params(music_type, voice_level, beats_level, noise_level)
    
    processing_jobs[job_id] = {
        'status': 'processing',
        'step': 0,
        'progress': 0,
        'message': 'Starting...',
        'output_path': output_path,
        'output_filename': output_filename
    }
    
    thread = threading.Thread(
        target=process_recording_with_progress,
        args=(job_id, input_path, output_path, params, music_type, input_filename, output_filename)
    )
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'processing'
    })

@app.route('/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    def generate():
        import time
        print(f"[{job_id}] SSE connection started")
        while True:
            if job_id not in processing_jobs:
                print(f"[{job_id}] Job not found in processing_jobs")
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            
            job = processing_jobs[job_id]
            data = {
                'status': job['status'],
                'step': job.get('step', 0),
                'progress': job.get('progress', 0),
                'message': job.get('message', ''),
            }
            
            if job['status'] == 'complete':
                data['duration'] = job.get('duration', '')
                print(f"[{job_id}] Sending complete status via SSE")
            elif job['status'] == 'error':
                data['error'] = job.get('error', 'Unknown error')
                print(f"[{job_id}] Sending error status via SSE")
            
            yield f"data: {json.dumps(data)}\n\n"
            
            if job['status'] in ['complete', 'error']:
                print(f"[{job_id}] SSE connection closing")
                time.sleep(1.0)
                break
            
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'X-Accel-Buffering': 'no'
    })

@app.route('/download/<job_id>', methods=['GET'])
def download_audio(job_id):
    print(f"[{job_id}] Download requested")
    if job_id in processing_jobs:
        job = processing_jobs[job_id]
        if job['status'] != 'complete':
            print(f"[{job_id}] Download rejected - status: {job['status']}")
            return jsonify({'error': 'Processing not complete'}), 400
        output_path = job['output_path']
    else:
        output_filename = f"{job_id}_processed.wav"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    print(f"[{job_id}] Serving file: {output_path}")
    if not os.path.exists(output_path):
        print(f"[{job_id}] File not found!")
        return jsonify({'error': 'Audio file not found'}), 404
    
    print(f"[{job_id}] Sending file...")
    return send_file(
        output_path,
        mimetype='audio/wav',
        as_attachment=True,
        download_name=f'processed_{job_id}.wav'
    )

@app.route('/recordings', methods=['GET'])
def get_recordings():
    metadata = load_metadata()
    return jsonify(metadata)

@app.route('/recordings/<recording_id>', methods=['GET'])
def get_recording(recording_id):
    metadata = load_metadata()
    record = next((r for r in metadata if r['id'] == recording_id), None)
    
    if not record:
        return jsonify({'error': 'Recording not found'}), 404
    
    return jsonify(record)

@app.route('/recordings/<recording_id>/audio', methods=['GET'])
def get_recording_audio(recording_id):
    metadata = load_metadata()
    record = next((r for r in metadata if r['id'] == recording_id), None)
    
    if not record:
        return jsonify({'error': 'Recording not found'}), 404
    
    audio_path = os.path.join(PROCESSED_FOLDER, record['processed_file'])
    
    if not os.path.exists(audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    
    return send_file(audio_path, mimetype='audio/wav')

@app.route('/recordings/<recording_id>/original', methods=['GET'])
def get_original_audio(recording_id):
    metadata = load_metadata()
    record = next((r for r in metadata if r['id'] == recording_id), None)
    
    if not record:
        return jsonify({'error': 'Recording not found'}), 404
    
    audio_path = os.path.join(UPLOAD_FOLDER, record['original_file'])
    
    if not os.path.exists(audio_path):
        return jsonify({'error': 'Original audio file not found'}), 404
    
    return send_file(audio_path, mimetype='audio/mpeg')

@app.route('/recordings/<recording_id>', methods=['DELETE'])
def delete_recording(recording_id):
    metadata = load_metadata()
    record = next((r for r in metadata if r['id'] == recording_id), None)
    
    if not record:
        return jsonify({'error': 'Recording not found'}), 404
    
    try:
        original_path = os.path.join(UPLOAD_FOLDER, record['original_file'])
        if os.path.exists(original_path):
            os.remove(original_path)
        
        processed_path = os.path.join(PROCESSED_FOLDER, record['processed_file'])
        if os.path.exists(processed_path):
            os.remove(processed_path)
        
        metadata = [r for r in metadata if r['id'] != recording_id]
        save_metadata(metadata)
        
        return jsonify({'success': True, 'message': 'Recording deleted'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'pyworld_available': PYWORLD_AVAILABLE,
        'noisereduce_available': NOISEREDUCE_AVAILABLE
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
