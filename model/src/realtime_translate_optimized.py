import queue
import sys
import time
import numpy as np
import sounddevice as sd
import torch
from difflib import SequenceMatcher
import re

# Try to use faster-whisper if available, fallback to regular whisper
try:
    from faster_whisper import WhisperModel
    USE_FASTER_WHISPER = True
    print("✓ Using faster-whisper for improved performance")
except ImportError:
    import whisper
    USE_FASTER_WHISPER = False
    print("⚠️  Using standard whisper (install faster-whisper for 4-8x speedup)")

# ===== OPTIMIZED CONFIG =====
SAMPLE_RATE = 16000
CHUNK_DURATION = 4.0  # Increased to 4 seconds for complete phrases
SILENCE_THRESHOLD = 0.005  # Lowered from 0.02 for better detection
MIN_AUDIO_LENGTH = 1.5
ENERGY_THRESHOLD = 0.01  # Lowered from 0.03 for better detection
DEBUG_MODE = True  # Show audio levels for debugging

# Target phrases in Nepali
TARGET_PHRASES = [
    "नमस्ते",
    "आरामै", 
    "तपाईंको नाम के हो?",
    "तपाईं कुन ठाउँ बस्नु हुन्छ?",
    "खाना भो?",
    "कति बज्यो?",
    "आज कुन बार?",
    "तपाईं कता लाग्नु भयो?",
    "तपाईंको उमेर कति?",
    "फेरि भेटौँला",
]

# Extended variations including common transcription errors
PHRASE_VARIATIONS = {
    "नमस्ते": ["namaste", "नमस्ते", "namaskar", "namaste", "नमस्कार", "namasthe"],
    "आरामै": ["are you okay", "आरामै", "are you fine", "aramai", "aaramai", "आराम"],
    "तपाईंको नाम के हो?": [
        "what is your name", "तपाईंको नाम के हो", "नाम के हो",
        "tapai ko naam ke ho", "tapai naam", "के हो", "तपाई को नाम"
    ],
    "तपाईं कुन ठाउँ बस्नु हुन्छ?": [
        "where do you live", "कुन ठाउँ बस्नु हुन्छ", "कहाँ बस्नु हुन्छ",
        "kun thaun", "kaha basnu", "ठाउँ", "बस्नु हुन्छ"
    ],
    "खाना भो?": ["did you eat", "खाना भो", "have you eaten", "khana bho", "खाना", "भो"],
    "कति बज्यो?": ["what time is it", "कति बज्यो", "what is the time", "kati bajyo", "बज्यो"],
    "आज कुन बार?": ["what day is today", "आज कुन बार", "which day today", "aaja kun baar", "कुन बार"],
    "तपाईं कता लाग्नु भयो?": [
        "where are you going", "कता लाग्नु भयो", "where did you go",
        "kata lagnu", "लाग्नु भयो", "कता"
    ],
    "तपाईंको उमेर कति?": [
        "how old are you", "उमेर कति", "what is your age",
        "umer kati", "तपाईको उमेर", "कति"
    ],
    "फेरि भेटौँला": [
        "see you again", "फेरि भेटौँला", "goodbye", "bye",
        "pheri bhetaula", "भेटौला", "फेरि"
    ],
}

SIMILARITY_THRESHOLD = 0.55

# ==================

def normalize_text(text):
    """Normalize text for better matching"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[।,?.!]', '', text)
    return text.lower()

def similarity(a, b):
    """Calculate similarity between two strings"""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def contains_keywords(transcribed_text, phrase):
    """Check if transcribed text contains key words from phrase"""
    transcribed_words = set(normalize_text(transcribed_text).split())
    phrase_words = set(normalize_text(phrase).split())
    
    all_variations = [phrase]
    if phrase in PHRASE_VARIATIONS:
        all_variations.extend(PHRASE_VARIATIONS[phrase])
    
    for variation in all_variations:
        var_words = set(normalize_text(variation).split())
        if var_words and len(transcribed_words & var_words) / len(var_words) >= 0.5:
            return True
    return False

def find_best_match(transcribed_text):
    """Find the best matching target phrase with multiple strategies"""
    best_match = None
    best_score = 0
    
    transcribed_text = normalize_text(transcribed_text)
    
    for phrase in TARGET_PHRASES:
        scores = []
        
        # Strategy 1: Direct similarity
        score = similarity(transcribed_text, phrase)
        scores.append(score)
        
        # Strategy 2: Check against all variations
        if phrase in PHRASE_VARIATIONS:
            for variation in PHRASE_VARIATIONS[phrase]:
                var_score = similarity(transcribed_text, variation)
                scores.append(var_score)
        
        # Strategy 3: Partial matching
        if contains_keywords(transcribed_text, phrase):
            scores.append(0.65)
        
        # Strategy 4: Substring matching
        if normalize_text(phrase) in transcribed_text or transcribed_text in normalize_text(phrase):
            scores.append(0.7)
        
        max_score = max(scores)
        
        if max_score > best_score:
            best_score = max_score
            best_match = phrase
    
    if best_score >= SIMILARITY_THRESHOLD:
        return best_match, best_score
    return None, 0

def preprocess_audio(audio_chunk):
    """Optimized audio preprocessing"""
    # Remove DC offset
    audio_chunk = audio_chunk - np.mean(audio_chunk)
    
    # Normalize
    max_val = np.max(np.abs(audio_chunk))
    if max_val > 0:
        audio_chunk = audio_chunk / max_val
    
    # Simple noise gate
    noise_floor = np.percentile(np.abs(audio_chunk), 10)
    audio_chunk[np.abs(audio_chunk) < noise_floor * 1.5] *= 0.3
    
    return audio_chunk

def has_speech(audio_chunk):
    """Speech detection"""
    energy = np.sqrt(np.mean(audio_chunk ** 2))
    variation = np.std(audio_chunk)
    return energy > ENERGY_THRESHOLD and variation > 0.01

def is_silent(audio_chunk):
    """Check if audio chunk is silent"""
    return np.abs(audio_chunk).mean() < SILENCE_THRESHOLD

# ===== MODEL LOADING =====
print("Loading Whisper model on CUDA...")

if USE_FASTER_WHISPER:
    # faster-whisper: 4-8x faster than standard whisper
    model = WhisperModel(
        "small",  # Using small model for speed (was medium)
        device="cuda",
        compute_type="float16",
        num_workers=4,  # Parallel processing
        cpu_threads=4
    )
    print("✓ Faster-Whisper 'small' model loaded (optimized for speed)")
else:
    # Standard whisper fallback
    model = whisper.load_model("small").to("cuda")  # Changed from medium to small
    print("✓ Standard Whisper 'small' model loaded")

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

print("\n🎙️  Listening for specific Nepali phrases...")
print("📝 Tips for better recognition:")
print("   - Speak clearly and at moderate pace")
print("   - Position yourself 20-40cm from laptop")
print("   - Minimize background noise")
print("   - Speak louder than usual for laptop mic")
print("\n🎯 Target phrases:")
for i, phrase in enumerate(TARGET_PHRASES, 1):
    print(f"  {i}. {phrase}")
print("\n" + "="*50)
print("Listening... (Ctrl+C to stop)\n")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=int(SAMPLE_RATE * 0.1),
):
    audio_buffer = np.zeros((0, 1), dtype=np.float32)
    consecutive_silence = 0

    while True:
        audio_buffer = np.concatenate(
            (audio_buffer, audio_queue.get())
        )

        if len(audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION:
            audio_chunk = audio_buffer[: int(SAMPLE_RATE * CHUNK_DURATION)]
            audio_buffer = audio_buffer[int(SAMPLE_RATE * CHUNK_DURATION) :]

            audio_chunk = audio_chunk.flatten()
            
            # Calculate audio metrics for debugging
            audio_mean = np.abs(audio_chunk).mean()
            audio_energy = np.sqrt(np.mean(audio_chunk ** 2))
            audio_variation = np.std(audio_chunk)
            
            if DEBUG_MODE:
                print(f"📊 Audio levels - Mean: {audio_mean:.4f}, Energy: {audio_energy:.4f}, Variation: {audio_variation:.4f}", end='\r')
            
            # Skip if silent
            if is_silent(audio_chunk):
                consecutive_silence += 1
                if consecutive_silence == 1 and not DEBUG_MODE:
                    print("🔇 [Silent - waiting for speech...]", end='\r')
                continue
            
            # Skip if no speech detected
            if not has_speech(audio_chunk):
                consecutive_silence += 1
                if DEBUG_MODE and consecutive_silence % 5 == 0:
                    print(f"\n⚠️  No speech detected (Energy: {audio_energy:.4f} < {ENERGY_THRESHOLD}, Variation: {audio_variation:.4f})")
                continue
            
            consecutive_silence = 0
            print("\n🎤 [Processing speech...]" + " "*30)

            # Preprocess audio
            audio_chunk = preprocess_audio(audio_chunk)

            # OPTIMIZED Whisper transcription
            if USE_FASTER_WHISPER:
                # faster-whisper API
                segments, info = model.transcribe(
                    audio_chunk,
                    language="ne",
                    task="transcribe",
                    beam_size=3,  # Reduced from 5 for speed
                    vad_filter=True,  # Voice activity detection
                    vad_parameters=dict(
                        threshold=0.5,
                        min_speech_duration_ms=250,
                        min_silence_duration_ms=100
                    ),
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                )
                text = " ".join([segment.text for segment in segments]).strip()
            else:
                # Standard whisper API
                result = model.transcribe(
                    audio_chunk,
                    language="ne",
                    task="transcribe",
                    fp16=torch.cuda.is_available(),
                    beam_size=3,  # Reduced from 5 for speed
                    best_of=3,    # Reduced from 5 for speed
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                )
                text = result["text"].strip()

            if text:
                # Try to match with target phrases
                matched_phrase, confidence = find_best_match(text)
                
                if matched_phrase:
                    print(f"✅ RECOGNIZED: {matched_phrase}")
                    print(f"   📝 Heard: '{text}'")
                    print(f"   📊 Confidence: {confidence:.2%}")
                    
                    # Countdown before continuing
                    print("\n⏳ Resuming in: ", end='', flush=True)
                    for i in range(2, 0, -1):
                        print(f"{i}... ", end='', flush=True)
                        time.sleep(1)
                    print("0! 🎙️ Listening again...\n")
                    
                    # Clear audio buffer to avoid processing old audio
                    audio_buffer = np.zeros((0, 1), dtype=np.float32)
                    # Clear the queue
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break
                else:
                    print(f"❌ NOT MATCHED: '{text}'")
                    print(f"   (Try speaking more clearly)")
                print()
            else:
                print("⚠️  No speech detected in audio chunk\n")

