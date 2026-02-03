from transformers import AutoProcessor, AutoModelForCTC
import torch
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from difflib import SequenceMatcher

# Target phrases
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

class NepaliLiveTranscriber:
    def __init__(self, model_name="anish-shilpakar/wav2vec2-nepali"):
        """Initialize with the specified Nepali Wav2Vec2 model"""
        print("Loading model and processor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor from HuggingFace
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Use safetensors to avoid PyTorch version requirement
        self.model = AutoModelForCTC.from_pretrained(
            model_name,
            use_safetensors=True
        ).to(self.device)
        self.model.eval()
        
        # Audio settings - optimized for complete sentence capture
        self.SAMPLE_RATE = 16000
        self.CHUNK_DURATION = 6  # 6 seconds - enough time for complete sentences
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        print("Model loaded successfully!")
        print(f"⏱️  Audio will be processed every {self.CHUNK_DURATION} seconds")
    
    def find_best_match(self, transcription):
        best_match = None
        best_score = 0
        for phrase in TARGET_PHRASES:
            score = SequenceMatcher(None, transcription.strip(), phrase).ratio()
            if score > best_score:
                best_score = score
                best_match = phrase
        return best_match, best_score
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio chunk"""
        inputs = self.processor(audio_data, sampling_rate=self.SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice audio stream"""
        if status:
            print(f"Audio status: {status}")
        # sounddevice provides float32 data, convert to int16 range
        audio_data = (indata[:, 0] * 32768.0).astype(np.int16)
        self.audio_queue.put(audio_data)
    
    def process_audio_stream(self):
        audio_buffer = np.array([], dtype=np.int16)
        chunk_count = 0
        
        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # Process when we have enough audio (6 seconds)
                if len(audio_buffer) >= self.CHUNK_SIZE:
                    chunk_count += 1
                    print(f"\n{'='*60}")
                    print(f"🎤 Processing audio chunk #{chunk_count}...")
                    
                    # Take exactly CHUNK_SIZE samples
                    audio_float = audio_buffer[:self.CHUNK_SIZE].astype(np.float32) / 32768.0
                    
                    # Clear the buffer completely - no overlap
                    audio_buffer = np.array([], dtype=np.int16)
                    
                    # Clear the queue to start fresh
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    # Transcribe
                    transcription = self.transcribe_audio(audio_float)
                    
                    if transcription.strip():
                        print(f"📝 Heard: '{transcription}'")
                        best_match, score = self.find_best_match(transcription)
                        
                        if score > 0.5:
                            print(f"✅ MATCHED: {best_match}")
                            print(f"📊 Confidence: {score:.2%}")
                        else:
                            print(f"❌ No clear match")
                            print(f"   Best guess: {best_match} ({score:.2%})")
                    else:
                        print("⚠️  No speech detected")
                    
                    # Countdown before next recording
                    print(f"\n⏳ Next recording in: ", end='', flush=True)
                    for i in range(3, 0, -1):
                        print(f"{i}... ", end='', flush=True)
                        time.sleep(1)
                    print("0! 🎙️ Recording now!\n")
                    print(f"{'='*60}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing audio: {e}")
    
    def start_recording(self):
        self.is_recording = True
        print("\n🎙️  Starting live transcription... Press Ctrl+C to stop\n")
        
        process_thread = threading.Thread(target=self.process_audio_stream)
        process_thread.start()
        
        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(self.SAMPLE_RATE * 0.1),
            ):
                while self.is_recording:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping transcription...")
        
        self.is_recording = False
        process_thread.join()
        print("✓ Transcription stopped")


def main():
    print("=" * 60)
    print("Nepali Live Audio Transcription")
    print("=" * 60)
    print("\nTarget phrases:")
    for i, phrase in enumerate(TARGET_PHRASES, 1):
        print(f"{i:2d}. {phrase}")
    print()
    
    transcriber = NepaliLiveTranscriber(model_name="anish-shilpakar/wav2vec2-nepali")
    transcriber.start_recording()


if __name__ == "__main__":
    main()
