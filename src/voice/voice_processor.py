"""
Voice input module for converting speech to text queries using Whisper
"""
import os
import tempfile
import time
import numpy as np
import whisper
import sounddevice as sd
import soundfile as sf

class VoiceInputProcessor:
    """
    Processes voice input and converts it to text using Whisper
    """
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the voice input processor
        
        Args:
            model_name: The name of the Whisper model to use
                        (tiny, base, small, medium, large)
        """
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        print("Whisper model loaded")
        
        # Default recording parameters
        self.sample_rate = 16000
        self.channels = 1
        
    def record_audio(self, duration: int = 5, show_countdown: bool = True) -> np.ndarray:
        """
        Record audio from the microphone
        
        Args:
            duration: The duration to record in seconds
            show_countdown: Whether to show a countdown
            
        Returns:
            The recorded audio as a numpy array
        """
        print(f"Recording for {duration} seconds...")
        
        if show_countdown:
            for i in range(3, 0, -1):
                print(f"Starting in {i}...")
                time.sleep(1)
        
        # Record audio
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        
        # Show progress
        for i in range(duration):
            print(f"Recording: {i+1}/{duration} seconds")
            time.sleep(1)
            
        # Wait for recording to finish
        sd.wait()
        print("Recording finished")
        
        return audio
    
    def save_audio(self, audio: np.ndarray, file_path: str):
        """
        Save audio to a file
        
        Args:
            audio: The audio to save
            file_path: The path to save the audio to
        """
        sf.write(file_path, audio, self.sample_rate)
        print(f"Audio saved to {file_path}")
    
    def transcribe_audio(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio: The audio to transcribe
            
        Returns:
            The transcribed text
        """
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            self.save_audio(audio, temp_path)
        
        # Transcribe audio
        print("Transcribing audio...")
        result = self.model.transcribe(temp_path)
        
        # Delete temporary file
        os.unlink(temp_path)
        
        # Return transcribed text
        transcription = result["text"].strip()
        print(f"Transcription: {transcription}")
        
        return transcription
    
    def get_voice_query(self, duration: int = 5) -> str:
        """
        Record audio and transcribe it to text
        
        Args:
            duration: The duration to record in seconds
            
        Returns:
            The transcribed text
        """
        # Record audio
        audio = self.record_audio(duration)
        
        # Transcribe audio
        return self.transcribe_audio(audio)