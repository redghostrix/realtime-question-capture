# Configuration constants for the real-time question detector

# Whisper model configuration
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# LLM server configuration
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

# Audio capture configuration
SAMPLE_RATE = 16000  # Hz - Whisper expects 16kHz audio
CHUNK_DURATION = 5  # seconds - duration of audio chunks to process
CHANNELS = 1  # mono audio

# Logging configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
