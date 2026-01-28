"""Custom exception classes for the real-time transcription capture pipeline.

This module defines custom exceptions for different components to enable
targeted error handling and improved debugging.
"""


class TranscriptionCaptureError(Exception):
    """Base exception class for all transcription capture pipeline errors."""
    pass


class AudioCaptureError(TranscriptionCaptureError):
    """Exception raised for errors in the audio capture component.
    
    Examples:
    - Device not found or unavailable
    - Stream open/start failures
    - Device validation failures
    """
    pass


class TranscriptionError(TranscriptionCaptureError):
    """Exception raised for errors in the transcription component.
    
    Examples:
    - Whisper model loading failures
    - GPU/CUDA errors
    - Audio format conversion errors
    """
    pass
