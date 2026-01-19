"""Custom exception classes for the real-time question capture pipeline.

This module defines custom exceptions for different components to enable
targeted error handling and improved debugging.
"""


class QuestionCaptureError(Exception):
    """Base exception class for all question capture pipeline errors."""
    pass


class AudioCaptureError(QuestionCaptureError):
    """Exception raised for errors in the audio capture component.
    
    Examples:
    - Device not found or unavailable
    - Stream open/start failures
    - Device validation failures
    """
    pass


class TranscriptionError(QuestionCaptureError):
    """Exception raised for errors in the transcription component.
    
    Examples:
    - Whisper model loading failures
    - GPU/CUDA errors
    - Audio format conversion errors
    """
    pass


class QuestionDetectionError(QuestionCaptureError):
    """Exception raised for errors in the question detection/extraction component.
    
    Examples:
    - LLM server connection failures
    - DSPy initialization errors
    - Question extraction failures
    """
    pass


class ClipboardError(QuestionCaptureError):
    """Exception raised for errors in the clipboard management component.
    
    Examples:
    - Clipboard access failures
    - Copy operation failures
    """
    pass
