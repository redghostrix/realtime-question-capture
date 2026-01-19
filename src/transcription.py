"""
Whisper-based transcription module for real-time audio processing.

This module provides GPU-accelerated speech-to-text transcription using faster-whisper.
It integrates with the AudioCapture module to process 5-second audio chunks.
"""

import logging
import numpy as np
from typing import Optional, Dict, List, Any
from faster_whisper import WhisperModel

from .config import settings
from .exceptions import TranscriptionError

# Module-level logger
logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    GPU-accelerated Whisper transcriber for real-time audio processing.
    
    This class handles speech-to-text transcription of audio chunks from AudioCapture.
    It converts int16 audio to float32 format, uses faster-whisper for GPU inference,
    and returns transcriptions with timestamps.
    
    Features:
    - GPU acceleration with automatic CPU fallback
    - Voice activity detection to skip silence
    - Segment-level timestamps
    - Comprehensive error handling and logging
    
    Usage:
        with WhisperTranscriber() as transcriber:
            result = transcriber.transcribe(audio_chunk)
            if result and result["text"]:
                print(result["text"])
    """
    
    def __init__(
        self,
        model_size: Optional[str] = None,
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True
    ):
        """
        Initialize the Whisper transcriber with GPU support.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
                       Defaults to settings.whisper_model.
            device: Device to run inference on ("cuda" or "cpu").
            compute_type: Computation type ("float16" for GPU, "int8" for CPU).
            language: Force specific language (None for auto-detection).
            beam_size: Beam size for transcription (higher = more accurate but slower).
            vad_filter: Enable voice activity detection to skip silence.
        """
        self.model_size = model_size or settings.whisper_model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.model = None
        
        logger.info(
            f"Initializing WhisperTranscriber: model={self.model_size}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )
        
        try:
            # Initialize faster-whisper model with GPU configuration
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None  # Use default cache location
            )
            
            logger.info(
                f"Successfully loaded Whisper model '{self.model_size}' "
                f"on device '{self.device}' with compute type '{self.compute_type}'"
            )
            
        except RuntimeError as e:
            # CUDA/GPU errors
            logger.error(
                f"RuntimeError loading Whisper model '{self.model_size}' on {self.device}: {e}. "
                f"GPU may not be available or out of memory."
            )
            raise TranscriptionError(
                f"Failed to initialize Whisper model on {self.device}. "
                f"Check GPU availability and memory."
            ) from e
            
        except OSError as e:
            # Model download/file errors
            logger.error(
                f"OSError loading Whisper model '{self.model_size}': {e}. "
                f"Model may not exist or download failed."
            )
            raise TranscriptionError(
                f"Failed to load or download Whisper model '{self.model_size}'. "
                f"Check network connection and model name."
            ) from e
            
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                f"Unexpected error initializing Whisper model '{self.model_size}': {e}"
            )
            raise TranscriptionError(
                f"Unexpected error initializing Whisper model: {e}"
            ) from e
    
    def _convert_audio_format(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert int16 audio to float32 format expected by Whisper.
        
        Args:
            audio_data: Numpy array of int16 samples from AudioCapture.
            
        Returns:
            Numpy array of float32 samples normalized to [-1.0, 1.0] range.
            
        Raises:
            ValueError: If audio_data is empty or invalid.
        """
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Received empty audio data for conversion")
            raise TranscriptionError("Audio data is empty")
        
        logger.debug(
            f"Converting audio: shape={audio_data.shape}, dtype={audio_data.dtype}"
        )
        
        # Convert int16 to float32 and normalize to [-1.0, 1.0]
        # int16 range is -32768 to 32767
        audio_float32 = audio_data.astype(np.float32) / 32768.0
        
        # Ensure array is contiguous in memory for efficient processing
        if not audio_float32.flags['C_CONTIGUOUS']:
            audio_float32 = np.ascontiguousarray(audio_float32)
        
        logger.debug(
            f"Converted audio: shape={audio_float32.shape}, dtype={audio_float32.dtype}, "
            f"range=[{audio_float32.min():.4f}, {audio_float32.max():.4f}]"
        )
        
        return audio_float32
    
    def transcribe(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio chunk to text with timestamps.
        
        Args:
            audio_data: Numpy array of int16 samples from AudioCapture.
            
        Returns:
            Dictionary with transcription results:
            {
                "text": "full transcription text",
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "segment 1"},
                    {"start": 2.5, "end": 5.0, "text": "segment 2"}
                ]
            }
            Returns None if transcription fails.
        """
        # Input validation
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Received empty audio data for transcription")
            return {"text": "", "segments": []}
        
        if audio_data.dtype != np.int16:
            logger.warning(
                f"Expected int16 audio data, got {audio_data.dtype}. "
                f"Attempting to proceed anyway."
            )
        
        # Check minimum audio length (e.g., 0.1 seconds = 1600 samples at 16kHz)
        min_samples = int(0.1 * settings.sample_rate)
        if len(audio_data) < min_samples:
            logger.warning(
                f"Audio chunk too short: {len(audio_data)} samples "
                f"({len(audio_data) / settings.sample_rate:.3f}s). Minimum: {min_samples} samples."
            )
            return {"text": "", "segments": []}
        
        try:
            # Convert audio format
            audio_float32 = self._convert_audio_format(audio_data)
            
            logger.debug(
                f"Starting transcription: {len(audio_data)} samples "
                f"({len(audio_data) / settings.sample_rate:.2f}s)"
            )
            
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_float32,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                word_timestamps=False  # Segment-level timestamps sufficient
            )
            
            # Process segments
            result_segments = []
            full_text_parts = []
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                result_segments.append(segment_dict)
                full_text_parts.append(segment.text.strip())
                
                logger.debug(
                    f"Segment [{segment.start:.2f}s - {segment.end:.2f}s]: "
                    f"'{segment.text.strip()}'"
                )
            
            # Combine segments into full text
            full_text = " ".join(full_text_parts)
            
            result = {
                "text": full_text,
                "segments": result_segments
            }
            
            logger.info(
                f"Transcription successful: {len(full_text)} characters, "
                f"{len(result_segments)} segments"
            )
            
            if not result_segments:
                logger.debug("No speech detected in audio chunk (empty transcription)")
            
            return result
            
        except RuntimeError as e:
            # GPU out-of-memory or CUDA errors
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                logger.error(
                    f"GPU error during transcription: {e}. "
                    f"Audio: {len(audio_data)} samples ({len(audio_data) / settings.sample_rate:.2f}s). "
                    f"Suggestions: reduce chunk duration, use smaller model, or close GPU applications."
                )
                
                # Attempt CPU fallback if currently on GPU
                if self.device == "cuda":
                    logger.warning("Attempting CPU fallback for this chunk...")
                    try:
                        return self._transcribe_cpu_fallback(audio_data)
                    except Exception as fallback_error:
                        logger.error(f"CPU fallback also failed: {fallback_error}")
                        return None
            else:
                logger.error(f"RuntimeError during transcription: {e}")
                return None
                
        except ValueError as e:
            # Audio conversion errors
            logger.error(f"Audio format error: {e}")
            return None
            
        except Exception as e:
            # General transcription failures
            logger.error(
                f"Unexpected error during transcription: {e}. "
                f"Audio: {len(audio_data)} samples ({len(audio_data) / settings.sample_rate:.2f}s)"
            )
            return None
    
    def _transcribe_cpu_fallback(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Fallback transcription on CPU when GPU fails.
        
        Args:
            audio_data: Numpy array of int16 samples.
            
        Returns:
            Transcription result or None if fallback fails.
        """
        logger.warning("Initializing temporary CPU model for fallback...")
        
        try:
            # Create temporary CPU model
            cpu_model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=None
            )
            
            audio_float32 = self._convert_audio_format(audio_data)
            
            segments, info = cpu_model.transcribe(
                audio_float32,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                word_timestamps=False
            )
            
            result_segments = []
            full_text_parts = []
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                result_segments.append(segment_dict)
                full_text_parts.append(segment.text.strip())
            
            full_text = " ".join(full_text_parts)
            
            result = {
                "text": full_text,
                "segments": result_segments
            }
            
            logger.info(f"CPU fallback transcription successful: {len(full_text)} characters")
            
            # Clean up temporary model
            del cpu_model
            
            return result
            
        except Exception as e:
            logger.error(f"CPU fallback transcription failed: {e}")
            return None
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get GPU/device availability and memory information.
        
        Returns:
            Dictionary with device information:
            {
                "device": "cuda" or "cpu",
                "gpu_available": bool,
                "gpu_memory_total": float (GB) or None,
                "gpu_memory_free": float (GB) or None
            }
        """
        info = {
            "device": self.device,
            "gpu_available": False,
            "gpu_memory_total": None,
            "gpu_memory_free": None
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_available"] = True
                info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info["gpu_memory_free"] = (
                    torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_allocated(0)
                ) / (1024**3)
                logger.debug(
                    f"GPU info: {info['gpu_memory_free']:.2f}GB free / "
                    f"{info['gpu_memory_total']:.2f}GB total"
                )
        except ImportError:
            logger.warning("PyTorch not available, cannot get GPU info")
        except Exception as e:
            logger.warning(f"Error getting GPU info: {e}")
        
        return info
    
    def close(self):
        """
        Clean up resources and free GPU memory.
        """
        if self.model is not None:
            try:
                logger.info("Closing WhisperTranscriber and freeing resources...")
                del self.model
                self.model = None
                logger.info("WhisperTranscriber cleanup completed")
            except Exception as e:
                logger.error(f"Error during WhisperTranscriber cleanup: {e}")
        else:
            logger.debug("WhisperTranscriber already closed or not initialized")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False  # Propagate exceptions
