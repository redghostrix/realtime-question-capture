"""Main pipeline orchestration for real-time question capture.

This module orchestrates the complete pipeline: audio capture → transcription →
question extraction → clipboard copy. It implements a continuous event loop with
graceful shutdown handling and comprehensive error recovery.
"""

import logging
import sys
import time
import signal
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from src.audio_capture import AudioCapture
from src.transcription import WhisperTranscriber
from src.question_extractor import QuestionExtractor
from src.clipboard_manager import ClipboardManager
from src.config import settings
from src.exceptions import (
    AudioCaptureError,
    TranscriptionError,
    QuestionDetectionError,
    ClipboardError
)

# Module-level logger
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT/Ctrl+C) gracefully.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global shutdown_requested
    shutdown_requested = True
    logger.info("Shutdown requested (Ctrl+C). Cleaning up...")


class RealtimeQuestionCapture:
    """Main pipeline orchestrator for real-time question capture.
    
    This class coordinates all components of the real-time question capture
    pipeline: audio capture, transcription, question extraction, and clipboard
    management. It implements a continuous event loop that polls for audio chunks,
    processes them through the pipeline, and copies detected questions to clipboard.
    
    Features:
    - Continuous audio capture and processing
    - Graceful shutdown on Ctrl+C
    - Automatic retry and error recovery
    - Comprehensive logging at all stages
    - Proper resource cleanup on exit
    
    Usage:
        pipeline = RealtimeQuestionCapture()
        pipeline.setup()
        pipeline.run()
        pipeline.cleanup()
    """
    
    def __init__(self):
        """Initialize the pipeline with component references set to None."""
        self.audio_capture: Optional[AudioCapture] = None
        self.transcriber: Optional[WhisperTranscriber] = None
        self.question_extractor: Optional[QuestionExtractor] = None
        self.clipboard_manager: Optional[ClipboardManager] = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        
        logger.debug("RealtimeQuestionCapture initialized")
    
    def check_cuda_version(self) -> None:
        """Check CUDA version compatibility.
        
        Validates that CUDA version meets requirements (>= 12.8 as per SPEC).
        Logs warnings if version is lower than required.
        """
        logger.debug("Checking CUDA version compatibility")
        
        if torch is None:
            logger.warning("PyTorch not available, cannot check CUDA version")
            return
        
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Running on CPU will be slower.")
            return
        
        try:
            cuda_version = torch.version.cuda
            if cuda_version is None:
                logger.warning("Could not determine CUDA version")
                return
            
            # Parse CUDA version (e.g., "12.8" -> 12.8)
            cuda_version_float = float('.'.join(cuda_version.split('.')[:2]))
            required_cuda = 12.8
            
            if cuda_version_float >= required_cuda:
                logger.info(f"CUDA version {cuda_version} meets requirements (>= {required_cuda})")
            else:
                logger.warning(
                    f"CUDA version {cuda_version} is lower than required {required_cuda}. "
                    f"Some features may not work optimally."
                )
                
        except Exception as e:
            logger.warning(f"Error checking CUDA version: {e}")
    
    def check_pytorch_version(self) -> None:
        """Check PyTorch version compatibility.
        
        Validates that PyTorch version meets requirements (>= 2.7.0).
        Logs warnings if version is lower than required.
        """
        logger.debug("Checking PyTorch version compatibility")
        
        if torch is None:
            logger.warning("PyTorch not available")
            return
        
        try:
            pytorch_version = torch.__version__
            # Parse version (e.g., "2.7.0+cu128" -> (2, 7, 0))
            version_parts = pytorch_version.split('+')[0].split('.')
            version_tuple = tuple(int(x) for x in version_parts[:3])
            required_version = (2, 7, 0)
            
            if version_tuple >= required_version:
                logger.info(f"PyTorch version {pytorch_version} meets requirements (>= 2.7.0)")
            else:
                logger.warning(
                    f"PyTorch version {pytorch_version} is lower than required 2.7.0. "
                    f"Some features may not work optimally."
                )
                
        except Exception as e:
            logger.warning(f"Error checking PyTorch version: {e}")
    
    def setup(self):
        """Set up and initialize all pipeline components.
        
        This method:
        - Configures logging
        - Logs startup banner with configuration
        - Initializes all components (audio, transcriber, extractor, clipboard)
        - Performs health checks (GPU availability, llama-server health)
        
        Raises:
            Exception: If any component fails to initialize
        """
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Log startup banner
        logger.info("=" * 80)
        logger.info("Real-Time Question Capture Pipeline")
        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  Whisper Model: {settings.whisper_model}")
        logger.info(f"  LLM Server URL: {settings.llama_server_url}")
        logger.info(f"  Sample Rate: {settings.sample_rate} Hz")
        logger.info(f"  Chunk Duration: {settings.chunk_duration} seconds")
        logger.info(f"  Channels: {settings.channels}")
        logger.info(f"  Log Level: {settings.log_level}")
        logger.info(f"  Question Extractor Max Retries: {settings.question_extractor_max_retries}")
        logger.info(f"  Question Extractor Timeout: {settings.question_extractor_timeout}s")
        logger.info("=" * 80)
        
        try:
            # Perform CUDA and PyTorch version checks
            logger.info("Performing CUDA and PyTorch compatibility checks...")
            self.check_cuda_version()
            self.check_pytorch_version()
            logger.info("Version compatibility checks completed")
            
            # Initialize AudioCapture
            logger.info("Initializing AudioCapture...")
            self.audio_capture = AudioCapture()
            logger.info("AudioCapture initialized successfully")
            
            # Initialize WhisperTranscriber
            logger.info("Initializing WhisperTranscriber...")
            self.transcriber = WhisperTranscriber(model_size=settings.whisper_model)
            logger.info("WhisperTranscriber initialized successfully")
            
            # Initialize QuestionExtractor
            logger.info("Initializing QuestionExtractor...")
            self.question_extractor = QuestionExtractor(
                llama_server_url=settings.llama_server_url,
                model_name=settings.question_extractor_model_name,
                max_retries=settings.question_extractor_max_retries,
                timeout=settings.question_extractor_timeout
            )
            logger.info("QuestionExtractor initialized successfully")
            
            # Initialize ClipboardManager
            logger.info("Initializing ClipboardManager...")
            self.clipboard_manager = ClipboardManager()
            logger.info("ClipboardManager initialized successfully")
            
            # Perform startup checks
            logger.info("Performing startup checks...")
            
            # Check GPU availability
            device_info = self.transcriber.get_device_info()
            if device_info["gpu_available"]:
                logger.info(
                    f"GPU available: {device_info['gpu_memory_free']:.2f}GB free / "
                    f"{device_info['gpu_memory_total']:.2f}GB total"
                )
            else:
                logger.warning("GPU not available, transcription will use CPU")
            
            # Check llama-server health
            server_healthy = self.question_extractor.check_server_health()
            if server_healthy:
                logger.info("LLM server health check: PASSED")
            else:
                logger.warning(
                    "LLM server health check: FAILED. "
                    f"Server at {settings.llama_server_url} may not be running. "
                    "Question extraction may fail."
                )
            
            logger.info("Setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            # Clean up any partially initialized components
            self.cleanup()
            raise
    
    def run(self):
        """Run the main event loop.
        
        This method implements the continuous processing loop:
        1. Start audio capture
        2. Poll for audio chunks
        3. Transcribe audio to text
        4. Extract questions from text
        5. Copy questions to clipboard
        
        The loop continues until shutdown_requested is set to True (Ctrl+C).
        Exceptions within the loop are logged but don't stop the pipeline.
        """
        global shutdown_requested
        
        logger.info("Starting main event loop...")
        
        try:
            # Start audio capture
            logger.info("Starting audio capture...")
            self.audio_capture.start()
            logger.info("Audio capture started, entering processing loop")
            
            # Main event loop
            while not shutdown_requested:
                try:
                    # Poll for audio chunk
                    audio_chunk = self.audio_capture.get_audio_chunk()
                    
                    if audio_chunk is None:
                        # No chunk available yet, sleep briefly to avoid busy-waiting
                        time.sleep(0.1)
                        continue
                    
                    # Log audio chunk received
                    chunk_duration = len(audio_chunk) / settings.sample_rate
                    logger.info(
                        f"Audio chunk received: {len(audio_chunk)} samples "
                        f"({chunk_duration:.2f}s)"
                    )
                    
                    # Transcribe audio
                    logger.debug("Starting transcription...")
                    transcribe_start = time.time()
                    transcription = self.transcriber.transcribe(audio_chunk)
                    transcribe_time = time.time() - transcribe_start
                    
                    if transcription is None or not transcription.get("text"):
                        logger.info("No transcription produced (silence or error)")
                        continue
                    
                    transcribed_text = transcription["text"]
                    logger.info(
                        f"Transcription completed: {len(transcribed_text)} characters "
                        f"in {transcribe_time:.2f}s - '{transcribed_text[:100]}...'"
                    )
                    
                    # Extract questions
                    logger.debug("Starting question extraction...")
                    extract_start = time.time()
                    questions = self.question_extractor.extract_questions(transcribed_text)
                    extract_time = time.time() - extract_start
                    
                    if questions is None:
                        logger.warning("Question extraction failed")
                        continue
                    
                    if not questions or not questions.strip():
                        logger.info("No questions detected in transcription")
                        continue
                    
                    logger.info(
                        f"Question extraction completed: {len(questions)} characters "
                        f"in {extract_time:.2f}s"
                    )
                    
                    # Copy to clipboard
                    logger.debug("Copying questions to clipboard...")
                    copy_success = self.clipboard_manager.copy_to_clipboard(questions)
                    
                    if copy_success:
                        logger.info("Questions successfully copied to clipboard")
                    else:
                        logger.warning("Failed to copy questions to clipboard")
                    
                except KeyboardInterrupt:
                    # Let signal handler manage this
                    break
                    
                except Exception as e:
                    # Log error but continue processing
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    logger.info("Continuing with next chunk...")
                    continue
            
            logger.info("Event loop terminated")
            
        except Exception as e:
            logger.error(f"Fatal error in event loop: {e}", exc_info=True)
            raise
    
    def cleanup(self):
        """Clean up all resources in reverse initialization order.
        
        This method:
        - Stops audio capture
        - Closes all components in reverse order
        - Ensures proper resource cleanup even on partial initialization
        - Logs each cleanup step
        """
        logger.info("Starting cleanup...")
        
        try:
            # Stop audio capture first
            if self.audio_capture is not None:
                try:
                    logger.info("Stopping audio capture...")
                    self.audio_capture.stop()
                    logger.info("Audio capture stopped")
                except Exception as e:
                    logger.error(f"Error stopping audio capture: {e}")
            
            # Close components in reverse initialization order
            if self.clipboard_manager is not None:
                try:
                    logger.info("Closing ClipboardManager...")
                    self.clipboard_manager.close()
                    logger.info("ClipboardManager closed")
                except Exception as e:
                    logger.error(f"Error closing ClipboardManager: {e}")
            
            if self.question_extractor is not None:
                try:
                    logger.info("Closing QuestionExtractor...")
                    self.question_extractor.close()
                    logger.info("QuestionExtractor closed")
                except Exception as e:
                    logger.error(f"Error closing QuestionExtractor: {e}")
            
            if self.transcriber is not None:
                try:
                    logger.info("Closing WhisperTranscriber...")
                    self.transcriber.close()
                    logger.info("WhisperTranscriber closed")
                except Exception as e:
                    logger.error(f"Error closing WhisperTranscriber: {e}")
            
            if self.audio_capture is not None:
                try:
                    logger.info("Closing AudioCapture...")
                    self.audio_capture.close()
                    logger.info("AudioCapture closed")
                except Exception as e:
                    logger.error(f"Error closing AudioCapture: {e}")
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)


def main() -> int:
    """Main entry point for the real-time question capture pipeline.
    
    This function:
    - Registers signal handler for graceful shutdown
    - Creates and runs the pipeline
    - Ensures cleanup is always called
    - Returns appropriate exit code
    
    Returns:
        0 for success, 1 for error
    """
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    pipeline = None
    
    try:
        # Create pipeline instance
        pipeline = RealtimeQuestionCapture()
        
        # Setup components
        pipeline.setup()
        
        # Run main loop
        pipeline.run()
        
        # Normal exit
        return 0
        
    except KeyboardInterrupt:
        # Ctrl+C handled by signal handler
        logger.info("Interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        return 1
        
    finally:
        # Always cleanup
        if pipeline is not None:
            pipeline.cleanup()
        
        logger.info("Application terminated")


if __name__ == '__main__':
    sys.exit(main())
