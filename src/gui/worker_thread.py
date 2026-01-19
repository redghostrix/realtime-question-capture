"""
Worker thread for running the pipeline in the background.

This module provides a QThread that runs the audio capture, transcription,
and question extraction pipeline without blocking the GUI.
"""

import logging
import time
from typing import Optional
from PyQt5.QtCore import QThread, pyqtSignal

from src.audio_capture import AudioCapture
from src.transcription import WhisperTranscriber
from src.question_extractor import QuestionExtractor
from src.clipboard_manager import ClipboardManager
from src.config import settings


logger = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """
    Worker thread that runs the question capture pipeline.
    
    This thread runs the complete pipeline: audio capture → transcription →
    question extraction → clipboard copy. It emits signals for GUI updates.
    
    Signals:
        transcription_ready: Emitted when transcription completes
                           Args: timestamp (str), text (str)
        question_extracted: Emitted when questions are extracted
                          Args: timestamp (str), question (str)
        timing_update: Emitted with timing metrics
                      Args: audio_time (float), trans_time (float), extract_time (float)
        status_changed: Emitted when pipeline status changes
                       Args: status (str) - "Running", "Stopped", "Paused", "Error"
        initialization_complete: Emitted when all components are initialized
                               Args: success (bool), message (str)
    """
    
    # Define signals
    transcription_ready = pyqtSignal(str, str)  # timestamp, text
    question_extracted = pyqtSignal(str, str)  # timestamp, question
    timing_update = pyqtSignal(float, float, float)  # audio_time, trans_time, extract_time
    status_changed = pyqtSignal(str)  # status message
    initialization_complete = pyqtSignal(bool, str)  # success, message
    
    def __init__(self):
        """Initialize the pipeline worker thread."""
        super().__init__()
        
        # Pipeline components
        self.audio_capture: Optional[AudioCapture] = None
        self.transcriber: Optional[WhisperTranscriber] = None
        self.question_extractor: Optional[QuestionExtractor] = None
        self.clipboard_manager: Optional[ClipboardManager] = None
        
        # Thread control flags
        self._running = False
        self._paused = False
        self._stop_requested = False
        
        logger.debug("PipelineWorker initialized")
    
    def run(self):
        """
        Main thread execution method.
        
        This method runs when start() is called on the thread.
        It initializes components and runs the main processing loop.
        """
        try:
            # Initialize all components
            self.status_changed.emit("Initializing...")
            
            if not self._initialize_components():
                self.initialization_complete.emit(False, "Failed to initialize components")
                self.status_changed.emit("Stopped")
                return
            
            self.initialization_complete.emit(True, "Initialization successful")
            
            # Start main processing loop
            self._running = True
            self.status_changed.emit("Running")
            self._process_loop()
            
        except Exception as e:
            logger.error(f"Fatal error in worker thread: {e}", exc_info=True)
            self.status_changed.emit(f"Error: {e}")
        finally:
            self._cleanup()
            self._running = False
            if not self._stop_requested:
                self.status_changed.emit("Stopped")
    
    def _initialize_components(self) -> bool:
        """
        Initialize all pipeline components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize AudioCapture
            logger.info("Initializing AudioCapture...")
            self.audio_capture = AudioCapture()
            logger.info("AudioCapture initialized")
            
            # Initialize WhisperTranscriber
            logger.info("Initializing WhisperTranscriber...")
            self.transcriber = WhisperTranscriber(model_size=settings.whisper_model)
            logger.info("WhisperTranscriber initialized")
            
            # Initialize QuestionExtractor
            logger.info("Initializing QuestionExtractor...")
            self.question_extractor = QuestionExtractor(
                llama_server_url=settings.llama_server_url,
                model_name=settings.question_extractor_model_name,
                max_retries=settings.question_extractor_max_retries,
                timeout=settings.question_extractor_timeout
            )
            logger.info("QuestionExtractor initialized")
            
            # Initialize ClipboardManager
            logger.info("Initializing ClipboardManager...")
            self.clipboard_manager = ClipboardManager()
            logger.info("ClipboardManager initialized")
            
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
                    f"Server at {settings.llama_server_url} may not be running."
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}", exc_info=True)
            return False
    
    def _process_loop(self):
        """
        Main processing loop.
        
        Continuously captures audio, transcribes, extracts questions,
        and copies to clipboard until stop is requested.
        """
        try:
            # Start audio capture
            logger.info("Starting audio capture...")
            self.audio_capture.start()
            logger.info("Audio capture started")
            
            # Main event loop
            while self._running and not self._stop_requested:
                # Handle pause
                if self._paused:
                    time.sleep(0.1)
                    continue
                
                try:
                    # Poll for audio chunk
                    audio_chunk = self.audio_capture.get_audio_chunk()
                    
                    if audio_chunk is None:
                        # No chunk available yet, sleep briefly
                        time.sleep(0.1)
                        continue
                    
                    # Calculate audio duration
                    chunk_duration = len(audio_chunk) / settings.sample_rate
                    audio_time = chunk_duration
                    
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
                    timestamp = time.strftime('%H:%M:%S')
                    
                    logger.info(
                        f"Transcription completed: {len(transcribed_text)} characters "
                        f"in {transcribe_time:.2f}s"
                    )
                    
                    # Emit transcription to GUI
                    self.transcription_ready.emit(timestamp, transcribed_text)
                    
                    # Extract questions
                    logger.debug("Starting question extraction...")
                    extract_start = time.time()
                    questions = self.question_extractor.extract_questions(transcribed_text)
                    extract_time = time.time() - extract_start
                    
                    if questions is None:
                        logger.warning("Question extraction failed")
                        # Still emit timing update
                        self.timing_update.emit(audio_time, transcribe_time, extract_time)
                        continue
                    
                    if not questions or not questions.strip():
                        logger.info("No questions detected in transcription")
                        # Still emit timing update
                        self.timing_update.emit(audio_time, transcribe_time, extract_time)
                        continue
                    
                    logger.info(
                        f"Question extraction completed: {len(questions)} characters "
                        f"in {extract_time:.2f}s"
                    )
                    
                    # Emit question to GUI
                    self.question_extracted.emit(timestamp, questions)
                    
                    # Emit timing metrics
                    self.timing_update.emit(audio_time, transcribe_time, extract_time)
                    
                    # Copy to clipboard
                    logger.debug("Copying questions to clipboard...")
                    copy_success = self.clipboard_manager.copy_to_clipboard(questions)
                    
                    if copy_success:
                        logger.info("Questions successfully copied to clipboard")
                    else:
                        logger.warning("Failed to copy questions to clipboard")
                    
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    continue
            
            logger.info("Processing loop terminated")
            
        except Exception as e:
            logger.error(f"Fatal error in processing loop: {e}", exc_info=True)
            self.status_changed.emit(f"Error: {e}")
    
    def _cleanup(self):
        """Clean up all pipeline components."""
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
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def pause(self):
        """Pause the processing loop."""
        if self._running and not self._paused:
            logger.info("Pausing pipeline...")
            self._paused = True
            
            # Stop audio capture but keep models loaded
            if self.audio_capture is not None:
                try:
                    self.audio_capture.stop()
                    logger.info("Audio capture paused")
                except Exception as e:
                    logger.error(f"Error pausing audio capture: {e}")
            
            self.status_changed.emit("Paused")
    
    def resume(self):
        """Resume the processing loop."""
        if self._running and self._paused:
            logger.info("Resuming pipeline...")
            
            # Restart audio capture
            if self.audio_capture is not None:
                try:
                    self.audio_capture.start()
                    logger.info("Audio capture resumed")
                except Exception as e:
                    logger.error(f"Error resuming audio capture: {e}")
            
            self._paused = False
            self.status_changed.emit("Running")
    
    def stop(self):
        """Stop the worker thread and cleanup."""
        logger.info("Stop requested")
        self._stop_requested = True
        self._running = False
        self.status_changed.emit("Stopping...")
