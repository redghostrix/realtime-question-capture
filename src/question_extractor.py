"""Question extraction module using DSPy and llama-server.

This module provides functionality to extract and correct questions from
transcribed speech text using a local LLM server through DSPy framework.
"""

import logging
import time
import warnings
from typing import Optional, List, Dict, Any
import dspy
import requests
from .config import settings
from .exceptions import QuestionDetectionError

# Suppress Pydantic serialization warnings from DSPy/OpenAI compatibility layer
# These occur when local LLM servers return fewer fields than the official OpenAI API
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=".*Pydantic serializer warnings.*"
)

# Module-level logger
logger = logging.getLogger(__name__)


class QuestionExtractionSignature(dspy.Signature):
    """Extract and correct questions from transcribed speech text"""
    
    transcribed_text: str = dspy.InputField(
        desc="Transcribed speech text that may contain questions"
    )
    extracted_questions: str = dspy.OutputField(
        desc="Cleaned and corrected questions extracted from the text, one per line. Return empty string if no questions found"
    )


class QuestionExtractor:
    """Extract and correct questions from transcribed speech using DSPy and llama-server.
    
    This class provides an interface to extract questions from transcribed audio text
    using a local LLM server. It uses the DSPy framework to communicate with the
    llama-server endpoint and extract/correct any questions present in the text.
    
    Features:
    - DSPy integration for structured LLM interactions
    - Communication with local llama-server endpoint
    - Question extraction and correction from transcribed text
    - Automatic retry logic for connection failures
    - Context manager support for resource cleanup
    
    Usage:
        with QuestionExtractor() as extractor:
            questions = extractor.extract_questions(transcription_text)
            if questions:
                print(f"Extracted questions: {questions}")
    
    Attributes:
        llama_server_url (str): URL of the llama-server endpoint
        model_name (str): Model identifier for DSPy
        api_key (str): API key for authentication (dummy for local server)
        max_retries (int): Maximum number of retry attempts
        timeout (int): Request timeout in seconds
    """
    
    def __init__(
        self,
        llama_server_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "local",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """Initialize the QuestionExtractor.
        
        Args:
            llama_server_url: URL of the llama-server endpoint. Defaults to settings.llama_server_url
            model_name: Model identifier for DSPy. Defaults to settings.question_extractor_model_name
            api_key: API key for authentication. Defaults to "local" (dummy for local server)
            max_retries: Maximum number of retry attempts for connection failures. Defaults to 3
            timeout: Request timeout in seconds. Defaults to 30
        """
        self.llama_server_url = llama_server_url or settings.llama_server_url
        self.model_name = model_name or settings.question_extractor_model_name
        self.api_key = api_key
        self.max_retries = max(1, max_retries)  # Ensure at least 1 retry
        self.timeout = max(1, timeout)  # Ensure positive timeout
        
        self.lm = None
        self.predictor = None
        
        logger.info(
            f"Initializing QuestionExtractor with llama_server_url={self.llama_server_url}, "
            f"model_name={self.model_name}, max_retries={self.max_retries}, timeout={self.timeout}"
        )
        
        # Initialize DSPy configuration
        self._initialize_dspy()
    
    def _initialize_dspy(self):
        """Configure DSPy with llama-server endpoint.
        
        Raises:
            ConnectionError: If unable to connect to llama-server
            Exception: For other configuration errors
        """
        try:
            logger.info(f"Configuring DSPy with endpoint: {self.llama_server_url}")
            
            # Initialize DSPy LM with llama-server endpoint
            self.lm = dspy.LM(
                model=f"openai/default",
                api_base=self.llama_server_url,
                api_key="not-needed",
                model_type="chat"
            )
            
            # Configure DSPy to use the LM
            dspy.configure(lm=self.lm)
            
            # Create predictor with question extraction signature
            self.predictor = dspy.Predict(QuestionExtractionSignature)
            
            logger.info(f"DSPy initialized successfully with endpoint: {self.llama_server_url}")
            
        except ConnectionError as e:
            logger.error(
                f"Failed to connect to llama-server at {self.llama_server_url}. "
                f"Please ensure llama-server is running. Error: {e}"
            )
            raise QuestionDetectionError(
                f"Failed to connect to llama-server at {self.llama_server_url}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize DSPy: {e}")
            raise QuestionDetectionError(f"Failed to initialize DSPy: {e}") from e
    
    def extract_questions(self, transcribed_text: str) -> Optional[str]:
        """Extract and correct questions from transcribed text.
        
        Args:
            transcribed_text: The transcribed speech text to extract questions from
        
        Returns:
            Extracted and corrected questions as a string (one per line), or None if extraction fails.
            Returns empty string if no questions are found in the text.
        """
        # Validate input
        if not transcribed_text or not transcribed_text.strip():
            logger.warning("Empty or whitespace-only transcribed text provided")
            return ""
        
        logger.debug(f"Processing transcribed text of length {len(transcribed_text)}")
        
        # Retry loop for handling connection failures with exponential backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                # Call DSPy predictor with transcribed text
                result = self.predictor(transcribed_text=transcribed_text)
                
                # Extract questions from result
                questions = result.extracted_questions.strip()
                
                logger.info(f"Extracted {len(questions)} characters of questions")
                
                if not questions:
                    logger.debug("No questions detected in transcribed text")
                
                return questions
                
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"HTTP request failed on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries:
                    # Calculate exponential backoff delay (1s, 2s, 4s, ...)
                    delay = 2 ** (attempt - 1)
                    logger.info(f"Retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"All retry attempts exhausted for HTTP request: {e}")
                    return None
                    
            except TimeoutError as e:
                logger.warning(
                    f"Request timeout on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries:
                    # Calculate exponential backoff delay (1s, 2s, 4s, ...)
                    delay = 2 ** (attempt - 1)
                    logger.info(f"Retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"All retry attempts exhausted due to timeout: {e}")
                    return None
                    
            except Exception as e:
                logger.warning(
                    f"Unexpected error on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries:
                    # Calculate exponential backoff delay (1s, 2s, 4s, ...)
                    delay = 2 ** (attempt - 1)
                    logger.info(f"Retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"All retry attempts exhausted due to unexpected error: {e}")
                    return None
        
        # Should not reach here, but handle gracefully
        logger.error("All retry attempts failed")
        return None
    
    def check_server_health(self) -> bool:
        """Check if the llama-server is healthy and responding.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Try to construct health endpoint from base URL
            health_url = self.llama_server_url.replace('/v1/chat/completions', '/health')
            
            response = requests.get(health_url, timeout=5)
            healthy = response.status_code == 200
            
            logger.debug(f"Server health check: {'healthy' if healthy else 'unhealthy'}")
            return healthy
            
        except Exception as e:
            logger.debug(f"Server health check failed: {e}")
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the configured llama-server.
        
        Returns:
            Dictionary containing server configuration and status
        """
        info = {
            'llama_server_url': self.llama_server_url,
            'model_name': self.model_name,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'server_healthy': self.check_server_health()
        }
        
        logger.debug(f"Server info: {info}")
        return info
    
    def close(self):
        """Clean up resources and close connections.
        
        This method should be called when the QuestionExtractor is no longer needed.
        It's automatically called when using the context manager interface.
        """
        try:
            logger.info("Closing QuestionExtractor...")
            
            # Clear predictor and LM references
            self.predictor = None
            self.lm = None
            
            logger.info("QuestionExtractor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during QuestionExtractor cleanup: {e}")
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources."""
        self.close()
        return False
