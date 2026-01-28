"""Configuration management for the real-time transcription capture pipeline.

This module provides type-safe configuration management using Pydantic Settings
with automatic .env file loading and validation.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings with type validation and .env file support.
    
    Configuration precedence: Environment variables > .env file > Defaults
    
    All settings can be overridden via environment variables.
    Environment variable names are case-insensitive and match the field names.
    
    Example .env file:
        WHISPER_MODEL=small
        SAMPLE_RATE=16000
    """
    
    # Whisper model configuration
    whisper_model: str = Field(
        default="base",
        description="Whisper model size for speech transcription"
    )
    
    # Audio capture configuration
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz (Whisper expects 16kHz)"
    )
    
    chunk_duration: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Audio chunk duration in seconds"
    )
    
    channels: int = Field(
        default=1,
        description="Number of audio channels (1=mono, 2=stereo)"
    )
    
    # Logging configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for application output"
    )
    
    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Validate sample rate is reasonable for audio processing."""
        if v < 8000 or v > 48000:
            raise ValueError("sample_rate must be between 8000 and 48000 Hz")
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in .env file
    )


# Create a global settings instance
# This will be imported and used throughout the application
settings = Settings()
