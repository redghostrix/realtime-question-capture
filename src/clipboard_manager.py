"""Clipboard management module for copying extracted questions.

This module provides a simple interface for copying text to the system clipboard
using pyperclip, with proper error handling and logging.
"""

import logging
from typing import Optional
import pyperclip

# Module-level logger
logger = logging.getLogger(__name__)


class ClipboardManager:
    """Manage clipboard operations for question text.
    
    This class provides a simple interface to copy text to the system clipboard
    with proper error handling and logging. It wraps the pyperclip library to
    provide consistent behavior across the application.
    
    Features:
    - Copy text to clipboard with validation
    - Retrieve current clipboard content
    - Error handling for clipboard access issues
    - Context manager support for consistency
    
    Usage:
        with ClipboardManager() as clipboard:
            success = clipboard.copy_to_clipboard("What is the meaning of life?")
            if success:
                print("Question copied to clipboard")
    """
    
    def __init__(self):
        """Initialize the ClipboardManager."""
        logger.debug("ClipboardManager initialized")
    
    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to the system clipboard.
        
        Args:
            text: The text to copy to clipboard
            
        Returns:
            True if copy was successful, False otherwise
        """
        # Validate input
        if not text or not text.strip():
            logger.debug("Skipping clipboard copy: empty or whitespace-only text")
            return False
        
        try:
            # Copy to clipboard
            pyperclip.copy(text)
            
            # Create preview (first 50 characters)
            preview = text[:50] + "..." if len(text) > 50 else text
            preview = preview.replace("\n", " ")  # Replace newlines for cleaner log
            
            logger.info(f"Copied to clipboard: {len(text)} characters - '{preview}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy text to clipboard: {e}")
            return False
    
    def get_clipboard_content(self) -> Optional[str]:
        """Retrieve current clipboard content.
        
        Returns:
            Current clipboard content as string, or None on failure
        """
        try:
            content = pyperclip.paste()
            logger.debug(f"Retrieved clipboard content: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Failed to retrieve clipboard content: {e}")
            return None
    
    def close(self):
        """Clean up resources.
        
        This is a no-op for ClipboardManager since pyperclip doesn't require
        explicit cleanup, but provided for consistency with other components.
        """
        logger.debug("ClipboardManager cleanup completed (no-op)")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False  # Propagate exceptions
