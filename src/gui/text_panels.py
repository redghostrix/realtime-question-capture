"""
Custom text panel widgets for displaying pipeline data.

This module provides specialized QWidget-based panels for displaying
transcriptions, accumulated text, and logs with appropriate formatting.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QTextCursor, QColor
import pyperclip


class TranscriptionPanel(QWidget):
    """
    Panel for displaying transcribed text with timestamps.
    
    Shows transcription results as they arrive from the worker thread,
    with automatic scrolling to latest entries.
    """
    
    def __init__(self):
        """Initialize the transcription panel."""
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title label
        title = QLabel("Transcription")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Text display area
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("Waiting for transcription...")
        layout.addWidget(self.text_display)
        
        self.setLayout(layout)
    
    def add_transcription(self, timestamp: str, text: str):
        """
        Add a new transcription entry.
        
        Args:
            timestamp: Timestamp string (HH:MM:SS)
            text: Transcribed text
        """
        # Format entry
        entry = f"<span style='color: #569cd6;'>[{timestamp}]</span> {text}"
        
        # Append to display
        self.text_display.append(entry)
        
        # Auto-scroll to bottom
        self.text_display.moveCursor(QTextCursor.End)
    
    def clear(self):
        """Clear all transcription entries."""
        self.text_display.clear()


class QuestionsPanel(QWidget):
    """
    Panel for displaying accumulated text.
    
    Shows accumulated transcribed text with timestamps and provides
    copy-to-clipboard functionality.
    """
    
    def __init__(self):
        """Initialize the accumulated text panel."""
        super().__init__()
        
        # Text accumulation
        self.accumulated_text = ""
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title label
        self.title_label = QLabel("Accumulated Text")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title_label)
        
        # Text display area
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("Waiting for speech...")
        layout.addWidget(self.text_display)
        
        self.setLayout(layout)
    
    def clear(self):
        """Clear all accumulated text."""
        self.text_display.clear()
        self.accumulated_text = ""
    
    def add_accumulated_text(self, timestamp: str, text: str):
        """
        Add or update accumulated text.
        
        Args:
            timestamp: Timestamp string (HH:MM:SS)
            text: Accumulated text (full buffer)
        """
        self.accumulated_text = text
        
        # Format entry
        entry = f"<span style='color: #569cd6;'>[{timestamp}]</span> {text}"
        
        # Replace display content
        self.text_display.clear()
        self.text_display.append(entry)
        
        # Auto-scroll to bottom
        self.text_display.moveCursor(QTextCursor.End)
    
    def clear_accumulated_text(self):
        """Clear accumulated text buffer and display."""
        self.accumulated_text = ""
        self.text_display.clear()
    
    def get_accumulated_text(self) -> str:
        """
        Get the current accumulated text.
        
        Returns:
            Current accumulated text string
        """
        return self.accumulated_text


class LogsPanel(QWidget):
    """
    Panel for displaying logs with color-coding by level.
    
    Shows log messages with timestamps and appropriate color coding:
    - INFO: White
    - WARNING: Yellow
    - ERROR: Red
    - DEBUG: Gray
    """
    
    def __init__(self):
        """Initialize the logs panel."""
        super().__init__()
        self._setup_ui()
        
        # Define color mapping for log levels
        self.level_colors = {
            'DEBUG': '#808080',      # Gray
            'INFO': '#d4d4d4',       # White
            'WARNING': '#dcdcaa',    # Yellow
            'ERROR': '#f48771',      # Red
            'CRITICAL': '#ff0000'    # Bright red
        }
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title label with clear button
        title_layout = QHBoxLayout()
        title = QLabel("Logs")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(60)
        clear_btn.clicked.connect(self.clear)
        title_layout.addWidget(clear_btn)
        
        layout.addLayout(title_layout)
        
        # Text display area
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("Waiting for logs...")
        layout.addWidget(self.text_display)
        
        self.setLayout(layout)
    
    def add_log(self, timestamp: str, level: str, message: str):
        """
        Add a new log entry.
        
        Args:
            timestamp: Timestamp string (HH:MM:SS)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
        """
        # Get color for log level
        color = self.level_colors.get(level, '#d4d4d4')
        
        # Format entry with color-coded level
        entry = (
            f"<span style='color: #569cd6;'>[{timestamp}]</span> "
            f"<span style='color: {color}; font-weight: bold;'>[{level}]</span> "
            f"<span style='color: {color};'>{message}</span>"
        )
        
        # Append to display
        self.text_display.append(entry)
        
        # Auto-scroll to bottom
        self.text_display.moveCursor(QTextCursor.End)
    
    def clear(self):
        """Clear all log entries."""
        self.text_display.clear()
