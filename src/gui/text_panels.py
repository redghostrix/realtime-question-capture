"""
Custom text panel widgets for displaying pipeline data.

This module provides specialized QWidget-based panels for displaying
transcriptions, extracted questions, and logs with appropriate formatting.
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
    Panel for displaying extracted questions with copy buttons.
    
    Shows extracted questions with timestamps and provides individual
    copy-to-clipboard buttons for each question.
    """
    
    def __init__(self):
        """Initialize the questions panel."""
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title label
        title = QLabel("Extracted Questions")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Text display area
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("Waiting for questions...")
        layout.addWidget(self.text_display)
        
        self.setLayout(layout)
        
        # Store questions for copy functionality
        self.questions_list = []
    
    def add_question(self, timestamp: str, question: str):
        """
        Add a new question entry.
        
        Args:
            timestamp: Timestamp string (HH:MM:SS)
            question: Extracted question text
        """
        # Store question
        self.questions_list.append((timestamp, question))
        
        # Format entry with copy button indicator
        entry = (
            f"<span style='color: #569cd6;'>[{timestamp}]</span> "
            f"<span style='color: #4ec9b0;'>{question}</span> "
            f"<span style='color: #808080; font-size: 10px;'>[Ctrl+C to copy]</span>"
        )
        
        # Append to display
        self.text_display.append(entry)
        
        # Auto-scroll to bottom
        self.text_display.moveCursor(QTextCursor.End)
    
    def copy_last_question(self):
        """Copy the last question to clipboard."""
        if self.questions_list:
            _, question = self.questions_list[-1]
            try:
                pyperclip.copy(question)
                return True
            except Exception:
                return False
        return False
    
    def clear(self):
        """Clear all question entries."""
        self.text_display.clear()
        self.questions_list.clear()


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
