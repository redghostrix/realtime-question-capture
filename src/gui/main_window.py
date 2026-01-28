"""
Main window for the GUI application.

This module provides the main application window that orchestrates all
GUI components and manages the pipeline worker thread.
"""

import logging
import pyperclip
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStatusBar, QSplitter
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent

from .control_panel import ControlPanel
from .text_panels import TranscriptionPanel, QuestionsPanel, LogsPanel
from .worker_thread import PipelineWorker
from .log_handler import QtLogHandler


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Provides the complete GUI interface with control panel, data display panels,
    status bar, and manages the pipeline worker thread.
    """
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        # Worker thread
        self.worker = None
        
        # Set up logging handler
        self.log_handler = None
        
        # Set up UI
        self._setup_ui()
        self._apply_dark_theme()
        self._setup_logging()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Set window properties
        self.setWindowTitle("Real-Time Transcription Monitor")
        self.setMinimumSize(1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (horizontal: control panel + content area)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create control panel (left sidebar)
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel)
        
        # Create content area (3 vertical panels)
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(5)
        
        # Create panels
        self.transcription_panel = TranscriptionPanel()
        self.questions_panel = QuestionsPanel()
        self.logs_panel = LogsPanel()
        
        # Add panels to content area with equal heights
        content_layout.addWidget(self.transcription_panel, 1)
        content_layout.addWidget(self.questions_panel, 1)
        content_layout.addWidget(self.logs_panel, 1)
        
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget, 1)
        
        central_widget.setLayout(main_layout)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Connect control panel signals
        self.control_panel.start_clicked.connect(self._on_start)
        self.control_panel.stop_clicked.connect(self._on_stop)
        self.control_panel.pause_clicked.connect(self._on_pause)
    
    def _setup_logging(self):
        """Set up the custom logging handler for GUI display."""
        # Create and configure Qt log handler
        self.log_handler = QtLogHandler()
        self.log_handler.setLevel(logging.INFO)
        
        # Connect log handler to logs panel
        self.log_handler.log_message.connect(self.logs_panel.add_log)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
        logger.info("Logging handler configured")
    
    def _apply_dark_theme(self):
        """Apply dark theme stylesheet to the application."""
        dark_theme = """
        QMainWindow {
            background-color: #1e1e1e;
        }
        
        QWidget {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11px;
        }
        
        QTextEdit {
            background-color: #252526;
            color: #d4d4d4;
            border: 1px solid #3e3e42;
            border-radius: 3px;
            padding: 5px;
        }
        
        QLabel {
            background-color: transparent;
            color: #d4d4d4;
        }
        
        QPushButton {
            background-color: #0e639c;
            color: #ffffff;
            border: none;
            border-radius: 3px;
            padding: 8px 15px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #1177bb;
        }
        
        QPushButton:pressed {
            background-color: #0d5a8f;
        }
        
        QPushButton:disabled {
            background-color: #3e3e42;
            color: #808080;
        }
        
        QStatusBar {
            background-color: #007acc;
            color: #ffffff;
            font-weight: bold;
        }
        
        QFrame[frameShape="4"] {
            color: #3e3e42;
        }
        
        QScrollBar:vertical {
            background-color: #1e1e1e;
            width: 12px;
            margin: 0px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #424242;
            min-height: 20px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #4e4e4e;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QScrollBar:horizontal {
            background-color: #1e1e1e;
            height: 12px;
            margin: 0px;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #424242;
            min-width: 20px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #4e4e4e;
        }
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        """
        
        self.setStyleSheet(dark_theme)
    
    def _on_start(self):
        """Handle start button click."""
        logger.info("Start button clicked")
        
        # Create and configure worker thread
        self.worker = PipelineWorker()
        
        # Connect worker signals to UI slots
        self.worker.transcription_ready.connect(self.transcription_panel.add_transcription)
        self.worker.accumulated_text_updated.connect(self.questions_panel.add_accumulated_text)
        self.worker.accumulated_text_cleared.connect(self.questions_panel.clear_accumulated_text)
        self.worker.timing_update.connect(self._update_timing_metrics)
        self.worker.status_changed.connect(self._update_status)
        self.worker.initialization_complete.connect(self._on_initialization_complete)
        
        # Start worker thread
        self.worker.start()
        
        logger.info("Worker thread started")
    
    def _on_stop(self):
        """Handle stop button click."""
        logger.info("Stop button clicked")
        
        if self.worker is not None:
            # Stop worker thread
            self.worker.stop()
            
            # Wait for thread to finish (with timeout)
            if not self.worker.wait(5000):  # 5 second timeout
                logger.warning("Worker thread did not stop gracefully, terminating...")
                self.worker.terminate()
                self.worker.wait()
            
            self.worker = None
            logger.info("Worker thread stopped")
        
        # Update status
        self.control_panel.update_status("Stopped")
        self.status_bar.showMessage("Stopped")
    
    def _on_pause(self):
        """Handle pause/resume button click."""
        if self.worker is not None:
            if self.worker._paused:
                logger.info("Resume button clicked")
                self.worker.resume()
            else:
                logger.info("Pause button clicked")
                self.worker.pause()
    
    def _update_timing_metrics(self, audio_time: float, trans_time: float):
        """
        Update timing metrics in status bar.
        
        Args:
            audio_time: Audio capture duration in seconds
            trans_time: Transcription time in seconds
        """
        # Format timing message
        message = (
            f"Audio: {audio_time:.2f}s | "
            f"Transcription: {trans_time:.2f}s"
        )
        
        # Update status bar
        self.status_bar.showMessage(message)
    
    def _update_status(self, status: str):
        """
        Update status indicator.
        
        Args:
            status: Status string
        """
        self.control_panel.update_status(status)
        
        # Also update status bar if not showing timing metrics
        if "Audio:" not in self.status_bar.currentMessage():
            self.status_bar.showMessage(status)
    
    def _on_initialization_complete(self, success: bool, message: str):
        """
        Handle initialization completion.
        
        Args:
            success: True if initialization successful
            message: Status message
        """
        if success:
            logger.info(f"Initialization successful: {message}")
        else:
            logger.error(f"Initialization failed: {message}")
            # Re-enable start button if initialization failed
            self.control_panel.enable_buttons(True)
            self.control_panel.update_status("Error")
    
    def keyPressEvent(self, event: QKeyEvent):
        """
        Handle keyboard events for hotkeys.
        
        Args:
            event: Key event
        """
        # Check for Ctrl+Shift+C
        if (event.key() == Qt.Key_C and 
            event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier)):
            self._copy_accumulated_text()
        else:
            # Pass event to parent
            super().keyPressEvent(event)
    
    def _copy_accumulated_text(self):
        """Copy accumulated text to clipboard and show status message."""
        # Get accumulated text from questions panel
        accumulated_text = self.questions_panel.get_accumulated_text()
        
        if not accumulated_text:
            logger.info("No accumulated text to copy")
            self.status_bar.showMessage("No text to copy", 2000)
            return
        
        try:
            # Copy to clipboard
            pyperclip.copy(accumulated_text)
            char_count = len(accumulated_text)
            logger.info(f"Copied {char_count} characters to clipboard via Ctrl+Shift+C")
            self.status_bar.showMessage(f"Copied {char_count} characters to clipboard", 3000)
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            self.status_bar.showMessage("Failed to copy to clipboard", 3000)
    
    def closeEvent(self, event):
        """
        Handle window close event.
        
        Args:
            event: Close event
        """
        logger.info("Application closing...")
        
        # Stop worker thread if running
        if self.worker is not None and self.worker.isRunning():
            logger.info("Stopping worker thread...")
            self.worker.stop()
            
            if not self.worker.wait(5000):
                logger.warning("Worker thread did not stop gracefully, terminating...")
                self.worker.terminate()
                self.worker.wait()
        
        # Remove log handler
        if self.log_handler is not None:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.log_handler)
        
        logger.info("Application closed")
        
        # Accept the close event
        event.accept()
