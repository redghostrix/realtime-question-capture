"""
Control panel for pipeline management.

This module provides the left sidebar control panel with start/stop/pause
buttons, status indicator, and configuration display.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFrame, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from src.config import settings


class ControlPanel(QWidget):
    """
    Control panel widget for managing the pipeline.
    
    Provides buttons to start, stop, and pause the pipeline,
    displays current status, and shows configuration settings.
    
    Signals:
        start_clicked: Emitted when Start button is clicked
        stop_clicked: Emitted when Stop button is clicked
        pause_clicked: Emitted when Pause button is clicked
    """
    
    # Define signals
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    extraction_mode_changed = pyqtSignal(bool)  # True = extraction mode, False = accumulation mode
    
    def __init__(self):
        """Initialize the control panel."""
        super().__init__()
        self._is_paused = False
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Control Panel")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)
        
        # Control buttons
        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self._on_start)
        layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setMinimumHeight(40)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._on_pause)
        layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop)
        layout.addWidget(self.stop_button)
        
        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Status section
        status_label = QLabel("Status:")
        status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(status_label)
        
        self.status_indicator = QLabel("● Stopped")
        self.status_indicator.setStyleSheet("color: #808080; font-size: 14px;")
        layout.addWidget(self.status_indicator)
        
        # Separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator3)
        
        # Mode selection
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(mode_label)
        
        self.extraction_checkbox = QCheckBox("Enable Question Extraction")
        self.extraction_checkbox.setChecked(True)  # Default to extraction mode
        self.extraction_checkbox.stateChanged.connect(self._on_extraction_mode_changed)
        layout.addWidget(self.extraction_checkbox)
        
        # Separator
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.HLine)
        separator4.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator4)
        
        # Configuration section
        config_label = QLabel("Configuration:")
        config_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(config_label)
        
        # Display current configuration from settings
        config_text = f"""
Model: {settings.whisper_model}
Rate: {settings.sample_rate} Hz
Chunk: {settings.chunk_duration}s
Channels: {settings.channels}
        """.strip()
        
        self.config_display = QLabel(config_text)
        self.config_display.setStyleSheet("color: #d4d4d4; font-size: 12px;")
        self.config_display.setWordWrap(True)
        layout.addWidget(self.config_display)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        self.setLayout(layout)
        self.setFixedWidth(200)
    
    def _on_start(self):
        """Handle start button click."""
        self.start_clicked.emit()
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self._is_paused = False
        self.pause_button.setText("Pause")
    
    def _on_pause(self):
        """Handle pause button click."""
        if self._is_paused:
            # Resume
            self.pause_clicked.emit()
            self._is_paused = False
            self.pause_button.setText("Pause")
        else:
            # Pause
            self.pause_clicked.emit()
            self._is_paused = True
            self.pause_button.setText("Resume")
    
    def _on_stop(self):
        """Handle stop button click."""
        self.stop_clicked.emit()
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._is_paused = False
        self.pause_button.setText("Pause")
    
    def _on_extraction_mode_changed(self, state):
        """Handle extraction mode checkbox state change."""
        is_checked = (state == 2)  # Qt.Checked = 2
        self.extraction_mode_changed.emit(is_checked)
    
    def update_status(self, status: str):
        """
        Update the status indicator.
        
        Args:
            status: Status string (Running, Stopped, Paused, etc.)
        """
        # Map status to color
        color_map = {
            'Running': '#4ec9b0',      # Green
            'Stopped': '#808080',      # Gray
            'Paused': '#dcdcaa',       # Yellow
            'Initializing': '#569cd6', # Blue
            'Stopping': '#dcdcaa',     # Yellow
            'Error': '#f48771'         # Red
        }
        
        # Get color (default to white for unknown statuses)
        color = color_map.get(status, '#d4d4d4')
        
        # Update status display
        self.status_indicator.setText(f"● {status}")
        self.status_indicator.setStyleSheet(f"color: {color}; font-size: 14px;")
    
    def enable_buttons(self, enabled: bool):
        """
        Enable or disable all control buttons.
        
        Args:
            enabled: True to enable, False to disable
        """
        if enabled:
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        else:
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
