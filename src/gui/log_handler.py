"""
Custom logging handler for GUI integration.

This module provides a thread-safe logging handler that emits Qt signals,
allowing logs from any thread to be displayed in the GUI.
"""

import logging
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal


class QtLogHandler(logging.Handler, QObject):
    """
    Custom logging handler that emits Qt signals for GUI display.
    
    This handler intercepts Python logging messages from all modules and
    emits them as Qt signals, which can be safely connected to GUI widgets
    for display.
    
    Signals:
        log_message: Emitted when a log message is received
                    Args: timestamp (str), level (str), message (str)
    """
    
    # Define signal for log messages
    log_message = pyqtSignal(str, str, str)  # timestamp, level, message
    
    def __init__(self):
        """Initialize the Qt log handler."""
        logging.Handler.__init__(self)
        QObject.__init__(self)
        
        # Set formatter for consistent log formatting
        formatter = logging.Formatter(
            '%(name)s - %(message)s'
        )
        self.setFormatter(formatter)
    
    def emit(self, record):
        """
        Emit a log record as a Qt signal.
        
        Args:
            record: LogRecord instance from Python logging
        """
        try:
            # Get timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Get log level name
            level = record.levelname
            
            # Format message
            message = self.format(record)
            
            # Emit signal (thread-safe)
            self.log_message.emit(timestamp, level, message)
            
        except Exception as e:
            # Fallback to stderr if signal emission fails
            self.handleError(record)
