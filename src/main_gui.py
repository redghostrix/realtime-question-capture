"""
GUI entry point for real-time question capture application.

This module provides the main entry point for the PyQt5-based desktop
monitoring application.
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from src.gui.main_window import MainWindow
from src.config import settings


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point for the GUI application."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Real-Time Question Capture - GUI Mode")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Whisper Model: {settings.whisper_model}")
    logger.info(f"  LLM Server URL: {settings.llama_server_url}")
    logger.info(f"  Sample Rate: {settings.sample_rate} Hz")
    logger.info(f"  Chunk Duration: {settings.chunk_duration} seconds")
    logger.info(f"  Channels: {settings.channels}")
    logger.info(f"  Log Level: {settings.log_level}")
    logger.info("=" * 80)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Real-Time Question Capture")
    
    # Enable high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    logger.info("GUI application started")
    
    # Run application event loop
    exit_code = app.exec_()
    
    logger.info(f"GUI application exited with code {exit_code}")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
