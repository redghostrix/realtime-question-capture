"""
Audio capture module using pyaudiowpatch for Windows WASAPI loopback.

This module captures system audio output from any application using Windows WASAPI
loopback capability via pyaudiowpatch.
"""

import logging
import queue
import threading
import numpy as np
import pyaudiowpatch as pyaudio

from .config import SAMPLE_RATE, CHUNK_DURATION, CHANNELS


logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Captures system audio using Windows WASAPI loopback.
    
    This class handles device enumeration, audio stream management, buffering,
    and provides clean start/stop/retrieve interfaces for real-time audio capture.
    """
    
    def __init__(self):
        """Initialize the AudioCapture instance."""
        # Initialize PyAudio instance with pyaudiowpatch support
        self.p = pyaudio.PyAudio()
        
        # Store configuration parameters
        self.sample_rate = SAMPLE_RATE
        self.chunk_duration = CHUNK_DURATION
        self.channels = CHANNELS
        
        # Calculate chunk size in frames, then convert to samples
        # frames_per_chunk = sample_rate * chunk_duration
        # target_samples = frames_per_chunk * channels
        self.frames_per_chunk = SAMPLE_RATE * CHUNK_DURATION
        self.chunk_size = self.frames_per_chunk * CHANNELS
        
        # Initialize thread-safe audio queue for buffering
        self.audio_queue = queue.Queue()
        
        # Initialize stream and thread references
        self.stream = None
        self.capture_thread = None
        
        # Capture state flag
        self.is_capturing = False
        
        logger.debug("AudioCapture initialized with sample_rate=%d, chunk_duration=%d, channels=%d",
                    self.sample_rate, self.chunk_duration, self.channels)
    
    def _get_default_wasapi_loopback_device(self):
        """
        Find and return the default WASAPI loopback device.
        
        Returns the loopback counterpart of the default output device to ensure
        we capture audio from the actual default system output.
        
        Returns:
            int: Device index of the loopback device
            
        Raises:
            RuntimeError: If no loopback device is found
        """
        logger.debug("Searching for default WASAPI loopback device...")
        
        try:
            # Get the default output device info
            default_output_info = self.p.get_default_output_device_info()
            default_output_name = default_output_info.get('name', '')
            default_output_index = default_output_info.get('index', -1)
            logger.debug("Default output device: %s (index=%d)", default_output_name, default_output_index)
            
            # Try to find the loopback counterpart of the default output device
            # pyaudiowpatch creates loopback devices with specific naming patterns
            device_count = self.p.get_device_count()
            logger.debug("Found %d audio devices", device_count)
            
            for i in range(device_count):
                try:
                    device_info = self.p.get_device_info_by_index(i)
                    device_name = device_info.get('name', '')
                    
                    # Check for WASAPI loopback capability
                    # pyaudiowpatch adds 'isLoopback' flag to device info
                    # Also check for "[Loopback]" in the device name
                    is_loopback = device_info.get('isLoopback', False)
                    is_loopback_name = '[Loopback]' in device_name or 'loopback' in device_name.lower()
                    has_input_channels = device_info.get('maxInputChannels', 0) > 0
                    
                    if (is_loopback or is_loopback_name) and has_input_channels:
                        # Check if this loopback device corresponds to the default output
                        # pyaudiowpatch loopback devices typically have similar names to their output counterpart
                        # or contain the output device name as a substring
                        if default_output_name in device_name or device_info.get('defaultSampleRate') == default_output_info.get('defaultSampleRate'):
                            logger.info("Found default WASAPI loopback device: %s (index=%d)", device_name, i)
                            return i
                        
                except Exception as e:
                    logger.debug("Error checking device %d: %s", i, e)
                    continue
            
            # If we didn't find an exact match, fall back to the first loopback device
            logger.warning("Could not find loopback device matching default output, searching for any loopback device")
            
            for i in range(device_count):
                try:
                    device_info = self.p.get_device_info_by_index(i)
                    device_name = device_info.get('name', '')
                    
                    is_loopback = device_info.get('isLoopback', False)
                    has_input_channels = device_info.get('maxInputChannels', 0) > 0
                    is_stereo_mix = 'Stereo Mix' in device_name
                    is_loopback_name = '[Loopback]' in device_name or 'loopback' in device_name.lower()
                    
                    if (is_loopback or is_stereo_mix or is_loopback_name) and has_input_channels:
                        logger.info("Found fallback WASAPI loopback device: %s (index=%d)", device_name, i)
                        return i
                        
                except Exception as e:
                    logger.debug("Error checking device %d: %s", i, e)
                    continue
                    
        except Exception as e:
            logger.warning("Could not get default output device: %s", e)
        
        # No loopback device found
        error_msg = (
            "No WASAPI loopback device found. "
            "Please ensure you are running on Windows with WASAPI support. "
            "You may need to enable 'Stereo Mix' in your sound settings or "
            "ensure your audio drivers support loopback capture."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Audio stream callback function.
        
        This is called by PyAudio in a separate thread whenever new audio data is available.
        
        Args:
            in_data: Raw audio bytes from the stream
            frame_count: Number of frames
            time_info: Timing information
            status: Status flags
            
        Returns:
            tuple: (None, pyaudio.paContinue) to continue the stream
        """
        if status:
            logger.warning("Audio callback status: %s", status)
        
        try:
            # Convert raw audio bytes to numpy array (16-bit signed integers)
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # If multi-channel audio is captured but fewer channels are required, convert
            # Multi-channel data is interleaved: [C1, C2, ..., CN, C1, C2, ..., CN, ...]
            if len(audio_data) > 0 and self.channels < self.device_channels:
                # Reshape to (n_frames, device_channels) and average across channels
                if len(audio_data) % self.device_channels == 0:
                    audio_data = audio_data.reshape(-1, self.device_channels).mean(axis=1).astype(np.int16)
            
            # Resample if device sample rate differs from requested sample rate
            if hasattr(self, 'device_sample_rate') and self.device_sample_rate != self.sample_rate:
                # Simple resampling using linear interpolation
                num_samples = len(audio_data)
                duration = num_samples / self.device_sample_rate
                target_samples = int(duration * self.sample_rate)
                
                # Create new sample indices
                old_indices = np.arange(num_samples)
                new_indices = np.linspace(0, num_samples - 1, target_samples)
                
                # Interpolate
                audio_data = np.interp(new_indices, old_indices, audio_data).astype(np.int16)
            
            # Put processed audio data into queue
            try:
                self.audio_queue.put(audio_data, block=False)
            except queue.Full:
                logger.warning("Audio queue is full, dropping audio chunk")
                
        except Exception as e:
            logger.error("Error in audio callback: %s", e)
        
        return (None, pyaudio.paContinue)
    
    def start(self):
        """
        Start capturing audio from the WASAPI loopback device.
        
        Raises:
            RuntimeError: If no loopback device is found or stream cannot be opened
        """
        if self.is_capturing:
            logger.warning("Audio capture already started")
            return
        
        try:
            # Get the default WASAPI loopback device
            device_index = self._get_default_wasapi_loopback_device()
            
            # Get device info for logging
            device_info = self.p.get_device_info_by_index(device_index)
            device_name = device_info.get('name', 'Unknown')
            
            logger.info("Opening audio stream on device: %s", device_name)
            
            # Determine device channels (most loopback devices are stereo)
            device_channels = device_info.get('maxInputChannels', 2)
            # Store device channels for callback processing
            self.device_channels = device_channels
            
            # Get device's native sample rate
            device_sample_rate = int(device_info.get('defaultSampleRate', 48000))
            logger.debug("Device native sample rate: %d Hz, requested: %d Hz", 
                        device_sample_rate, self.sample_rate)
            
            # Store device sample rate for potential resampling
            self.device_sample_rate = device_sample_rate
            
            # Open audio stream with WASAPI loopback
            # Note: For loopback devices, we don't need as_loopback parameter
            # The device itself is already a loopback device
            # We use the device's native sample rate and will resample if needed
            self.stream = self.p.open(
                format=pyaudio.paInt16,  # 16-bit audio
                channels=device_channels,  # Capture in device's native channel count
                rate=device_sample_rate,  # Use device's native sample rate
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,  # Small buffer for low latency
                stream_callback=self._audio_callback
            )
            
            # Start the audio stream
            self.stream.start_stream()
            
            # Update capture state
            self.is_capturing = True
            
            logger.info("Audio capture started successfully on device: %s (index=%d)", 
                       device_name, device_index)
            
        except Exception as e:
            logger.error("Failed to start audio capture: %s", e)
            raise RuntimeError(f"Failed to start audio capture: {e}")
    
    def stop(self):
        """Stop capturing audio and clean up the stream."""
        if not self.is_capturing:
            logger.warning("Audio capture is not running")
            return
        
        try:
            # Update capture state first
            self.is_capturing = False
            
            # Stop the audio stream
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
            
            # Close the stream
            if self.stream:
                self.stream.close()
                self.stream = None
            
            # Clear the audio queue to prevent memory buildup
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("Audio capture stopped successfully")
            
        except Exception as e:
            logger.error("Error stopping audio capture: %s", e)
    
    def get_audio_chunk(self):
        """
        Retrieve a chunk of audio data from the buffer.
        
        Only returns audio when at least a full chunk is available in the buffer.
        If insufficient data is available, buffered data remains in the queue
        and None is returned.
        
        Returns:
            numpy.ndarray or None: Audio samples as numpy array of shape (chunk_size,)
                                  with dtype np.int16, or None if insufficient data
        """
        # First, check if we have enough data without removing from queue
        # We need to peek at the queue to determine total available samples
        temp_chunks = []
        accumulated_samples = 0
        target_samples = self.chunk_size
        
        # Collect chunks from queue to check if we have enough
        while accumulated_samples < target_samples:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.01)
                temp_chunks.append(audio_data)
                accumulated_samples += len(audio_data)
            except queue.Empty:
                # No more data available in queue
                break
        
        # If we don't have enough samples, put everything back and return None
        if accumulated_samples < target_samples:
            # Put all chunks back into the queue in order
            for chunk in temp_chunks:
                try:
                    self.audio_queue.put(chunk, block=False)
                except queue.Full:
                    logger.warning("Queue full while restoring partial data")
            return None
        
        # We have enough data, concatenate and process
        audio_array = np.concatenate(temp_chunks)
        
        # Extract exactly chunk_size samples
        chunk_data = audio_array[:target_samples]
        
        # If there are leftover samples, put them back in the queue
        if len(audio_array) > target_samples:
            leftover = audio_array[target_samples:]
            try:
                self.audio_queue.put(leftover, block=False)
            except queue.Full:
                logger.warning("Queue full while storing leftover samples")
        
        logger.debug("Retrieved audio chunk: %d samples (%.2f seconds)", 
                    len(chunk_data), len(chunk_data) / self.sample_rate)
        
        return chunk_data
    
    def close(self):
        """Clean up resources and terminate PyAudio."""
        try:
            # Stop capture if currently running
            if self.is_capturing:
                self.stop()
            
            # Terminate PyAudio instance
            if self.p:
                self.p.terminate()
                self.p = None
            
            logger.info("AudioCapture cleanup completed")
            
        except Exception as e:
            logger.error("Error during cleanup: %s", e)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False
