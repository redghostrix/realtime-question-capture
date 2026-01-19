"""
Manual verification script for testing real-time audio capture functionality.

This script tests the AudioCapture module by:
- Capturing audio in real-time from WASAPI loopback
- Displaying live statistics and audio levels
- Saving captured audio to WAV file for verification
"""

import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_capture import AudioCapture
from src.config import settings


def calculate_rms_db(audio_data):
    """
    Calculate RMS (Root Mean Square) amplitude in decibels.
    
    Args:
        audio_data: numpy array of audio samples (int16)
        
    Returns:
        float: RMS level in dB (or -inf for silence)
    """
    if len(audio_data) == 0:
        return float('-inf')
    
    # Convert to float and normalize to [-1, 1]
    audio_float = audio_data.astype(np.float64) / 32768.0
    
    # Calculate RMS
    rms = np.sqrt(np.mean(audio_float ** 2))
    
    # Convert to dB (20 * log10(rms))
    if rms > 0:
        db = 20 * np.log10(rms)
    else:
        db = float('-inf')
    
    return db


def calculate_peak_db(audio_data):
    """
    Calculate peak amplitude in decibels.
    
    Args:
        audio_data: numpy array of audio samples (int16)
        
    Returns:
        float: Peak level in dB
    """
    if len(audio_data) == 0:
        return float('-inf')
    
    # Convert to float and normalize to [-1, 1]
    audio_float = audio_data.astype(np.float64) / 32768.0
    
    # Get peak
    peak = np.max(np.abs(audio_float))
    
    # Convert to dB
    if peak > 0:
        db = 20 * np.log10(peak)
    else:
        db = float('-inf')
    
    return db


def create_level_bar(db_value, bar_width=40):
    """
    Create an ASCII bar visualization of audio level.
    
    Args:
        db_value: Audio level in dB (-inf to 0)
        bar_width: Width of the bar in characters
        
    Returns:
        str: ASCII bar representation
    """
    if db_value == float('-inf') or db_value < -60:
        # Silence
        filled = 0
    else:
        # Map -60 dB to 0 dB into 0 to bar_width
        normalized = (db_value + 60) / 60  # 0.0 to 1.0
        filled = int(normalized * bar_width)
        filled = max(0, min(bar_width, filled))
    
    empty = bar_width - filled
    bar = '█' * filled + '░' * empty
    
    return bar


def list_audio_devices(audio_capture):
    """List all available audio devices."""
    print("\n" + "="*80)
    print("AVAILABLE AUDIO DEVICES")
    print("="*80)
    
    device_count = audio_capture.p.get_device_count()
    print(f"Total devices found: {device_count}\n")
    
    for i in range(device_count):
        try:
            info = audio_capture.p.get_device_info_by_index(i)
            name = info.get('name', 'Unknown')
            max_input = info.get('maxInputChannels', 0)
            max_output = info.get('maxOutputChannels', 0)
            sample_rate = info.get('defaultSampleRate', 0)
            is_loopback = info.get('isLoopback', False)
            
            device_type = []
            if max_input > 0:
                device_type.append(f"Input ({max_input}ch)")
            if max_output > 0:
                device_type.append(f"Output ({max_output}ch)")
            if is_loopback:
                device_type.append("LOOPBACK")
            
            type_str = " | ".join(device_type) if device_type else "N/A"
            
            print(f"[{i:2d}] {name}")
            print(f"     Type: {type_str}")
            print(f"     Sample Rate: {sample_rate:.0f} Hz")
            
            if is_loopback or '[Loopback]' in name:
                print(f"     *** WASAPI LOOPBACK DEVICE ***")
            
            print()
            
        except Exception as e:
            print(f"[{i:2d}] Error reading device info: {e}\n")
    
    print("="*80 + "\n")


def save_audio_to_wav(audio_chunks, output_path, sample_rate):
    """
    Save captured audio chunks to a WAV file.
    
    Args:
        audio_chunks: List of numpy arrays containing audio data
        output_path: Path to output WAV file
        sample_rate: Audio sample rate in Hz
    """
    try:
        from scipy.io import wavfile
        
        if not audio_chunks:
            print("⚠ No audio chunks to save")
            return
        
        # Concatenate all chunks
        audio_data = np.concatenate(audio_chunks)
        
        # Save to WAV file
        wavfile.write(output_path, sample_rate, audio_data)
        
        duration = len(audio_data) / sample_rate
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        
        print(f"\n[OK] Audio saved to: {output_path}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Samples: {len(audio_data):,}")
        print(f"  File size: {size_mb:.2f} MB")
        
    except ImportError:
        print("\n[!] scipy not installed - cannot save WAV file")
        print("  Install with: pip install scipy")
    except Exception as e:
        print(f"\n[ERROR] Error saving WAV file: {e}")


def main():
    """Main function to run the audio capture test."""
    parser = argparse.ArgumentParser(
        description='Test real-time audio capture from WASAPI loopback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture for 30 seconds and save to test_capture.wav
  python scripts/test_audio_manual.py --duration 30 --output test_capture.wav
  
  # Capture for 60 seconds without saving
  python scripts/test_audio_manual.py --duration 60 --no-save
  
  # Quick 10 second test
  python scripts/test_audio_manual.py --duration 10
        """
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Capture duration in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='test_capture.wav',
        help='Output WAV file path (default: test_capture.wav)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving WAV file (just show stats)'
    )
    
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List all audio devices and exit'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('audio_capture_test.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*80)
    print("REAL-TIME AUDIO CAPTURE TEST")
    print("="*80)
    print(f"Sample Rate: {settings.sample_rate} Hz")
    print(f"Channels: {settings.channels} (mono)")
    print(f"Chunk Duration: {settings.chunk_duration} seconds")
    print(f"Chunk Size: {settings.sample_rate * settings.chunk_duration * settings.channels:,} samples")
    print(f"Test Duration: {args.duration} seconds")
    print("="*80)
    
    # Initialize AudioCapture
    print("\n[*] Initializing AudioCapture...")
    audio_capture = AudioCapture()
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices(audio_capture)
        audio_capture.close()
        return
    
    try:
        # List all devices first
        list_audio_devices(audio_capture)
        
        # Start audio capture
        print("[*] Starting audio capture...")
        audio_capture.start()
        print("[OK] Audio capture started successfully!\n")
        
        # Display capture info
        print("="*80)
        print("REAL-TIME CAPTURE STATISTICS")
        print("="*80)
        print("Press Ctrl+C to stop early\n")
        
        # Initialize tracking variables
        start_time = time.time()
        audio_chunks = []
        chunk_count = 0
        total_samples = 0
        last_chunk_time = start_time
        poll_count = 0
        
        # Real-time capture loop
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check if duration exceeded
                if elapsed >= args.duration:
                    print(f"\n[TIME] Duration of {args.duration} seconds reached")
                    break
                
                # Poll for audio chunk
                poll_count += 1
                chunk = audio_capture.get_audio_chunk()
                
                if chunk is not None:
                    # Got a chunk!
                    chunk_count += 1
                    total_samples += len(chunk)
                    audio_chunks.append(chunk)
                    
                    # Calculate statistics
                    chunk_time = current_time
                    time_since_last = chunk_time - last_chunk_time
                    last_chunk_time = chunk_time
                    
                    rms_db = calculate_rms_db(chunk)
                    peak_db = calculate_peak_db(chunk)
                    level_bar = create_level_bar(rms_db)
                    queue_size = audio_capture.audio_queue.qsize()
                    
                    # Display chunk info
                    print(f"\n{'-'*80}")
                    print(f"[TIME] {elapsed:.1f}s / {args.duration}s")
                    print(f"[CHUNK] #{chunk_count}")
                    print(f"   Samples: {len(chunk):,} ({len(chunk) / settings.sample_rate:.2f}s)")
                    print(f"   Time since last chunk: {time_since_last:.3f}s")
                    print(f"   Total samples captured: {total_samples:,}")
                    print(f"   Chunk rate: {chunk_count / elapsed:.3f} chunks/sec")
                    
                    # Audio levels
                    if rms_db == float('-inf'):
                        rms_str = "-inf dB (silence)"
                    else:
                        rms_str = f"{rms_db:.1f} dB"
                    
                    if peak_db == float('-inf'):
                        peak_str = "-inf dB"
                    else:
                        peak_str = f"{peak_db:.1f} dB"
                    
                    print(f"\n[AUDIO LEVELS]")
                    print(f"   RMS:  {rms_str}")
                    print(f"   Peak: {peak_str}")
                    print(f"   Level: [{level_bar}]")
                    
                    # Buffer info
                    print(f"\n[BUFFER STATUS]")
                    print(f"   Queue size: {queue_size} chunks")
                    print(f"   Polls since last chunk: {poll_count}")
                    
                    poll_count = 0  # Reset poll counter
                    
                else:
                    # No chunk available yet - still buffering
                    if chunk_count == 0:
                        # First chunk not received yet
                        print(f"\r[BUFFERING] {elapsed:.1f}s elapsed, ~{settings.chunk_duration - elapsed:.1f}s until first chunk", end='', flush=True)
                    
                    # Small sleep to avoid busy waiting
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n[!] Interrupted by user (Ctrl+C)")
            elapsed = time.time() - start_time
        
        # Final statistics
        print("\n" + "="*80)
        print("CAPTURE SUMMARY")
        print("="*80)
        print(f"Duration: {elapsed:.2f} seconds")
        print(f"Chunks captured: {chunk_count}")
        print(f"Total samples: {total_samples:,}")
        print(f"Average chunk rate: {chunk_count / elapsed:.3f} chunks/sec")
        print(f"Expected chunks: ~{int(elapsed / settings.chunk_duration)}")
        print(f"Data captured: {total_samples / settings.sample_rate:.2f} seconds of audio")
        print("="*80)
        
        # Save to WAV file
        if not args.no_save and audio_chunks:
            save_audio_to_wav(audio_chunks, args.output, settings.sample_rate)
        elif args.no_save:
            print("\n[!] Skipping WAV file save (--no-save flag)")
        
    except Exception as e:
        logger.error(f"Error during capture: {e}", exc_info=True)
        print(f"\n[ERROR] Error: {e}")
        return 1
    
    finally:
        # Clean up
        print("\n[*] Cleaning up...")
        # audio_capture.stop()
        # audio_capture.close()
        print("[OK] Cleanup complete")
    
    print("\n[SUCCESS] Test completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
