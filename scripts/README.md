# Test Scripts

This directory contains scripts for testing and verifying the audio capture functionality.

## test_audio_manual.py

Manual verification script for testing real-time audio capture from WASAPI loopback.

### Prerequisites

1. Activate the conda environment:
   ```bash
   conda activate question-capture
   ```

2. Ensure scipy is installed:
   ```bash
   pip install scipy
   ```

3. Make sure audio is playing on your system (YouTube, Spotify, etc.)

### Usage

**Basic test (30 seconds):**
```bash
python scripts/test_audio_manual.py
```

**Custom duration:**
```bash
python scripts/test_audio_manual.py --duration 60
```

**Custom output file:**
```bash
python scripts/test_audio_manual.py --output my_test.wav
```

**Just show stats, don't save WAV:**
```bash
python scripts/test_audio_manual.py --no-save
```

**List all audio devices:**
```bash
python scripts/test_audio_manual.py --list-devices
```

### What to Expect

1. **Device Listing**: Shows all audio devices and identifies WASAPI loopback device
2. **Initial Buffering**: First ~5 seconds will show "Buffering..." message
3. **First Chunk**: After ~5 seconds, first audio chunk arrives with statistics
4. **Real-Time Updates**: Every ~5 seconds, new chunk appears with:
   - Audio levels (RMS and Peak in dB)
   - Visual level meter
   - Buffer status
   - Timing information
5. **WAV File**: Saved at end for playback verification

### Testing Steps

1. **Start with audio playing** (e.g., YouTube video)
2. **Run the script** and observe initial buffering
3. **Watch levels change** as audio plays
4. **Pause the audio** and verify levels drop to silence
5. **Resume audio** and verify levels increase
6. **Let it complete** the full duration
7. **Play back the WAV file** to verify captured audio quality

### Troubleshooting

**"No WASAPI loopback device found"**
- Ensure you're on Windows
- Check if "Stereo Mix" is enabled in sound settings
- Verify audio is working on your system

**No chunks captured**
- Ensure audio is actually playing
- Check that the default output device is playing sound
- Verify the loopback device was correctly detected

**Audio level always shows silence**
- Make sure audio is playing through the default output device
- Try playing audio from multiple sources
- Check Windows sound mixer to ensure apps are not muted

### Log Files

The script creates `audio_capture_test.log` with detailed debug information.
