# Real-Time Question Detector

A real-time speech monitoring application that captures system audio output, transcribes speech using faster-whisper, analyzes transcription with a local LLM to detect questions, and automatically copies detected questions to the clipboard.

## Prerequisites

**Hardware:**
- CUDA-capable GPU (tested on RTX 5090)
- Audio output device

**Software:**
- Windows OS
- Conda package manager with `question-capture` environment (PyTorch 2.7.0 + CUDA 12.8)
- llama-server running externally on port 8080

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd realtime-question-capture
```

### 2. Activate existing Conda environment
```bash
conda activate question-capture
```

### 3. Install project dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to customize configuration (optional)
# The default values work out-of-the-box if llama-server is running on localhost:8080
```

### 5. Verify installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
- PyTorch: 2.7.0
- CUDA available: True

### 6. Start llama-server
Ensure llama-server is running on `http://localhost:8080` before starting the application.

## Configuration

The application uses environment-based configuration. Create a `.env` file in the project root (or copy from `.env.example`):

```bash
# Whisper Configuration
WHISPER_MODEL=base                    # Options: tiny, base, small, medium, large

# LLM Server Configuration
LLAMA_SERVER_URL=http://localhost:8080/v1/chat/completions
QUESTION_EXTRACTOR_MODEL_NAME=llama-3.3-70b-versatile
QUESTION_EXTRACTOR_MAX_RETRIES=3     # 0-10
QUESTION_EXTRACTOR_TIMEOUT=30        # seconds (1-300)

# Audio Capture Configuration
SAMPLE_RATE=16000                    # Hz (8000-48000)
CHUNK_DURATION=5                     # seconds (1-60)
CHANNELS=1                           # 1=mono, 2=stereo

# Logging Configuration
LOG_LEVEL=INFO                       # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

All settings are optional and will use the defaults shown above if not specified. The configuration provides type validation and will report errors if invalid values are provided.

## Architecture

```
System Audio Output → Audio Loopback → faster-whisper → llama-server → Clipboard
```

## Components

1. **Audio Capture**: Windows WASAPI loopback via pyaudiowpatch
2. **Transcription**: faster-whisper with GPU acceleration
3. **Question Detection**: DSPy with llama-server
4. **Clipboard**: pyperclip for automatic copying

## Usage

### CLI Mode

Run the application in command-line mode:

```bash
conda activate question-capture
python src/main.py
```

Press `Ctrl+C` to stop the application.

### GUI Mode

Launch the desktop monitoring application:

```bash
conda activate question-capture
python src/main_gui.py
```

**Features:**
- Real-time monitoring of transcription and question extraction
- Start/Stop/Pause controls
- Timing metrics for each pipeline stage (Audio, Transcription, Extraction)
- Dark mode interface
- Copy individual questions to clipboard
- Live log display with color-coding (INFO/WARNING/ERROR)

**Interface Layout:**
- **Control Panel** (Left): Start/Stop/Pause buttons and configuration display
- **Transcription Panel** (Top Right): Shows raw transcribed text with timestamps
- **Questions Panel** (Middle Right): Shows extracted questions with timestamps
- **Logs Panel** (Bottom Right): Shows application logs with color-coding
- **Status Bar** (Bottom): Displays timing metrics for each pipeline stage

## Testing

The project includes comprehensive testing scripts to verify each component:

### Test Audio Capture

Verify WASAPI loopback audio capture is working:

```bash
python scripts/test_audio_manual.py --duration 30 --output test_capture.wav
```

See [TESTING.md](TESTING.md) for detailed audio capture testing instructions.

### Test Transcription

Verify Whisper transcription with GPU acceleration:

```bash
# Check GPU availability
python scripts/test_transcription_manual.py --gpu-info

# Test with WAV file
python scripts/test_transcription_manual.py --wav-file test_capture.wav

# Test with live audio capture
python scripts/test_transcription_manual.py --live --duration 30
```

See [TESTING_TRANSCRIPTION.md](TESTING_TRANSCRIPTION.md) for detailed transcription testing instructions.

### Integration Test

Test the complete audio capture → transcription pipeline:

```bash
# 1. Capture audio while playing something (YouTube, podcast, etc.)
python scripts/test_audio_manual.py --duration 30 --output integration_test.wav

# 2. Transcribe the captured audio
python scripts/test_transcription_manual.py --wav-file integration_test.wav

# 3. Test live capture + transcription
python scripts/test_transcription_manual.py --live --duration 30
```

## Troubleshooting

**CUDA not available:**
- Verify CUDA 12.8 is installed: `nvidia-smi`
- Reinstall PyTorch with CUDA support
- Check GPU availability: `python scripts/test_transcription_manual.py --gpu-info`

**Audio capture issues:**
- Ensure audio is playing on the system
- Check Windows audio settings for loopback device
- Run diagnostic: `python scripts/test_audio_manual.py --list-devices`
- See [TESTING.md](TESTING.md) for detailed troubleshooting

**Transcription issues:**
- **CUDA out of memory**: Set `WHISPER_MODEL=tiny` in `.env` or reduce `CHUNK_DURATION`
- **Slow transcription**: Verify GPU is being used, close other GPU applications
- **Inaccurate transcriptions**: Set `WHISPER_MODEL=small` or `WHISPER_MODEL=medium` in `.env`
- See [TESTING_TRANSCRIPTION.md](TESTING_TRANSCRIPTION.md) for detailed troubleshooting

**llama-server connection failed:**
- Verify llama-server is running: `curl http://localhost:8080/health`
- Check `LLAMA_SERVER_URL` in `.env` file

## License

[Add license information]
