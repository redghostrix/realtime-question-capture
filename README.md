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

### 4. Verify installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
- PyTorch: 2.7.0
- CUDA available: True

### 5. Start llama-server
Ensure llama-server is running on `http://localhost:8080` before starting the application.

## Configuration

Edit `src/config.py` to customize:
- `WHISPER_MODEL`: Whisper model size (tiny, base, small, medium, large)
- `LLAMA_SERVER_URL`: LLM server endpoint
- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- `CHUNK_DURATION`: Audio processing chunk duration (default: 5 seconds)

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

```bash
conda activate question-capture
python src/main.py
```

Press `Ctrl+C` to stop the application.

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
- **CUDA out of memory**: Use smaller model (`--model tiny`) or reduce `CHUNK_DURATION` in `src/config.py`
- **Slow transcription**: Verify GPU is being used, close other GPU applications
- **Inaccurate transcriptions**: Use larger model (`--model small` or `--model medium`)
- See [TESTING_TRANSCRIPTION.md](TESTING_TRANSCRIPTION.md) for detailed troubleshooting

**llama-server connection failed:**
- Verify llama-server is running: `curl http://localhost:8080/health`
- Check `LLAMA_SERVER_URL` in `src/config.py`

## License

[Add license information]
