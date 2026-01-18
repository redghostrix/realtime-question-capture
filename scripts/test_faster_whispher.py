import pathlib
from faster_whisper import WhisperModel

FILE_PARENT_DIR = pathlib.Path(__file__).parent
MODEL_PATH = FILE_PARENT_DIR / '..' / 'models' / 'whisper-fast'

# Use CPU to avoid CUDA issues
print("loading a model", MODEL_PATH)
model = WhisperModel(
    str(MODEL_PATH),
    device="cuda", 
)
print('model', model)

segments, info = model.transcribe("audio.mp3")
print('hello', segments, info)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
