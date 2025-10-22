from pyannote.audio import Pipeline
from huggingface_hub import login
import os

# login(token = "")
# login()
# token = os.getenv("HUGGINGFACE_TOKEN")
# print("Token:", token[:10], "...")

# Try loading (with the token as 2nd positional argument)
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token= "tokenhere")
# pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
# pipe = Pipeline.from_pretrained(
#   "pyannote/speaker-diarization-3.1",
#   use_auth_token="")
print("Pipeline loaded successfully ✅")


# https://huggingface.co/pyannote/speaker-diarization-community-1