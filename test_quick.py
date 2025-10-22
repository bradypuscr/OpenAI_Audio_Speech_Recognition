from huggingface_hub import whoami, HfApi
import os

api = HfApi()
token = os.getenv("HUGGINGFACE_TOKEN")
print("Token found:", bool(token))
print("whoami():", api.whoami(token))
