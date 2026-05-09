import os
import urllib.request

CHECKPOINT_URL = "https://huggingface.co/shekxyy/cdvae_checkpoint_epoch_110.pt/resolve/main/cdvae_checkpoint_epoch_110.pt"
CHECKPOINT_PATH = "cdvae_checkpoint_epoch_110.pt"

if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading checkpoint from Hugging Face...")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    print("Download complete.")
else:
    print("Checkpoint already exists, skipping download.")