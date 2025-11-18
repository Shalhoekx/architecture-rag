import shutil
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)