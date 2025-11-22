import os
from pathlib import Path

def check_download_progress(model_name):
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    
    if model_dir.exists():
        snapshots = list(model_dir.glob("snapshots/*"))
        if snapshots:
            latest_snapshot = snapshots[0]
            files = list(latest_snapshot.glob("*"))
            print(f"📂 Файлов в модели: {len(files)}")
            for file in files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name}: {size_mb:.1f} MB")
    else:
        print("Модель еще не скачана")

# Проверить прогресс
check_download_progress("BAAI/bge-base-en-v1.5")