"""
setup_check.py
--------------
Chạy script này đầu tiên để kiểm tra môi trường trước khi train.
Hoạt động trên Windows, Linux, Mac.

    python setup_check.py
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check(label, ok, detail=""):
    status = "OK " if ok else "FAIL"
    mark   = "✓" if ok else "✗"
    print(f"  [{status}] {mark} {label}" + (f"  ({detail})" if detail else ""))
    return ok


def main():
    print("=" * 55)
    print(" Tree-RL Environment Check")
    print("=" * 55)

    all_ok = True

    # Python version
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 9
    all_ok &= check(f"Python {major}.{minor}", ok, "cần >= 3.9")

    # Packages
    print("\n── Packages ──────────────────────────────────────")
    packages = {
        "torch":        "PyTorch",
        "torchvision":  "torchvision",
        "numpy":        "NumPy",
        "cv2":          "OpenCV (opencv-python)",
        "PIL":          "Pillow",
        "yaml":         "PyYAML",
        "tqdm":         "tqdm",
        "scipy":        "scipy",
        "matplotlib":   "matplotlib",
        "torch.utils.tensorboard": "TensorBoard",
        "lxml":         "lxml",
    }
    for pkg, name in packages.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            ok  = True
        except ImportError:
            ver = "NOT FOUND"
            ok  = False
        all_ok &= check(name, ok, ver)

    # CUDA
    print("\n── GPU / CUDA ─────────────────────────────────────")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            check("CUDA available", True, f"{gpu_name}, {vram:.1f} GB VRAM")
            check("VRAM >= 8 GB", vram >= 7.5, f"{vram:.1f} GB")
        else:
            check("CUDA available", False, "sẽ dùng CPU (rất chậm)")
            all_ok = False
    except Exception as e:
        check("CUDA check", False, str(e))
        all_ok = False

    # Thư mục
    print("\n── Project structure ──────────────────────────────")
    required_files = [
        "configs/default.yaml",
        "data/voc_dataset.py",
        "models/feature_extractor.py",
        "models/q_network.py",
        "models/agent.py",
        "utils/actions.py",
        "utils/metrics.py",
        "utils/replay_memory.py",
        "train.py",
        "test.py",
        "scripts/download_voc.py",
        "scripts/extract_features.py",
    ]
    for f in required_files:
        exists = Path(f).exists()
        all_ok &= check(f, exists)

    # VOCdevkit
    print("\n── Data ────────────────────────────────────────────")
    voc07 = Path("VOCdevkit/VOC2007/JPEGImages")
    voc12 = Path("VOCdevkit/VOC2012/JPEGImages")
    check("VOC2007", voc07.exists(),
          f"{len(list(voc07.glob('*.jpg')))} ảnh" if voc07.exists() else "chưa download")
    check("VOC2012", voc12.exists(),
          f"{len(list(voc12.glob('*.jpg')))} ảnh" if voc12.exists() else "chưa download")

    cache_dir = Path("feature_cache")
    check("Feature cache", cache_dir.exists(),
          "chưa chạy extract_features.py" if not cache_dir.exists() else "OK")

    # Kết luận
    print("\n" + "=" * 55)
    if all_ok:
        print(" ✓ Môi trường sẵn sàng! Chạy lệnh sau để bắt đầu:")
        print()
        print("   python scripts/download_voc.py          # nếu chưa có data")
        print("   python scripts/extract_features.py --year 2007 --split trainval")
        print("   python scripts/extract_features.py --year 2012 --split trainval")
        print("   python scripts/extract_features.py --year 2007 --split test")
        print("   python train.py")
    else:
        print(" ✗ Còn thiếu package hoặc dữ liệu. Chạy:")
        print()
        print("   pip install -r requirements.txt")
        print("   python scripts/download_voc.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
