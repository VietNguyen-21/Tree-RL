"""
scripts/download_voc.py
-----------------------
Download PASCAL VOC 2007 + 2012 — chạy được trên Windows, Linux, Mac.

Dùng:
    python scripts/download_voc.py
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path


URLS = {
    "VOC2007_trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    "VOC2007_test":     "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    "VOC2012_trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
}

DATA_DIR = Path("./VOCdevkit")


def download_with_progress(url: str, dest: Path):
    """Download file với progress bar đơn giản."""
    filename = url.split("/")[-1]
    dest_file = dest / filename

    if dest_file.exists():
        print(f"  [Bỏ qua] {filename} đã tồn tại.")
        return dest_file

    print(f"  Đang download: {filename}")
    print(f"  URL: {url}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = count * block_size * 100 // total_size
            mb_done = count * block_size / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            sys.stdout.write(
                f"\r  {percent}% ({mb_done:.0f}/{mb_total:.0f} MB)   "
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_file, reporthook)
        print()  # newline sau progress
        print(f"  Saved: {dest_file}")
    except Exception as e:
        print(f"\n  Lỗi download: {e}")
        if dest_file.exists():
            dest_file.unlink()
        raise

    return dest_file


def extract_tar(tar_path: Path, dest: Path):
    """Giải nén file .tar vào dest."""
    print(f"  Đang giải nén: {tar_path.name} ...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest)
    print(f"  Giải nén xong -> {dest}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_dir = DATA_DIR / "_tars"
    tar_dir.mkdir(exist_ok=True)

    for name, url in URLS.items():
        year = "2007" if "2007" in name else "2012"
        voc_dir = DATA_DIR / f"VOC{year}"

        print(f"\n=== {name} ===")

        # Kiểm tra đã giải nén chưa
        if voc_dir.exists() and any(voc_dir.iterdir()):
            # Kiểm tra thêm split test riêng cho VOC2007
            if "test" in name:
                test_dir = voc_dir / "ImageSets" / "Main" / "test.txt"
                if test_dir.exists():
                    print(f"  [Bỏ qua] {voc_dir} đã có đủ dữ liệu.")
                    continue
            else:
                print(f"  [Bỏ qua] {voc_dir} đã tồn tại.")
                continue

        # Download
        tar_path = download_with_progress(url, tar_dir)

        # Giải nén vào VOCdevkit/
        extract_tar(tar_path, DATA_DIR)

    print("\n=== Hoàn thành! Cấu trúc thư mục ===")
    for item in sorted(DATA_DIR.iterdir()):
        if item.is_dir() and item.name != "_tars":
            print(f"  {item}/")
            for sub in sorted(item.iterdir()):
                print(f"    {sub.name}/")

    print("\nBước tiếp theo:")
    print("  python scripts/extract_features.py --year 2007 --split trainval")
    print("  python scripts/extract_features.py --year 2012 --split trainval")
    print("  python scripts/extract_features.py --year 2007 --split test")


if __name__ == "__main__":
    main()
