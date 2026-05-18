#!/bin/bash
# scripts/download_voc.sh
# Download PASCAL VOC 2007 và 2012

set -e

DATA_DIR="./VOCdevkit"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Download PASCAL VOC 2007 ==="
if [ ! -d "VOC2007" ]; then
    wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    tar xf VOCtrainval_06-Nov-2007.tar
    tar xf VOCtest_06-Nov-2007.tar
    echo "VOC2007 done."
else
    echo "VOC2007 đã tồn tại, bỏ qua."
fi

echo "=== Download PASCAL VOC 2012 ==="
if [ ! -d "VOC2012" ]; then
    wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar xf VOCtrainval_11-May-2012.tar
    echo "VOC2012 done."
else
    echo "VOC2012 đã tồn tại, bỏ qua."
fi

cd ..
echo ""
echo "=== Hoàn thành! Cấu trúc thư mục ==="
ls VOCdevkit/
