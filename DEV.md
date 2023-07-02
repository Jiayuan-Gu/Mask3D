# Point-SAM

## Installation

Please follow [Mask3D README](./README.md) to install dependencies.

## Data

### ScanNet

```bash
# Download ScanNet
wget https://kaldir.vc.in.tum.de/scannet/download-scannet.py
export SCANNET_ROOT=/data/scannet
# scannetv2-labels.combined.tsv
python download-scannet.py -o $SCANNET_ROOT --label_map
# description
python download-scannet.py -o $SCANNET_ROOT --type '.txt'
# point cloud
python download-scannet.py -o $SCANNET_ROOT --type '_vh_clean_2.ply'
# segmentation labels
python download-scannet.py -o $SCANNET_ROOT --type '_vh_clean_2.labels.ply'
# instance (info) labels
python download-scannet.py -o $SCANNET_ROOT --type '.aggregation.json'
# instance (mask) labels
python download-scannet.py -o $SCANNET_ROOT --type '_vh_clean_2.0.010000.segs.json'
```

```bash
python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="/data/scannet" \
--save_dir="data/processed/scannet" \
--git_repo="third_party/ScanNet" \
--scannet200=False

python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="/data/scannet" \
--save_dir="data/processed/scannet200" \
--git_repo="third_party/ScanNet" \
--scannet200=True
```

## Mask3D

<https://github.com/JonasSchult/Mask3D/issues/40>
<https://github.com/JonasSchult/Mask3D/issues/103>