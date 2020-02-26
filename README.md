# dense-fusion

## Prerequisite

Download [YCB Video Dataset](https://drive.google.com/uc?id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi) and extract it.

Also set,

```bash
export YCB_VIDEO_DATASET_PATH=<YOUR YCB_Video_Dataset PATH>
```

## Installation

```bash
pip install dense-fusion
```

## Sample

```bash
cd examples
python eval_ycb.py
```

## ROS Interface

Providing ROS Interface [https://github.com/iory/dense-fusion-ros](https://github.com/iory/dense-fusion-ros).
If you are ROS User, Please try it.

## Features

- [x] Inference
- [x] ROS Interface ([https://github.com/iory/dense-fusion-ros](https://github.com/iory/dense-fusion-ros))
- [ ] Training YCB Video Dataset
- [ ] Training custom Dataset
