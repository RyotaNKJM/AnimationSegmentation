# Video Instance Segmentaion for Animation
Video instance segmentaion for animation with pretrained models, instance segmentation for anime characters image and semi-supervised video object segmentation. 

This program uses [Yet-Another-Anime-Segmenter](https://github.com/zymk9/Yet-Another-Anime-Segmenter) as instance segmentation and [XMem](https://github.com/hkchengrex/XMem) as video object segmentation. The result of YAAS is used as XMem input.

## Installation
Both Yet-Another-Anime-Segmenter and XMem are required. Please refer to the official guide from [Yet-Another-Anime-Segmenter](https://github.com/zymk9/Yet-Another-Anime-Segmenter#installation) and [XMem](https://github.com/hkchengrex/XMem/blob/main/docs/GETTING_STARTED.md) is provided.

Structure projects like this:
```bash
├─this project
├─Yet-Another-Anime-Segmenter
├─XMem
```

## Inference
1. Download pretrained models and the corresponding config file. [Yet-Another-Anime-Segmenter](https://github.com/zymk9/Yet-Another-Anime-Segmenter#inference), [XMem](https://github.com/hkchengrex/XMem/blob/main/docs/INFERENCE.md)

2. Run inference with 
  ```bash
  python3 main.py \ 
    --config-file path/to/config.yaml/of/Yet-Another-Anime-Segmenter \
    --model path/to/pretrained/model/of/XMem \
    --input path/to/frame_images \
    --output path/to/output/directory \
    --opts MODEL.WEIGHTS path/to/pretrained/model/of/Yet-Another-Anime-Segmenter
  ```

