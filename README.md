## load COCO 1.0 dataset

```sh
python src/train.py \
  --data-dir /path/to/images \
  --annotation-file /path/to/annotations.json \
  --batch-size 32 \
  --epochs 100 \
  --lr 0.001 \
  --save-dir ./checkpoints
```

## 基本的な使用方法

`python tag_annotator.py /path/to/images`

## カスタム出力ファイル名を指定

`python tag_annotator.py /path/to/images -o custom_annotations.json`

## データベースを初期化して実行

`python tag_annotator.py /path/to/images --reset`
