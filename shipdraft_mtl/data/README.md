# DraftFormer dataset formats

DraftFormer supports dataset formats, not dataset names. Every sample can contain detection and
segmentation targets. Detection category IDs are contiguous from zero; segmentation IDs follow
the detection IDs.

## LabelMe JSON

Use `name: JsonMultitaskDataset` and provide only `data_root`. The loader scans all splits to infer
one stable class mapping. LabelMe `rectangle` shapes are detection targets and `polygon` shapes are
segmentation targets.

```text
<data_root>/
  train/
    sample.jpg
    sample.json
  val/
  test/
```

```yaml
dataset:
  name: JsonMultitaskDataset
  data_root: E:\data\PingLuRiver\dataset\ShipDraft
  split: train
```

The optional top-level LabelMe field `draft depth` is exposed as `draft_depth` in each sample.

## YOLO TXT

Use `name: YoloMultitaskDataset`. In addition to `data_root`, declare the source ID-to-name mapping
for both tasks. A list maps its index to the class name; an integer-keyed mapping is also accepted.

```text
<data_root>/
  images/<split>/*.{jpg,png,...}
  labels/detection/<split>/*.txt
  labels/segmentation/<split>/*.txt
```

Detection rows use `class_id cx cy width height`. Segmentation rows use
`class_id x1 y1 x2 y2 ...`. All coordinates are normalized to `[0, 1]`.

```yaml
dataset:
  name: YoloMultitaskDataset
  data_root: /datasets/ship_draft_yolo
  class_names:
    detection: {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "8", 8: "M"}
    segmentation: {0: water}
```

Train and Eval metadata must be identical. During dataloader construction the metadata is written
to `Data`, `Architecture.Head`, and `Metric.det_class_names`, so the model dimensions cannot drift
from the dataset mapping.
