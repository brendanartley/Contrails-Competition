# Notes

Code for ICRGW.

Test w/ only contrails data (1/2 the size), and train final model on all the data.

### TODO

- Save preds directory does not work yet (writes to main preds dir)
- When save_preds=True, preds are written on every validation step...

### Ideas

- reduce val 

- Write a validate script that takes a file path of predictions and scores against ground truth
- DepthwiseConv2D layer using the two neigbhouring frames
- Write predictions to a file and look at correlation between model predictions.
- Fix KFold w/ Sweeps Error
- Read into the multi-frame model approach in the paper (Resnet3D?)

- Write a script that computes score / correlation of model predictions, and assign weights for weighted ensemble?
    - Make this portable to other competitions as well.

### Preprint Notes / Findings

- Using 3 frames before yields the best results for resnet3D
    - 0B-0A = 71.4
    - 1B-0A = 71.7
    - 3B-0A = 72.7
    - 4B-0A = 72.0

- Best per-pixel threshold is 0.4 (optimize this by writing prediction to disk and comparing score across thresholds)

### EfficientNet Sizes

| Model                | Img Size | Params |
|----------------------|----------|--------|
| EfficientNetB0       | 224      | 4M     |
| EfficientNetB1       | 240      | 6M     |
| EfficientNetB2       | 260      | 7M     |
| EfficientNetB3       | 300      | 10M    |
| EfficientNetB4       | 380      | 17M    |
| EfficientNetB5       | 456      | 28M    |
| EfficientNetB6       | 528      | 40M    |
| EfficientNetB7       | 600      | 63M    |
| efficientnetv2_rw_t  | -        | 13M    |
| efficientnetv2_rw_s  | -        | 23M    |
| efficientnetv2_rw_m  | -        | 53M    |
| efficientnetv2_s     | -        | 23M    |
| efficientnetv2_m     | -        | 53M    |
| mit_b1               | -        | 13M    |
| mit_b2               | -        | 24M    |
| mit_b3               | -        | 44M    |
| mit_b4               | -        | 60M    |
| mit_b5               | -        | 81M    |



### Links

Segmentation Models Documentation: https://segmentation-modelspytorch.readthedocs.io/en/latest/


### Commands

CUDA_VISIBLE_DEVICES=0 python main.py --save_preds --fast_dev_run
CUDA_VISIBLE_DEVICES=1 python main.py --save_preds

CUDA_VISIBLE_DEVICES=0 python main.py --model_type="timm" --model_name="efficientnetv2_rw_s.ra2_in1k" --all_folds=True

CUDA_VISIBLE_DEVICES=0 python main.py --model_weights="/data/bartley/gpu_test/models/segmentation/test_weights.pt" --save_preds --fast_dev_run
