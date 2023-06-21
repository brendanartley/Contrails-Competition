# Notes

Code for ICRGW.

### Ideas

- DepthwiseConv2D layer using the the two neigbhouring frames

### EfficientNet Sizes

Model           Img Size   Params
EfficientNetB0	224        4M
EfficientNetB1	240        6M
EfficientNetB2	260        7M
EfficientNetB3	300        10M
EfficientNetB4	380        17M
EfficientNetB5	456        28M
EfficientNetB6	528        40M
EfficientNetB7	600        63M

### Segmentation Models Docs

https://segmentation-modelspytorch.readthedocs.io/en/latest/

### Commands

CUDA_VISIBLE_DEVICES=0 python main.py --all_folds --fast_dev_run

CUDA_VISIBLE_DEVICES=0 python main.py --epochs=7 --model_name="efficientnet-b0" --all_folds
CUDA_VISIBLE_DEVICES=1 python main.py --epochs=7 --model_name="efficientnet-b1" --all_folds
CUDA_VISIBLE_DEVICES=2 python main.py --epochs=7 --model_name="efficientnet-b2" --all_folds
CUDA_VISIBLE_DEVICES=3 python main.py --epochs=7 --model_name="efficientnet-b3" --all_folds