### Improvenments

Need to use checkpointing for long training runs.
    - This will enable the usage of pausing and restarting long training runs
    - https://pytorch-lightning.readthedocs.io/en/1.6.2/common/checkpointing.html

### Attempted

- Efficientnetv2, DPT
- Losses: Tversky, LogCoshDice
- Downsampling Interpolation Methods
- Removing Islands
- Openmmlab (upernet, swin)
- Deepsupervision

### Positives

- Created a diverse ensemble using different backbones
- Implemented novel ideas (deepsupervision, custom model architectures, etc)
- Iterated quickly with small-scale version of imgs/models, and scaled up at the end
- First segmentation competition!