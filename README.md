# SMP-Attack
[**\[ICCV 2025\] "SMP-Attack: Boosting the Transferability of Feature Importance-based Adversarial Attack with Semantics-aware Multi-granularity Patchout", Wen Yang, Guodong Liu, Di Ming*.**]([https://github.com/advml-group](https://github.com/AdvML-Group/SMP-Attack)) 

## Introduction

Transfer-based attacks pose a significant security threat to deep neural networks (DNNs), due to their strong performance on unseen models in real-world black-box scenarios.Building on this, feature importance-based attacks further improve the transferability of adversarial examples by effectively suppressing model-specific feature patterns. However, existing methods primarily focus on single-granularity patch and single-stage training, leading to suboptimal solutions. To address these limitations, we propose a general multi-stage optimization framework based on Semantics-aware Multi-granularity Patchout, dubbed as SMP-Attack. Compared to the non-deformable/regular patch definition, we incorporate multi-granularity into the generation process of deformable/irregular patches, thereby enhancing the quality of the computed aggregate gradient. In contrast to conventional joint optimization of multi-layer losses, we introduce an effective multi-stage training strategy that systematically explores significant model-agnostic features from shallow to intermediate layers. Employing the ImageNet dataset, we conduct extensive experiments on undefended/defended CNNs and ViTs, which unequivocally demonstrate the superior performance of our proposed SMP-Attack over current state-of-the-art methods in black-box scenarios. Furthermore, we assess the compatibility of our multi-stage optimization, which supersedes single-stage training employed in existing feature-based methods, culminating in substantial performance improvement.

![](D:\goole\gitproject\SMP-Attack\show_image\Home.png)

# Getting Started

## Dependencies

- Python 3.6.0
- Keras 2.2.4
- Tensorflow (GPU) 1.15.1
- Numpy 1.19.5
- Pillow 8.3.2

## Usage Instructions

#### Models and Datasets

1. Download the pretrained checkpoints into `./models_tf` before running the code.

- [Normlly trained models]( https://github.com/tensorflow/models/tree/master/research/slim)
- [Adversarial trained models]( https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)
- [Vision transformers models](https://github.com/rwightman/pytorch-image-models)

2. We conduct experiments on the ImageNet-compatible dataset, comprising 1000 images, used in the NIPS 2017 adversarial competition. The image path is `./dataset/images`.

#### Running attacks

Simply execute `./runSMP.sh` in the terminal to launch the SMP-Attack.

#### Evaluate the success rate

- Normlly trained models and Adversarial trained models.

  `python verify_cnns.py`

- Vision transformers models.

  `python verify_vits.py`

## Acknowledgments

Code refers to [FIA](https://github.com/hcguoO0/FIA), [RPA](https://github.com/alwaysfoggy/RPA).

## Citation

If you find this work is useful in your research, please cite our paper:

```
@InProceedings{Yang_2025_ICCV,
    author    = {Yang, Wen and Liu, Guodong and Ming, Di},
    title     = {SMP-Attack: Boosting the Transferability of Feature Importance-based Adversarial Attack with Semantics-aware Multi-granularity Patchout},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {4444-4454}
}
```