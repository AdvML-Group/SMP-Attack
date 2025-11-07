## SMP-Attack

[[Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_SMP-Attack_Boosting_the_Transferability_of_Feature_Importance-based_Adversarial_Attack_with_ICCV_2025_paper.pdf)] · [[Supplementary](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Yang_SMP-Attack_Boosting_the_ICCV_2025_supplemental.pdf)] · [[Poster](https://iccv.thecvf.com/virtual/2025/poster/635)] · [[Video](https://youtu.be/5p4LxrBjxV0)]

Official TensorFlow implementation of [**SMP-Attack: Boosting the Transferability of Feature Importance-based Adversarial Attack with Semantics-aware Multi-granularity Patchout**](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_SMP-Attack_Boosting_the_Transferability_of_Feature_Importance-based_Adversarial_Attack_with_ICCV_2025_paper.html) (ICCV 2025) by Wen Yang, Guodong Liu, and Di Ming.

## Overview

Transfer-based adversarial attacks pose a substantial security risk to deep neural networks (DNNs) because they remain effective on unseen, black-box models. Feature-importance attacks further improve transferability by suppressing model-specific feature patterns, yet prior work typically relies on a single patch granularity and single-stage training. SMP-Attack addresses these limitations with a semantics-aware, multi-granularity patchout strategy coupled with a multi-stage optimization pipeline. By assembling deformable, irregular patches and progressively mining model-agnostic features from shallow to intermediate layers, SMP-Attack delivers state-of-the-art black-box performance across both CNN and ViT architectures on ImageNet-scale benchmarks.

![SMP-Attack overview](./show_image/Home.png)

## Getting Started

## Dependencies

- Python 3.6.0
- Keras 2.2.4
- Tensorflow (GPU) 1.15.1
- Numpy 1.19.5
- Pillow 8.3.2

## Usage Instructions

### Models and datasets

1. Download the pretrained checkpoints into `./models_tf` before running the code.
   - [Normally trained models](https://github.com/tensorflow/models/tree/master/research/slim)
   - [Adversarially trained models](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)
   - [Vision transformer models](https://github.com/rwightman/pytorch-image-models)

2. Experiments are conducted on the ImageNet-compatible subset (1,000 images) from the NIPS 2017 adversarial competition. Place the images under `./dataset/images`.

### Running attacks

- `attack_SMP.py`: TensorFlow implementation of the SMP-Attack pipeline (main entry script).
- `cpp/SLICSP.cpp`: C++ implementation of Semantics-aware Multi-granularity PatchOut. Compile it into a shared library (`SLICSP.so`) with `g++ SLICSP.cpp -fPIC -shared -o SLICSP.so`, and load it from `attack_SMP.py` for accelerated masking.
- `runSMP.sh`: Shell script containing example configurations for single-stage and multi-stage SMP-Attack runs. Modify it to reproduce paper experiments or customize hyperparameters.

### Evaluating success rate

- CNNs (standard and adversarially trained):

  ```bash
  python verify_cnns.py
  ```

- Vision transformer models:

  ```bash
  python verify_vits.py
  ```

## Acknowledgments

Code builds upon [FIA](https://github.com/hcguoO0/FIA) and [RPA](https://github.com/alwaysfoggy/RPA). We sincerely thank the authors for sharing their work.

## Citation

If you find this repository useful in your research, please cite:

```
@InProceedings{SMP_Attack,
    author    = {Yang, Wen and Liu, Guodong and Ming, Di},
    title     = {SMP-Attack: Boosting the Transferability of Feature Importance-based Adversarial Attack with Semantics-aware Multi-granularity Patchout},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {4444--4454}
}

```

## Contact

[Wen Yang](https://github.com/Winwina8/Winwina8.github.io/): [winwina8@126.com](mailto:winwina8@126.com)

[Di Ming](https://midasdming.github.io/): [diming@cqut.edu.cn](mailto:diming@cqut.edu.cn)











