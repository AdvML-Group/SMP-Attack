# SMP-Attack
[**\[ICCV 2025\] "SMP-Attack: Boosting the Transferability of Feature Importance-based Adversarial Attack with Semantics-aware Multi-granularity Patchout", Wen Yang, Guodong Liu, Di Ming*.**]([https://github.com/advml-group](https://github.com/AdvML-Group/SMP-Attack)) 

## Requirements

- Python 3.6.0
- Keras 2.2.4
- Tensorflow (GPU) 1.15.1
- Numpy 1.19.5
- Pillow 8.3.2

## Experiments

### Prepare pretrained models

- [Normlly trained models]( https://github.com/tensorflow/models/tree/master/research/slim)
- [Adversarial trained models]( https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)
- [Visual transformers models](https://github.com/rwightman/pytorch-image-models)

### Running attacks

​	`./runSMP.sh` : the implementation for SMP attack.

### Evaluate the success rate

- Normlly trained models nad Adversarial trained models

  `python verify_cnns.py`

- Visual transformers models

  `python verify_vits.py`

## Acknowledgments

​	Code refers to [FIA](https://github.com/hcguoO0/FIA), [RPA](https://github.com/alwaysfoggy/RPA)

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