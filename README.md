# SMP-Attack
[**\[ICCV 2025\] "SMP-Attack: Boosting the Transferability of Feature Importance-based Adversarial Attack with Semantics-aware Multi-granularity Patchout", Wen Yang, Guodong Liu, Di Ming*.**]([https://github.com/advml-group](https://github.com/AdvML-Group/SMP-Attack)) 

## Quick Start

### Prepare pretrained models

- [Normlly trained models]( https://github.com/tensorflow/models/tree/master/research/slim)
- [Adversarial trained models]( https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)
- [Visual transformers models](https://github.com/rwightman/pytorch-image-models)

### Running attacks

`./runSMP.sh`

### Evaluate the success rate

- Normlly trained models nad Adversarial trained models

`python verify_cnns.py`

- Visual transformers models

`python verify_vits.py`

## Acknowledgments

Code refers to [FIA](https://github.com/hcguoO0/FIA), [RPA](https://github.com/alwaysfoggy/RPA)

## Citing this work

```
@inproceedings{SMP,
  title={SMP-Attack: Boosting the Transferability of Feature Importance-based Adversarial Attack with Semantics-aware Multi-granularity Patchout},
  author={Wen Yang and Guodong Liu and Di Ming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025},
  pages={},
}
```