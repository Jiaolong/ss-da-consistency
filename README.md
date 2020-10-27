# Self-Supervised Domain Adaptation with Consistency Training
[![arXiv](https://img.shields.io/badge/arXiv-2010.07539-b31b1b)](https://arxiv.org/abs/2010.07539)

Repository for the paper ["Self-Supervised Domain Adaptation with Consistency Training"](https://arxiv.org/abs/2010.07539).
```
@inproceedings{ss-da-consistency:2019,
  title={Self-Supervised Domain Adaptation with Consistency Training},
  author={L. Xiao, J. Xu, D. Zhao etal},
  booktitle={ICPR},
  year={2020}
}
```

## Requirements

- python3.5+

- pytorch 1.0+

## Pretrained models

- [caffenet](https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing)

## Prepare dataset

- [Digits](https://github.com/thuml/CDAN#digits)

- [PACS](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)

- [Image-clef](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view)

- [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)

- [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)

- [VLCS](http://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file)

## Running experiments

The configuration files for each experiment can be found at `config/` folder.

For example:

```shell
python3 main.py --config configs/<experiment>.yaml --seed <random_seed>
```

## References

[1] J. Xu, L. Xiao, A. M. Lopez. Self-supervised domain adaptation for computer vision tasks. IEEE Access, 2019.

[2] F. M. Carlucci, A. D’Innocente, S. Bucci, B. Caputo, and T. Tommasi. Domain generalization by solving jigsaw puzzles. In CVPR, 2019.
