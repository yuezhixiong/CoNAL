# CoNAL: Learning Conflict-Noticed Architecture for Multi-Task Learning

This repository includes the pytorch implementation of "[Learning Conflict-Noticed Architecture for Multi-Task Learning](https://yuezhixiong.github.io/Papers/CoNAL.pdf)".

![scheme](./figures/figure-app.jpg)

## Citation

If you find ``CoNAL`` useful for research or development, please cite our paper:

```latex
@article{CoNAL,
  title={Learning Conflict-Noticed Architecture for Multi-Task Learning},
  author={Zhixiong Yue, Yu Zhang, Jie Liang},
  journal={Proceedings of the 2023 National Conference of the American Association for Artificial Intelligence (AAAI2023)},
  year={2023}
}
```

## Setup Environment

Please install following python packages:
```
- python
- numpy
- pytorch
- torchvision
- tensorboard
```

## Example Usage

Learn MTL architecture in NYUv2 dataset:
```
cd CoNAL
python train.py --model CoNAL
```

## Visualization
We use tensorboard to visualize the architecture learning process
```
tensorboard --logdir CoNAL/out
```
The learned architecture can be found in ```out/alpha_arrs.npy```

![method](./figures/figure1.jpg)