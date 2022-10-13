# PyramidNet-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Deep Pyramidal Residual Networks](https://arxiv.org/pdf/1610.02915.pdf).

## Table of contents

- [PyramidNet-PyTorch](#pyramidnet-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Deep Residual Learning for Image Recognition](#deep-residual-learning-for-image-recognition)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

modify the `config.py` file.

- line 29: `model_arch_name` change to `pyramidnet101`.
- line 31: `model_alpha` change to `360`.
- line 33: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 34: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 36: `model_num_classes` change to `1000`.
- line 38: `mode` change to `test`.
- line 90: `model_weights_path` change to `./results/pretrained_models/PyrmaidNet101_a360-ImageNet_1K-f040502b.pth.tar`.

```bash
python3 test.py
```

### Train model

modify the `config.py` file.

- line 29: `model_arch_name` change to `pyramidnet101`.
- line 31: `model_alpha` change to `360`.
- line 33: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 34: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 36: `model_num_classes` change to `1000`.
- line 38: `mode` change to `train`.

```bash
python3 train.py
```

### Resume train model

modify the `config.py` file.

- line 29: `model_arch_name` change to `pyramidnet101`.
- line 31: `model_alpha` change to `360`.
- line 33: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 34: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 36: `model_num_classes` change to `1000`.
- line 38: `mode` change to `train`.
- line 55: `resume` change to `./samples/pyramidnet101-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1610.02915.pdf](https://arxiv.org/pdf/1610.02915.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|       Model        |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:------------------:|:-----------:|:-----------------:|:-----------------:|
| PyramidNet200-a300 | ImageNet_1K |   19.5%(**-**)    |    4.8%(**-**)    |
| PyramidNet200-a450 | ImageNet_1K |   19.2%(**-**)    |    4.7%(**-**)    |

```bash
# Download `PyrmaidNet101_a360-ImageNet_1K-f040502b.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `pyramidnet101` model successfully.
Load `pyramidnet101` model weights `/PyramidNet-PyTorch/results/pretrained_models/PyrmaidNet101_a360-ImageNet_1K-f040502b.pth.tar` successfully.
tench, Tinca tinca                                                          (98.47%)
barracouta, snoek                                                           (0.94%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.07%)
reel                                                                        (0.03%)
sturgeon                                                                    (0.03%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Deep Residual Learning for Image Recognition

*Dongyoon Han, Jiwhan Kim, Junmo Kim*

##### Abstract

Deep convolutional neural networks (DCNNs) have shown remarkable performance in image classification tasks in recent
years. Generally, deep neural network architectures are stacks consisting of a large number of convolutional layers, and
they perform downsampling along the spatial dimension via pooling to reduce memory usage. Concurrently, the feature map
dimension (i.e., the number of channels) is sharply increased at downsampling locations, which is essential to ensure
effective performance because it increases the diversity of high-level attributes. This also applies to residual
networks and is very closely related to their performance. In this research, instead of sharply increasing the feature
map dimension at units that perform downsampling, we gradually increase the feature map dimension at all units to
involve as many locations as possible. This design, which is discussed in depth together with our new insights, has
proven to be an effective means of improving generalization ability. Furthermore, we propose a novel residual unit
capable of further improving the classification accuracy with our new network architecture. Experiments on benchmark
CIFAR-10, CIFAR-100, and ImageNet datasets have shown that our network architecture has superior generalization ability
compared to the original residual networks. Code is available at (this https URL)[https://github.com/jhkim89/PyramidNet]
.

[[Paper]](https://arxiv.org/pdf/1610.02915.pdf) [[Code]](https://github.com/jhkim89/PyramidNet)

```bibtex
@article{DPRN,
  title={Deep Pyramidal Residual Networks},
  author={Han, Dongyoon and Kim, Jiwhan and Kim, Junmo},
  journal={IEEE CVPR},
  year={2017}
}
```