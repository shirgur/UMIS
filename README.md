# UMIS
Official PyTorch implementation of "Unsupervised Microvascular Image Segmentation Using an Active Contours Mimicking Neural Network" ([link](https://arxiv.org/abs/1908.01373))

## Prerequisites
- Python 3.6
- Pytorch +1.4
- Numpy
- Scipy
- OpenCV
- Path
- tqdm
- h5py
- tifffile
- libtiff

### Morphological Pooling Layer
In order to build the Morphological Pooling layer on your own machine, run the following line
```
python src/setup.py install
```

## Train
You can now train using the Euler-Lagrange (original paper), or the PDE (level-set) loss with additional regularization for stability.
```
python train_unsup.py --loss <EL/LS>
```

# Citation
```
@inproceedings{gur2019unsupervised,
  title={Unsupervised Microvascular Image Segmentation Using an Active Contours Mimicking Neural Network},
  author={Gur, Shir and Wolf, Lior and Golgher, Lior and Blinder, Pablo},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={10722--10731},
  year={2019}
}
```