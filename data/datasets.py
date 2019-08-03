import torch
from torch.utils.data import Dataset
import glob
import tifffile as T
from libtiff import TIFF
import numpy as np


def range_normalize(v):
    v = (v - v.mean(axis=(1, 2), keepdims=True)) / (v.std(axis=(1, 2), keepdims=True) + 1e-12)
    v_min, v_max = v.min(axis=(1, 2), keepdims=True), v.max(axis=(1, 2), keepdims=True)
    v = (v - v_min) / (v_max - v_min + 1e-5)

    return v


def smart_padding(img, data_shape, lables_shape, stride):
    if img.shape[0] < data_shape[0]:
        img = np.pad(img, ((0, data_shape[0] - img.shape[0]), (0, 0), (0, 0)), mode='reflect')
    if img.shape[1] < data_shape[1]:
        img = np.pad(img, ((0, 0), (0, data_shape[1] - img.shape[1]), (0, 0)), mode='reflect')
    if img.shape[2] < data_shape[2]:
        img = np.pad(img, ((0, 0), (0, 0), (0, data_shape[2] - img.shape[1])), mode='reflect')

    dz = int(np.floor((img.shape[0] - data_shape[0]) / stride[0] + 1))
    dy = int(np.floor((img.shape[1] - data_shape[1]) / stride[1] + 1))
    dx = int(np.floor((img.shape[2] - data_shape[2]) / stride[2] + 1))
    effective_data_shape = (
        data_shape[0] * dz - (data_shape[0] - stride[0]) * (dz - 1),
        data_shape[1] * dy - (data_shape[1] - stride[1]) * (dy - 1),
        data_shape[2] * dx - (data_shape[2] - stride[2]) * (dx - 1)
    )

    if effective_data_shape[0] < img.shape[0]:
        img = np.pad(img,
                     (
                         (0, (data_shape[0] * (dz + 1) - (data_shape[0] - stride[0]) * dz) - img.shape[0]),
                         (0, 0),
                         (0, 0)),
                     mode='reflect')
    if effective_data_shape[1] < img.shape[1]:
        img = np.pad(img,
                     (
                         (0, 0),
                         (0, (data_shape[1] * (dy + 1) - (data_shape[1] - stride[1]) * dy) - img.shape[1]),
                         (0, 0)),
                     mode='reflect')
    if effective_data_shape[2] < img.shape[2]:
        img = np.pad(img,
                     (
                         (0, 0),
                         (0, 0),
                         (0, (data_shape[2] * (dx + 1) - (data_shape[2] - stride[2]) * dx) - img.shape[2])),
                     mode='reflect')

    effective_data_shape = img.shape
    effective_lable_shape = (
        effective_data_shape[0] - (data_shape[0] - lables_shape[0]),
        effective_data_shape[1] - (data_shape[1] - lables_shape[1]),
        effective_data_shape[2] - (data_shape[2] - lables_shape[2])
    )
    if effective_lable_shape[0] < img.shape[0]:
        img = np.pad(img, (((data_shape[0] - lables_shape[0]) // 2,
                            (data_shape[0] - lables_shape[0]) // 2 + (data_shape[0] - lables_shape[0]) % 2),
                           (0, 0),
                           (0, 0)),
                     mode='reflect')
    if effective_lable_shape[1] < img.shape[1]:
        img = np.pad(img, ((0, 0),
                           ((data_shape[1] - lables_shape[1]) // 2,
                            (data_shape[1] - lables_shape[1]) // 2 + (data_shape[1] - lables_shape[1]) % 2),
                           (0, 0)),
                     mode='reflect')
    if effective_lable_shape[2] < img.shape[2]:
        img = np.pad(img, ((0, 0),
                           (0, 0),
                           ((data_shape[2] - lables_shape[2]) // 2,
                            (data_shape[2] - lables_shape[2]) // 2 + (
                                    data_shape[2] - lables_shape[2]) % 2)),
                     mode='reflect')

    return img


class Single_Image_Eval(Dataset):
    def __init__(self,
                 image_path='HaftJavaherian_DeepVess2018_GroundTruthImage.tif',
                 label_path='HaftJavaherian_DeepVess2018_GroundTruthLabel.tif',
                 data_shape=(7, 33, 33),
                 lables_shape=(1, 4, 4),
                 stride=(1, 1, 1),
                 range_norm=False):
        self.range_norm = range_norm
        try:
            img = T.imread(image_path)
        except:
            img = []
            tif = TIFF.open(image_path)
            for _image in tif.iter_images():
                img.append(_image)
            img = np.stack(img, 0)
        try:
            lbl = T.imread(label_path)
        except:
            lbl = []
            tif = TIFF.open(label_path)
            for _lable in tif.iter_images():
                lbl.append(_lable)
            lbl = np.stack(lbl, 0)

        img = smart_padding(img, data_shape, lables_shape, stride)
        lbl = smart_padding(lbl, data_shape, lables_shape, stride)
        self.org_shape = img.shape

        self.img = img.astype(np.float32)
        self.lbl = lbl.astype(np.float32)
        self.shape = self.img.shape
        self.data_shape = data_shape
        self.lables_shape = lables_shape
        self.stride = stride

        self.dz = int(np.floor((self.shape[0] - data_shape[0]) / stride[0] + 1))
        self.dy = int(np.floor((self.shape[1] - data_shape[1]) / stride[1] + 1))
        self.dx = int(np.floor((self.shape[2] - data_shape[2]) / stride[2] + 1))
        self.effective_data_shape = (
            data_shape[0] * self.dz - (data_shape[0] - stride[0]) * (self.dz - 1),
            data_shape[1] * self.dy - (data_shape[1] - stride[1]) * (self.dy - 1),
            data_shape[2] * self.dx - (data_shape[2] - stride[2]) * (self.dx - 1)
        )

        self.effective_lable_shape = (
            self.effective_data_shape[0] - (data_shape[0] - lables_shape[0]),
            self.effective_data_shape[1] - (data_shape[1] - lables_shape[1]),
            self.effective_data_shape[2] - (data_shape[2] - lables_shape[2])
        )

        self.effective_lable_idx = (
            ((data_shape[0] - lables_shape[0]) // 2,
             self.effective_data_shape[0] - (
                     (data_shape[0] - lables_shape[0]) // 2 + (data_shape[0] - lables_shape[0]) % 2)),
            ((data_shape[1] - lables_shape[1]) // 2,
             self.effective_data_shape[1] - (
                     (data_shape[1] - lables_shape[1]) // 2 + (data_shape[1] - lables_shape[1]) % 2)),
            ((data_shape[2] - lables_shape[2]) // 2,
             self.effective_data_shape[2] - (
                     (data_shape[2] - lables_shape[2]) // 2 + (data_shape[2] - lables_shape[2]) % 2))
        )

        self.lbl_z = ((data_shape[0] - lables_shape[0]) // 2,
                      (data_shape[0] - lables_shape[0]) // 2 + lables_shape[0])
        self.lbl_y = ((data_shape[1] - lables_shape[1]) // 2,
                      (data_shape[1] - lables_shape[1]) // 2 + lables_shape[1])
        self.lbl_x = ((data_shape[2] - lables_shape[2]) // 2,
                      (data_shape[2] - lables_shape[2]) // 2 + lables_shape[2])

        self.max_iter = self.dz * self.dy * self.dx

    def __len__(self):
        return self.max_iter

    def __getitem__(self, index):
        z, y, x = np.unravel_index(index, (self.dz, self.dy, self.dx))
        z = z * self.stride[0]
        y = y * self.stride[1]
        x = x * self.stride[2]

        v = self.img[z: z + self.data_shape[0],
            y: y + self.data_shape[1],
            x: x + self.data_shape[2]]
        lbl = self.lbl[z: z + self.data_shape[0],
              y: y + self.data_shape[1],
              x: x + self.data_shape[2]]
        lbl = lbl[self.lbl_z[0]: self.lbl_z[1],
              self.lbl_y[0]: self.lbl_y[1],
              self.lbl_x[0]: self.lbl_x[1]]

        # Normalize
        if self.range_norm:
            v = range_normalize(v)
        else:
            v = (v - v.mean(axis=(1, 2), keepdims=True)) / (v.std(axis=(1, 2), keepdims=True) + 1e-12)

        # To Tensor
        data = torch.Tensor(v).unsqueeze(0)
        lables = torch.Tensor(lbl // self.lbl.max()).long()

        return data, lables


class Directory_Image_Train(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 max_iter=1000,
                 data_shape=(7, 33, 33),
                 lables_shape=(1, 4, 4),
                 stride=(1, 1, 1),
                 range_norm=False):
        self.range_norm = range_norm
        images = sorted(glob.glob(images_path + '/*tif'))
        labels = sorted(glob.glob(labels_path + '/*tif'))

        self.org_shape = []
        self.shape = []
        self.img = []
        self.lbl = []
        self.data_shape = data_shape
        self.lables_shape = lables_shape
        self.stride = stride
        self.dz = []
        self.dy = []
        self.dx = []
        self.effective_data_shape = []
        self.effective_lable_shape = []
        self.effective_lable_idx = []
        self.lbl_z = []
        self.lbl_y = []
        self.lbl_x = []

        for img_path, lbl_path in zip(images, labels):
            try:
                img = T.imread(img_path)
            except:
                img = []
                tif = TIFF.open(img_path)
                for _image in tif.iter_images():
                    img.append(_image)
                img = np.stack(img, 0)
            try:
                lbl = T.imread(lbl_path)
            except:
                lbl = []
                tif = TIFF.open(lbl_path)
                for _lable in tif.iter_images():
                    lbl.append(_lable)
                lbl = np.stack(lbl, 0)

            img = smart_padding(img, data_shape, lables_shape, stride)
            lbl = smart_padding(lbl, data_shape, lables_shape, stride)

            self.org_shape.append(img.shape)

            self.img.append(img.astype(np.float32))
            self.lbl.append(lbl.astype(np.float32))

            shape = img.shape
            self.shape.append(shape)

            dz = int(np.floor((shape[0] - data_shape[0]) / stride[0] + 1))
            dy = int(np.floor((shape[1] - data_shape[1]) / stride[1] + 1))
            dx = int(np.floor((shape[2] - data_shape[2]) / stride[2] + 1))
            effective_data_shape = (
                data_shape[0] * dz - (data_shape[0] - stride[0]) * (dz - 1),
                data_shape[1] * dy - (data_shape[1] - stride[1]) * (dy - 1),
                data_shape[2] * dx - (data_shape[2] - stride[2]) * (dx - 1)
            )

            effective_lable_shape = (
                effective_data_shape[0] - (data_shape[0] - lables_shape[0]),
                effective_data_shape[1] - (data_shape[1] - lables_shape[1]),
                effective_data_shape[2] - (data_shape[2] - lables_shape[2])
            )

            effective_lable_idx = (
                ((data_shape[0] - lables_shape[0]) // 2,
                 effective_data_shape[0] - (
                         (data_shape[0] - lables_shape[0]) // 2 + (data_shape[0] - lables_shape[0]) % 2)),
                ((data_shape[1] - lables_shape[1]) // 2,
                 effective_data_shape[1] - (
                         (data_shape[1] - lables_shape[1]) // 2 + (data_shape[1] - lables_shape[1]) % 2)),
                ((data_shape[2] - lables_shape[2]) // 2,
                 effective_data_shape[2] - (
                         (data_shape[2] - lables_shape[2]) // 2 + (data_shape[2] - lables_shape[2]) % 2))
            )

            lbl_z = ((data_shape[0] - lables_shape[0]) // 2,
                     (data_shape[0] - lables_shape[0]) // 2 + lables_shape[0])
            lbl_y = ((data_shape[1] - lables_shape[1]) // 2,
                     (data_shape[1] - lables_shape[1]) // 2 + lables_shape[1])
            lbl_x = ((data_shape[2] - lables_shape[2]) // 2,
                     (data_shape[2] - lables_shape[2]) // 2 + lables_shape[2])

            self.dz.append(dz)
            self.dy.append(dy)
            self.dx.append(dx)
            self.effective_data_shape.append(effective_data_shape)
            self.effective_lable_shape.append(effective_lable_shape)
            self.effective_lable_idx.append(effective_lable_idx)
            self.lbl_z.append(lbl_z)
            self.lbl_y.append(lbl_y)
            self.lbl_x.append(lbl_x)

        self.max_iter = max_iter

    def __len__(self):
        return self.max_iter

    def __getitem__(self, index):
        i = np.random.randint(0, len(self.img))
        z = np.random.randint(0, self.dz[i])
        y = np.random.randint(0, self.dy[i])
        x = np.random.randint(0, self.dx[i])
        z = z * self.stride[0]
        y = y * self.stride[1]
        x = x * self.stride[2]

        v = self.img[i][z: z + self.data_shape[0],
            y: y + self.data_shape[1],
            x: x + self.data_shape[2]]
        lbl = self.lbl[i][z: z + self.data_shape[0],
              y: y + self.data_shape[1],
              x: x + self.data_shape[2]]
        lbl = lbl[self.lbl_z[i][0]: self.lbl_z[i][1],
              self.lbl_y[i][0]: self.lbl_y[i][1],
              self.lbl_x[i][0]: self.lbl_x[i][1]]

        # Normalize
        if self.range_norm:
            v = range_normalize(v)
        else:
            v = (v - v.mean(axis=(1, 2), keepdims=True)) / (v.std(axis=(1, 2), keepdims=True) + 1e-12)

        # To Tensor
        data = torch.Tensor(v).unsqueeze(0)
        lables = torch.Tensor(lbl // 255).long()

        return data, lables
