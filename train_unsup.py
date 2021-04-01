import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import numpy as np

from morphologicalpool.morphpoollayer import MorphPool3D
from data.datasets import Directory_Image_Train, Single_Image_Eval
from networks.segmentation import SegmentNet3D_Resnet
from networks.utils import GradXYZ, norm_range

from utils import Saver, TensorboardSummary
from networks.loss_functions import euler_lagrange, level_set

# args
parser = argparse.ArgumentParser(description='PyTorch Unsupervised Vessels Segmentation')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=6, metavar='N',
                    help='#CUDA * batch_size')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='Learning rate')
parser.add_argument('--lmd1', type=int, default=1, metavar='N',
                    help='lambda 1')
parser.add_argument('--lmd2', type=int, default=2, metavar='N',
                    help='lambda 2')
parser.add_argument('--range-norm', action='store_true',
                    help='range-norm')
parser.add_argument('--loss', type=str, default='LS', metavar='N',
                    help='Loss function')
parser.add_argument('--train-dataset', type=str, default='VesselNN', metavar='N',
                    help='Training dataset name')
parser.add_argument('--train-images-path', type=str, default='/path/to/VesselNN/train/images', metavar='N',
                    help='Training dataset images path')
parser.add_argument('--train-labels-path', type=str, default='/path/to/VesselNN/train/labels', metavar='N',
                    help='Training dataset labels path')
parser.add_argument('--val-image-path', type=str, default='/path/to/VesselNN/train/image', metavar='N',
                    help='Validation image path')
parser.add_argument('--val-label-path', type=str, default='/path/to/VesselNN/train/label', metavar='N',
                    help='Validation label path')

parser.add_argument('--validate', action='store_true',
                    help='validate')

# checking point
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default='VesselNN_Unsupervised',
                    help='set the checkpoint name')
args = parser.parse_args()

# Define Saver
saver = Saver(args)
saver.save_experiment_config()

# Define Tensorboard Summary
summary = TensorboardSummary(saver.experiment_dir)
writer = summary.create_summary()

# Data
dataset = Directory_Image_Train(images_path=args.train_images_path,
                                labels_path=args.train_labels_path,
                                data_shape=(32, 128, 128),
                                lables_shape=(32, 128, 128),
                                range_norm=args.range_norm)
dataloader = DataLoader(dataset, batch_size=torch.cuda.device_count() * args.batch_size, shuffle=True, num_workers=2)

# Data - validation
dataset_val = Single_Image_Eval(image_path=args.val_image_path,
                                label_path=args.val_label_path,
                                data_shape=(32, 128, 128),
                                lables_shape=(32, 128, 128),
                                stride=(8, 16, 16),
                                range_norm=args.range_norm)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)

# Train
model = SegmentNet3D_Resnet().cuda()
grad_fn = GradXYZ().cuda()
mp3d = MorphPool3D().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# DataParallel
model = torch.nn.DataParallel(model)
grad_fn = torch.nn.DataParallel(grad_fn)
mp3d = torch.nn.DataParallel(mp3d)

if args.resume:
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    model.module.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_pred = checkpoint['best_pred']

is_best = False
best_pred = 0

for epoch in range(args.epochs):
    model.train()
    iterator = tqdm(dataloader,
                    leave=True,
                    dynamic_ncols=True)
    for i, (data, _) in enumerate(iterator):
        # To CUDA
        data = data.cuda()

        # Misc
        dimsum = list(range(1, len(data.shape)))

        # Network
        seg, rec = model(data)
        grad_seg = grad_fn(seg)
        grad_rec = grad_fn(rec)

        # References
        area = seg.sum(dim=dimsum, keepdim=True)
        area_m = (1 - seg).sum(dim=dimsum, keepdim=True)
        c0 = (data * seg).sum(dim=dimsum, keepdim=True) / (area + 1e-8)
        c1 = (data * (1 - seg)).sum(dim=dimsum, keepdim=True) / (area_m + 1e-8)

        # Smooth
        seg = mp3d(seg)
        seg = mp3d(seg, True)
        seg = mp3d(seg)
        seg = mp3d(seg, True)
        seg = mp3d(seg)
        seg = mp3d(seg, True)

        # loss function
        if args.loss == 'EL':
            loss = euler_lagrange(data, seg, area, c0, c1, rec, grad_seg, grad_rec, args)
        elif args.loss == 'LS':
            loss = level_set(data, seg, area, c0, c1)
        else:
            raise Exception('Unsupported loss function')

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        iterator.set_description(
            'Epoch [{epoch}/{epochs}] :: Train Loss {loss:.4f}'.format(epoch=epoch, epochs=args.epochs,
                                                                       loss=loss.item()))
        writer.add_scalar('train/{loss_type}/total_loss_iter', loss.item(), epoch * len(dataloader) + i)

        if i % (len(dataloader) // 10):
            summary.visualize_image(writer, data, seg, epoch * len(dataloader) + i)

    if not epoch % 1:
        saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
        }, is_best)

        if args.validate:
            # Validate
            with torch.no_grad():
                model.eval()
                iterator = tqdm(dataloader_val,
                                leave=True,
                                dynamic_ncols=True,
                                desc='Validation ::')
                input = dataset_val.img[
                        dataset_val.effective_lable_idx[0][0]:dataset_val.effective_lable_idx[0][1],
                        dataset_val.effective_lable_idx[1][0]:dataset_val.effective_lable_idx[1][1],
                        dataset_val.effective_lable_idx[2][0]:dataset_val.effective_lable_idx[2][1]
                        ]
                input_gt = dataset_val.lbl[
                           dataset_val.effective_lable_idx[0][0]:dataset_val.effective_lable_idx[0][1],
                           dataset_val.effective_lable_idx[1][0]:dataset_val.effective_lable_idx[1][1],
                           dataset_val.effective_lable_idx[2][0]:dataset_val.effective_lable_idx[2][1]
                           ]
                input_gt = input_gt // input_gt.max()

                output = np.zeros((1,
                                   dataset_val.effective_lable_shape[0],
                                   dataset_val.effective_lable_shape[1],
                                   dataset_val.effective_lable_shape[2]))
                idx_sum = np.zeros((1,
                                    dataset_val.effective_lable_shape[0],
                                    dataset_val.effective_lable_shape[1],
                                    dataset_val.effective_lable_shape[2]))

                for index, (data, lables) in enumerate(iterator):
                    # To CUDA
                    data = data.cuda()
                    lables = lables.cuda()

                    # Network
                    seg, _ = model(data)

                    # Smooth
                    seg = mp3d(seg)
                    seg = mp3d(seg, True)
                    seg = mp3d(seg)
                    seg = mp3d(seg, True)
                    seg = mp3d(seg)
                    seg = mp3d(seg, True)

                    for batch_idx, val in enumerate(seg[:, 0]):
                        out_i = index * dataloader_val.batch_size + batch_idx
                        z, y, x = np.unravel_index(out_i, (dataset_val.dz, dataset_val.dy, dataset_val.dx))
                        z = z * dataset_val.stride[0]
                        y = y * dataset_val.stride[1]
                        x = x * dataset_val.stride[2]

                        idx_sum[0,
                        z: z + dataset_val.lables_shape[0],
                        y: y + dataset_val.lables_shape[1],
                        x: x + dataset_val.lables_shape[2]] += 1

                        output[0,
                        z: z + dataset_val.lables_shape[0],
                        y: y + dataset_val.lables_shape[1],
                        x: x + dataset_val.lables_shape[2]] += val.cpu().data.numpy()

                output = output / idx_sum
                output = torch.Tensor(output).unsqueeze(0).cuda()
                input_gt = torch.Tensor(input_gt).unsqueeze(0)

                # Normalize
                output = norm_range(output)

            # Plot
            input = torch.Tensor(input).unsqueeze(0).unsqueeze(0)
            summary.visualize_image_val(writer, input, output, epoch)
