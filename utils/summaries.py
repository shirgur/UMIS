import os
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, image, output, global_step):
        z = np.random.randint(0, image.shape[2])

        grid_image = make_grid(image[:3, :, z, :, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        grid_image = make_grid(output[:3, :, z, :, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Output', grid_image, global_step)

        output[output > 0.9] = 1
        output[output <= 0.9] = 0
        grid_image = make_grid(output[:3, :, z, :, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Output - mask', grid_image, global_step)

    def visualize_image_val(self, writer, image, output, global_step):
        z = np.random.randint(0, image.shape[2])

        grid_image = make_grid(image[:3, :, z, :, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image VAL', grid_image, global_step)

        grid_image = make_grid(output[:3, :, z, :, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Output VAL', grid_image, global_step)

        output[output > 0.9] = 1
        output[output <= 0.9] = 0
        grid_image = make_grid(output[:3, :, z, :, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Output - mask VAL', grid_image, global_step)