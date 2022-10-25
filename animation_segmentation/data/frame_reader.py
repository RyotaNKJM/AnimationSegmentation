import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import numpy as np

from xmem.dataset.range_transform import im_normalization

from detectron2.data.detection_utils import read_image


class FrameReader(Dataset):
    def __init__(self, scene_name, frame_dir, to_save=None):
        """
        frame_dir: a directory of frames
        """
        self.scene_name = scene_name
        self.frame_dir = frame_dir

        self.frames = sorted(os.listdir(self.frame_dir))

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame

        im_path = path.join(self.frame_dir, frame)

        #ani_seg_img = read_image(im_path, format="BGR")

        vos_img = Image.open(im_path).convert('RGB')
        vos_img = np.array(vos_img)
        vos_img = self.im_transform(vos_img.copy())

        #data['ani_seg_img'] = ani_seg_img
        data['vos_img'] = vos_img
        data['info'] = info

        return data

    def __len__(self):
        return len(self.frames)
