import os
from os import path

from .frame_reader import FrameReader


class FrameDataset:
    def __init__(self, frame_root):
        self.frame_root = frame_root
        self.scene_list = sorted(os.listdir(self.frame_root))

    def get_datasets(self):
        for scene in self.scene_list:
            yield FrameReader(
                scene,
                path.join(self.frame_root, scene)
            )

    def __len__(self):
        return len(self.scene_list)

