import argparse
import glob
import multiprocessing as mp
import os
from os import path
import time
import cv2
from tqdm import tqdm
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from annotation import FlameMaskAnnotation

from animation_segmentation.data.frame_datasets import FrameDataset
from ..xmem.inference.data.mask_mapper import MaskMapper
from ..xmem.model.network import XMem
from ..xmem.inference.inference_core import InferenceCore

from progressbar import progressbar

try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument('--model', default='../xmem/saves/XMem.pth')
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--output", help="A directory to save outputs. ", default="./outputs")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')

    parser.add_argument('--disable_long_term', action='store_true')
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements',
                        help='LT_max in paper, increase if objects disappear for a long time',
                        type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int,
                        default=-1)

    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    config = vars(args)
    config['enable_long_term'] = not config['disable_long_term']

    out_path = args.output

    torch.autograd.set_grad_enabled(False)

    meta_dataset = FrameDataset(args.input)
    meta_loader = meta_dataset.get_datasets()

    visualization = FlameMaskAnnotation(cfg)  # AnimeSegmentationModel

    network = XMem(config, args.model).cuda().eval()  # VOSModel
    model_weights = torch.load(args.model)
    network.load_weights(model_weights, init_as_zero_if_needed=True)

    total_process_time = 0
    total_frames = 0

    with tqdm(total=len(meta_dataset)) as scene_set:

        for scene_reader in meta_loader:

            loader = DataLoader(scene_reader, batch_size=1, shuffle=False, num_workers=2)
            scene_name = scene_reader.scene_name
            scene_length = len(loader)
            # no need to count usage for LT if the video is not that long anyway
            config['enable_long_term_count_usage'] = (
                    config['enable_long_term'] and
                    (scene_length
                     / (config['max_mid_term_frames'] - config['min_mid_term_frames'])
                     * config['num_prototypes'])
                    >= config['max_long_term_elements']
            )

            scene_set.set_description('Progress {}'.format(scene_name))

            ani_seg_scores = []
            ani_seg_result = []

            with tqdm(total=scene_length, leave=False) as frame_set:

                for ti, frame in enumerate(loader):
                    ani_seg_img = frame['ani_seg_img']
                    info = frame['info']
                    frame_name = info['frame']

                    frame_set.set_description('Progress {}'.format(frame_name))

                    # Anime Segmentation
                    predictions, visualized_output = visualization.run_to_mask(ani_seg_img)
                    ani_seg_scores.append(predictions["instances"].scores[0].item())
                    ani_seg_result.append(visualized_output)

                    frame_set.update(1)

            # best annotation mask
            best_frame = np.argmax(ani_seg_scores)
            mask = Image.fromarray(ani_seg_result[best_frame]).convert('P')
            mask = np.asarray(mask, dtype=np.uint8)

            # be possible to split and reverse
            frames = []
            for data in loader:
                frames.append(data)

            mapper = MaskMapper()
            processor = InferenceCore(network, config=config)

            indexes = list(range(scene_length))

            # VOS
            for ti, frame in zip(indexes[best_frame::-1], frames[best_frame::-1]):
                with torch.cuda.amp.autocast(enabled=not args.benchmark):
                    vos_img = frame['vos_img'].cuda()[0]
                    info = frame['info']
                    frame_name = info['frame'][0]

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                    if ti == best_frame:
                        vos_mask = mask
                    else:
                        vos_mask = None

                    if vos_mask is not None:
                        vos_mask, labels = mapper.convert_mask(vos_mask[0].numpy())
                        vos_mask = torch.Tensor(vos_mask).cuda()
                        processor.set_all_labels(list(mapper.remappings.values()))
                    else:
                        labels = None

                    prob = processor.step(vos_img, vos_mask, labels, end=(ti == 0))

                    end.record()
                    torch.cuda.synchronize()
                    total_process_time += (start.elapsed_time(end) / 1000)
                    total_frames += 1

                    # Probability mask -> index mask
                    out_mask = torch.argmax(prob, dim=0)
                    out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

                    # Save the mask
                    this_out_path = path.join(out_path, scene_name)
                    os.makedirs(this_out_path, exist_ok=True)
                    out_mask = mapper.remap_index_mask(out_mask)
                    out_img = Image.fromarray(out_mask)
                    out_img.save(os.path.join(this_out_path, frame_name[:-4] + '.png'))

            for ti, frame in zip(indexes[best_frame:], frames[best_frame:]):
                with torch.cuda.amp.autocast(enabled=not args.benchmark):
                    vos_img = frame['vos_img'].cuda()[0]
                    info = frame['info']
                    frame_name = info['frame'][0]

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                    if ti == best_frame:
                        vos_mask = mask
                    else:
                        vos_mask = None

                    if vos_mask is not None:
                        vos_mask, labels = mapper.convert_mask(vos_mask[0].numpy())
                        vos_mask = torch.Tensor(vos_mask).cuda()
                        processor.set_all_labels(list(mapper.remappings.values()))
                    else:
                        labels = None

                    prob = processor.step(vos_img, vos_mask, labels, end=(ti == scene_length-1))

                    end.record()
                    torch.cuda.synchronize()
                    total_process_time += (start.elapsed_time(end) / 1000)
                    total_frames += 1

                    # Probability mask -> index mask
                    out_mask = torch.argmax(prob, dim=0)
                    out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

                    # Save the mask
                    this_out_path = path.join(out_path, scene_name)
                    os.makedirs(this_out_path, exist_ok=True)
                    out_mask = mapper.remap_index_mask(out_mask)
                    out_img = Image.fromarray(out_mask)
                    out_img.save(os.path.join(this_out_path, frame_name[:-4] + '.png'))

            scene_set.update(1)

    print(f'Total processing time: {total_process_time}')
    print(f'Total processed frames: {total_frames}')
    print(f'FPS: {total_frames / total_process_time}')
    print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')


