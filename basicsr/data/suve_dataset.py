import os
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils import data as data

from basicsr.data.data_util import read_img_seq, generate_frame_indices
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SUVEDataset(data.Dataset):
    """SUVE dataset for training.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_trans (str): Data root path for transmission map.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(SUVEDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

        self.data_infos = []
        with open( opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, sequence_length = line.strip().split(' ')
                data_info = dict(
                    folder=folder,
                    sequence_length=int(sequence_length)
                    )
                self.data_infos.append(data_info)

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = 1 # self.opt['scale']
        gt_size = self.opt['gt_size']
        item_dict = self.data_infos[index]
        gt_name = "_".join(item_dict['folder'].rsplit("_", 1)[:-1])

        # determine the temporal interval between neighboring frames
        interval = random.choice(self.interval_list)

        # randomly select a frame as start
        sequence_length = item_dict['sequence_length']
        if sequence_length - self.num_frame * interval < 0:
            raise ValueError(f'The input sequence is not long enough to support the current '
                             f'choice of {interval} or {self.num_frame}.')
        start_frame_idx = np.random.randint(
            0, sequence_length - self.num_frame * interval + 1
        )
        end_frame_idx = start_frame_idx + self.num_frame * interval
        frames_idx = list(range(start_frame_idx, end_frame_idx, interval))
        # randomly reverse
        if self.random_reverse and random.random() < 0.5:
            frames_idx.reverse()

        assert len(frames_idx) == self.num_frame, (f'Wrong length of frames list: {len(frames_idx)}')

        # get frames
        frames = os.listdir(os.path.join(self.lq_root, item_dict['folder']))
        frames.sort()

        # load lq frames
        lq_path = [
            os.path.join(self.lq_root, item_dict['folder'], str(frames[idx]))
            for idx in frames_idx
        ]
        imgs_lq = []
        for lq_frame_path in lq_path:
            img_bytes = self.file_client.get(lq_frame_path)
            img_lq = imfrombytes(img_bytes, float32=True)  # [0,1]
            imgs_lq.append(img_lq)

        # load gt frames
        gt_frame_path = os.path.join(self.gt_root, gt_name, str(frames[frames_idx[self.num_frame//2]]))
        img_bytes = self.file_client.get(gt_frame_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # randomly crop
        img_gt, imgs_lq = paired_random_crop(img_gt, imgs_lq, gt_size, scale)

        # augmentation - flip, rotate
        # if not isinstance(imgs_lq, list):
        #     imgs_lq = [imgs_lq]
        imgs_lq.append(img_gt)

        img_results = augment(imgs_lq, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        imgs_lq = torch.stack(img_results[:self.num_frame], dim=0)
        img_gt = img_results[-1]

        # imgs_lq: Tensor, size (t, c, h, w), RGB, [0, 1].
        # imgs_gt: Tensor, size (c, h, w), RGB, [0, 1].
        # folder: bashroomXXXX_23
        return {'lq': imgs_lq, 'gt': img_gt, 'folder': item_dict['folder']}

    def __len__(self):
        return len(self.data_infos)


@DATASET_REGISTRY.register()
class SUVETestDataset(data.Dataset):
    """SUVE dataset for testing.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test folders. If not provided, all the folders
                in the dataroot will be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(SUVETestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        self.num_frame = opt['num_frame']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate test dataset - {opt["name"]}')

        self.imgs_lq, self.imgs_gt = {}, {}
        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
            subfolders_lq = [os.path.join(self.lq_root, key) for key in subfolders]
            subfolders_gt = [os.path.join(self.gt_root, "_".join(key.rsplit("_", 1)[:-1])) for key in subfolders]

        logger.info(f'Example folder of any test video, lq: {subfolders_lq[0]}, gt: {subfolders_gt[0]}')

        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_name = os.path.basename(subfolder_lq)
            img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                    f' and gt folders ({len(img_paths_gt)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.num_frame // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                logger.info(f'Cache {subfolder_name} for SUVETestDataset...')
                self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.num_frame, padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])

