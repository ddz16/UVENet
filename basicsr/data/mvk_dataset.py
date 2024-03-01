import os
import numpy as np
from pathlib import Path
from torch.utils import data as data

from basicsr.data.data_util import read_img_seq
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MVKInferenceDataset(data.Dataset):
    """MVK dataset for inference.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            name (str): Dataset name.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(MVKInferenceDataset, self).__init__()
        self.opt = opt
        self.lq_root = opt['dataroot_lq']
        self.num_frame = opt['num_frame']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate test dataset - {opt["name"]}')

        self.data_infos = sorted(list(scandir(self.lq_root, full_path=True)))

    def __getitem__(self, index):
        lq_path = self.data_infos[index]
        folder = os.path.basename(lq_path)

        imgs_lq = read_img_seq(lq_path)

        # imgs_lq: Tensor, size (t, c, h, w), RGB, [0, 1].
        # folder: xxx
        return {'lq': imgs_lq, 'folder': folder}

    def __len__(self):
        return len(self.data_infos)