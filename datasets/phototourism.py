import numpy as np
# import tensorflow as tf
import torch
from pathlib import Path
import torch.utils.data as data
from os import path as osp

# from .base_dataset import BaseDataset
from settings import DATA_PATH, EXPER_PATH
from utils.tools import dict_update
import cv2
from utils.utils import homography_scaling_torch as homography_scaling
from utils.utils import filter_points


class Phototourism(data.Dataset):
    default_config = {
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
        'homography_adaptation': {
            'enable': False
        }
    }

    def __init__(self, split="train", **config):
        self.split = split
        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        # self.transforms = transform
        # self.action = 'train' if task == 'train' else 'val'

        if self.split not in ["train", "val"]:
            raise ValueError("Check split. It should be [train, val]")
        # get files
        self.data_path = "/ssd/data/phototourism/"
        self.image_path = osp.join(self.data_path, "orig")
        self.imgs = [line.rstrip("\n") for line in open(osp.join(self.data_path,
                                                                 self.split + "_phototourism_ms.txt"))]

        sequence_set = []
        for img_fname in self.imgs:
            sample = {'image': osp.join(self.image_path, img_fname),
                      'name': img_fname}
            sequence_set.append(sample)
        self.samples = sequence_set

        self.init_var()
        pass

    def init_var(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        from utils.homographies import sample_homography_np as sample_homography
        from utils.utils import compute_valid_mask
        from utils.photometric import ImgAugTransform, customizedTransform
        from utils.utils import inv_warp_image, inv_warp_image_batch, warp_points

        self.sample_homography = sample_homography
        self.inv_warp_image = inv_warp_image
        self.inv_warp_image_batch = inv_warp_image_batch
        self.compute_valid_mask = compute_valid_mask
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform
        self.warp_points = warp_points

        self.enable_photo_train = self.config['augmentation']['photometric']['enable']
        self.enable_homo_train = self.config['augmentation']['homographic']['enable']

        self.enable_homo_val = False
        self.enable_photo_val = False

        self.cell_size = 8
        if self.config['preprocessing']['resize']:
            self.sizer = self.config['preprocessing']['resize']

    def format_sample(self, sample):
        return sample

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            image: tensor (H, W, channel=1)
        '''

        def _read_image(path):
            input_image = cv2.imread(path)
            # print(f"path: {path}, image: {image}")
            # print(f"path: {path}, image: {input_image.shape}")
            input_image = cv2.resize(input_image,
                                     (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = input_image.astype('float32') / 255.0
            return input_image

        '''
        def _preprocess(image):
            if self.transforms is not None:
                image = self.transforms(image)
            return image
        '''

        def imgPhotometric(img):
            """

            :param img:
                numpy (H, W)
            :return:
            """
            augmentation = self.ImgAugTransform(**self.config['augmentation'])
            img = img[:, :, np.newaxis]
            img = augmentation(img)
            cusAug = self.customizedTransform()
            img = cusAug(img, **self.config['augmentation'])
            return img

        from numpy.linalg import inv
        sample = self.samples[index]
        sample = self.format_sample(sample)
        input = {}
        input.update(sample)
        # image
        img_o = _read_image(sample['image'])
        H, W = img_o.shape[0], img_o.shape[1]
        # print(f"image: {image.shape}")
        img_aug = img_o.copy()
        '''
        if (self.enable_photo_train and self.action == 'train') or (
                self.enable_photo_val and self.action == 'val'):
            img_aug = imgPhotometric(img_o)  # numpy array (H, W, 1)
        '''
        # img_aug = _preprocess(img_aug[:,:,np.newaxis])
        img_aug = torch.tensor(img_aug, dtype=torch.float32).view(-1, H, W)

        valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
        input.update({'image': img_aug})
        input.update({'valid_mask': valid_mask})

        if self.config['homography_adaptation']['enable']:
            # img_aug = torch.tensor(img_aug)
            homoAdapt_iter = self.config['homography_adaptation']['num']
            homographies = np.stack([self.sample_homography(np.array([2, 2]), shift=-1,
                                                            **self.config['homography_adaptation']['homographies'][
                                                                'params'])
                                     for i in range(homoAdapt_iter)])
            ##### use inverse from the sample homography
            homographies = np.stack([inv(homography) for homography in homographies])
            homographies[0, :, :] = np.identity(3)
            # homographies_id = np.stack([homographies_id, homographies])[:-1,...]

            ######

            homographies = torch.tensor(homographies, dtype=torch.float32)
            inv_homographies = torch.stack([torch.inverse(homographies[i, :, :]) for i in range(homoAdapt_iter)])

            # images
            warped_img = self.inv_warp_image_batch(img_aug.squeeze().repeat(homoAdapt_iter, 1, 1, 1), inv_homographies,
                                                   mode='bilinear').unsqueeze(0)
            warped_img = warped_img.squeeze()
            # masks
            valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homographies,
                                                 erosion_radius=self.config['augmentation']['homographic'][
                                                     'valid_border_margin'])
            input.update({'image': warped_img, 'valid_mask': valid_mask, 'image_2D': img_aug})
            input.update({'homographies': homographies, 'inv_homographies': inv_homographies})

        name = sample['name']
        input.update({'name': name, 'scene_name': "./"})  # dummy scene name
        return input

    def __len__(self):
        return len(self.samples)
