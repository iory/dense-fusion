from __future__ import division

import os.path as osp

import cameramodels
import numpy as np
import numpy.ma as ma
import PIL
import scipy.io
import torch

from dense_fusion.datasets.ycb.ycb_utils import get_data_list
from dense_fusion.datasets.ycb.ycb_utils import get_ycb_video_dataset
from dense_fusion.datasets.ycb.ycb_utils import label_names


def get_bbox(label, img_height, img_width, base=40):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    r_b = int((r_b + base - 1) / base) * base
    c_b = cmax - cmin
    c_b = int((c_b + base - 1) / base) * base
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_height:
        delt = rmax - img_height
        rmax = img_height
        rmin -= delt
    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    return rmin, cmin, rmax, cmax


class YCBVideoPoseDataset(torch.utils.data.Dataset):

    def __init__(self, split='train', dataset_path=None):
        if split not in ['train', 'test']:
            raise ValueError(
                "{} split {} is not supported. "
                "Only 'train' and 'test' are supported.".format(
                    self.__class__.__name__, split))
        super(YCBVideoPoseDataset, self).__init__()

        if dataset_path is None:
            self.data_dir = get_ycb_video_dataset()
        else:
            self.data_dir = dataset_path
        self.label_names = label_names
        self.ids = get_data_list(split)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rgb_img = self.get_image(idx)
        depth_img = self.get_depth(idx)
        meta_data = self.load_meta_data(idx)

        depth_scale = self.get_depth_scale(data=meta_data)
        depth_img = np.array(depth_img, dtype=np.float32) / depth_scale

        height, width, _ = rgb_img.shape
        intrinsic_matrix = self.get_intrinsic(data=meta_data)
        cm = cameramodels.PinholeCameraModel.from_intrinsic_matrix(
            intrinsic_matrix, height, width)
        uv = np.hstack([np.tile(np.arange(width), height)[:, None],
                        np.repeat(np.arange(height), width)[:, None]])
        depth = np.array(
            depth_img / self.get_depth_scale(data=meta_data), 'f')
        points = cm.batch_project_pixel_to_3d_ray(uv) * depth.reshape(-1, 1)
        points = points.reshape(height, width, 3)

        poses = self.get_pose(data=meta_data)
        label = self.get_label(data=meta_data)
        label_img = self.get_label_image(idx)

        bboxes = []
        new_label = []
        for label_idx in label:
            mask_label = ma.getmaskarray(
                ma.masked_equal(label_img, label_idx))
            if not np.any(mask_label):
                continue
            bboxes.append(get_bbox(mask_label, height, width))
            new_label.append(label_idx)
        new_label = np.array(new_label, dtype=np.int32)

        return rgb_img, depth_img, label_img, poses, new_label, bboxes

    def load_meta_data(self, idx):
        meta_path = osp.join(
            self.data_dir, '{}-meta.mat'.format(self.ids[idx]))
        data = scipy.io.loadmat(meta_path)
        return data

    def get_label_image(self, idx):
        img_path = osp.join(
            self.data_dir, '{}-label.png'.format(self.ids[idx]))
        img = np.array(PIL.Image.open(img_path))
        return img

    def get_pose(self, idx=None, data=None):
        if data is None:
            data = self.load_meta_data(idx)
        rt = data['poses'].transpose((2, 0, 1))
        pose = np.zeros((len(rt), 4, 4), dtype=np.float32)
        pose[:, 3, 3] = 1
        pose[:, :3, :3] = rt[:, :, :3]
        pose[:, :3, 3] = rt[:, :, 3]
        pose = pose.transpose((0, 2, 1))
        return pose

    def get_image(self, i):
        imgpath = osp.join(self.data_dir, '{}-color.png'.format(self.ids[i]))
        img = np.array(PIL.Image.open(imgpath))
        return img

    def get_depth(self, i):
        depthpath = osp.join(self.data_dir, '{}-depth.png'.format(self.ids[i]))
        depth = np.array(PIL.Image.open(depthpath), dtype=np.uint16)
        return depth

    def get_label(self, idx=None, data=None):
        if data is None:
            data = self.load_meta_data(idx)
        object_ids = data['cls_indexes'].flatten()
        object_ids = object_ids
        return object_ids

    def get_depth_scale(self, idx=None, data=None):
        if data is None:
            data = self.load_meta_data(idx)
        depth_scale = data['factor_depth'][0][0]
        return depth_scale

    def get_intrinsic(self, idx=None, data=None):
        if data is None:
            data = self.load_meta_data(idx)
        intrinsic_matrix = data['intrinsic_matrix']
        return intrinsic_matrix


if __name__ == '__main__':
    import cv2

    from dense_fusion.visualizations import vis_bboxes

    dataset = YCBVideoPoseDataset(split='test')

    index = 0
    prev_index = -1
    while True:
        if prev_index != index:
            rgb_img, depth_img, label_img, poses, label, bboxes = \
                dataset[index]
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            bboxes = np.array(bboxes, dtype=np.int32)
            vis_bboxes(bgr_img, bboxes, label.reshape(-1),
                       label_names=label_names)
            cv2.imshow(dataset.__class__.__name__, bgr_img)
        prev_index = index

        k = cv2.waitKey(10)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        elif k == ord('n'):
            if index == len(dataset) - 1:
                print('WARNING: reached edge index of dataset: %d' % index)
                continue
            index += 1
        elif k == ord('p'):
            if index == 0:
                print('WARNING: reached edge index of dataset: %d' % index)
                continue
            index -= 1
