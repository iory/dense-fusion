import cv2
import numpy as np

from dense_fusion.datasets.ycb.ycb_utils import get_object_pcds
from dense_fusion.datasets import YCBVideoPoseDataset
from dense_fusion.models import DenseFusion
from dense_fusion.visualizations.vis_bboxes import voc_colormap
from dense_fusion.visualizations import vis_pcd

import cameramodels

if __name__ == '__main__':
    num_points = 1000
    num_obj = 21
    model = DenseFusion()
    model.cuda()
    model.eval()

    dataset = YCBVideoPoseDataset(split='test')
    object_pcds = get_object_pcds()
    color_map = voc_colormap(num_obj + 1, order='bgr')

    for dataset_index in range(len(dataset)):
        rgb_img, depth, label_img, poses, label, bboxes = dataset[
            dataset_index]
        intrinsic_matrix = dataset.get_intrinsic(dataset_index)

        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        img_height, img_width, _ = rgb_img.shape
        cameramodel = cameramodels.PinholeCameraModel.from_intrinsic_matrix(
            intrinsic_matrix, img_height, img_width)

        rotations, translations = model.predict(
            rgb_img, depth, label_img, label, bboxes, intrinsic_matrix)

        # visualization
        for rotation, translation, itemid in zip(rotations,
                                                 translations,
                                                 label):
            pcd = np.array(object_pcds[itemid].points)
            bgr_img = vis_pcd(bgr_img, pcd,
                              cameramodel,
                              rotation=rotation,
                              translation=translation,
                              color=color_map[itemid])

        cv2.imwrite('./results/{0:09}.jpg'.format(dataset_index), bgr_img)
