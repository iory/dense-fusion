import cv2
import numpy as np
import skrobot


def vis_pcd(bgr_img, pcd, cameramodel, translation, rotation,
            color=(0, 255, 0), radius=2):
    coords = skrobot.coordinates.Coordinates(pos=translation, rot=rotation)
    transformed_pcd = coords.transform_vector(pcd)
    uvs = cameramodel.batch_project3d_to_pixel(
        transformed_pcd, project_valid_depth_only=True)
    uvs = np.array(uvs, dtype=np.int32)

    for uv in uvs:
        cv2.circle(bgr_img, tuple(uv), radius, color, thickness=-1)
    return bgr_img
