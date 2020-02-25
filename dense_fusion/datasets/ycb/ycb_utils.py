import os
import os.path as osp

import gdown
import open3d

from dense_fusion import CACHE_ROOT_PATH


dataset_config_dir = osp.join(osp.abspath(osp.dirname(__file__)), 'config')
ycb_video_dataset_url = 'https://drive.google.com/uc?id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi'  # NOQA
YCB_VIDEO_DATASET_PATH = os.environ.get(
    'YCB_VIDEO_DATASET_PATH',
    osp.join(CACHE_ROOT_PATH, 'YCB_Video_Dataset'))


def get_ycb_video_dataset():
    if not osp.exists(YCB_VIDEO_DATASET_PATH):
        print('Downloading YCB_Video_Dataset')
        print("This will take time.")
        print("If you already downloaded YCB_Video_Dataset, "
              "Please 'export YCB_VIDEO_DATASET_PATH="
              "<YOUR YCB_Video_Dataset PATH>'"
              " and rerun program.")
        gdown.cached_download(
            url=ycb_video_dataset_url,
            path=osp.expanduser(
                osp.join(CACHE_ROOT_PATH, "YCB_Video_Dataset.zip")),
            md5="c9122e177a766a9691cab13c5cda41a9",
            postprocess=gdown.extractall,
            quiet=True,
        )
    return YCB_VIDEO_DATASET_PATH


def get_data_path(split='train'):
    return osp.join(dataset_config_dir,
                    '{}_data_list.txt'.format(split))


def get_data_list(split='train'):
    test_data_list_path = get_data_path(split)
    with open(test_data_list_path) as f:
        return f.read().split('\n')


def get_object_pcds():
    pcds = []
    for label_name in label_names:
        if label_name == '__background__':
            pcd = []
        else:
            xyzpath = osp.join(
                '/home/iory/dataset/YCB_Video_Dataset',
                'models/{}/points.xyz'.format(label_name))
            pcd = open3d.io.read_point_cloud(xyzpath)
        pcds.append(pcd)
    return pcds


label_names = ('__background__',
               '002_master_chef_can',
               '003_cracker_box',
               '004_sugar_box',
               '005_tomato_soup_can',
               '006_mustard_bottle',
               '007_tuna_fish_can',
               '008_pudding_box',
               '009_gelatin_box',
               '010_potted_meat_can',
               '011_banana',
               '019_pitcher_base',
               '021_bleach_cleanser',
               '024_bowl',
               '025_mug',
               '035_power_drill',
               '036_wood_block',
               '037_scissors',
               '040_large_marker',
               '051_large_clamp',
               '052_extra_large_clamp',
               '061_foam_brick')
