import os.path as osp

import gdown

from dense_fusion import CACHE_ROOT_PATH


url = "https://drive.google.com/uc?id=1gfOnOojzVdEwPzSaPmS3t3aJaQptbys6"


pose_net_model_names = {
    'ycb': 'pose_model_26_0.012863246640872631.pth',
    'linemod': 'pose_model_9_0.01310166542980859.pth',
}

pose_refine_net_model_names = {
    'ycb': 'pose_refine_model_69_0.009449292959118935.pth',
    'linemod': 'pose_refine_model_493_0.006761023565178073.pth',
}


def pose_net_pretrained_model_path(dataset='ycb'):
    if dataset not in ['ycb', 'linemod']:
        raise ValueError("current supported dataset type is "
                         "'ycb' and 'linemod' only.")
    model_path = osp.join(
        CACHE_ROOT_PATH,
        'trained_checkpoints',
        dataset,
        pose_net_model_names[dataset])
    if not osp.exists(model_path):
        gdown.cached_download(
            url=url,
            path=osp.expanduser(
                osp.join(CACHE_ROOT_PATH, "trained_check_points.zip")),
            md5="32c8274ac17674ffa69ade549973ca48",
            postprocess=gdown.extractall,
            quiet=True,
        )
    return model_path


def pose_refine_net_pretrained_model_path(dataset='ycb'):
    if dataset not in ['ycb', 'linemod']:
        raise ValueError("current supported dataset type is "
                         "'ycb' and 'linemod' only.")
    model_path = osp.join(
        CACHE_ROOT_PATH,
        'trained_checkpoints',
        dataset,
        pose_refine_net_model_names[dataset])
    if not osp.exists(model_path):
        gdown.cached_download(
            url=url,
            path=osp.expanduser(
                osp.join(CACHE_ROOT_PATH, "trained_check_points.zip")),
            md5="32c8274ac17674ffa69ade549973ca48",
            postprocess=gdown.extractall,
            quiet=True,
        )
    return model_path
