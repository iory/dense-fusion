# flake8: noqa

import os.path as osp
import pkg_resources


__version__ = pkg_resources.get_distribution("dense-fusion").version
CACHE_ROOT_PATH = osp.expanduser('~/.dense_fusion')


import dense_fusion.data
import dense_fusion.models
import dense_fusion.nn
import dense_fusion.visualizations
