# flake8: noqa

import pkg_resources


__version__ = pkg_resources.get_distribution("dense-fusion").version


import dense_fusion.models
import dense_fusion.nn
import dense_fusion.visualizations
