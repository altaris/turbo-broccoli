"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""
__docformat__ = "google"

from pkg_resources import get_distribution, DistributionNotFound

from turbo_broccoli.turbo_broccoli import (
    TurboBroccoliDecoder,
    TurboBroccoliEncoder,
)
from turbo_broccoli.environment import (
    register_dataclass,
    set_artifact_path,
    set_keras_format,
    set_max_nbytes,
    set_pandas_format,
)

try:
    __version__ = get_distribution("turbo-broccoli").version
except DistributionNotFound:
    __version__ = "local"
