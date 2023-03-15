"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""
__docformat__ = "google"

from pkg_resources import DistributionNotFound, get_distribution

from turbo_broccoli.environment import (
    register_dataclass_type,
    register_pytorch_module_type,
    set_artifact_path,
    set_keras_format,
    set_max_nbytes,
    set_pandas_format,
)
from turbo_broccoli.guard import guarded_call, produces_document
from turbo_broccoli.turbo_broccoli import (
    TurboBroccoliDecoder,
    TurboBroccoliEncoder,
    from_json,
    to_json,
)

try:
    __version__ = get_distribution("turbo-broccoli").version
except DistributionNotFound:
    __version__ = "local"
