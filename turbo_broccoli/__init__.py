"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""

from pkg_resources import DistributionNotFound, get_distribution

from turbo_broccoli.context import Context
from turbo_broccoli.custom.embedded import EmbeddedDict, EmbeddedList
from turbo_broccoli.guard import GuardedBlockHandler
from turbo_broccoli.native import load, save
from turbo_broccoli.parallel import Parallel, delayed
from turbo_broccoli.turbo_broccoli import (
    from_json,
    load_json,
    save_json,
    to_json,
)
from turbo_broccoli.user import register_decoder, register_encoder

try:
    from turbo_broccoli.custom.secret import (
        LockedSecret,
        Secret,
        SecretDict,
        SecretFloat,
        SecretInt,
        SecretList,
        SecretStr,
    )
except ModuleNotFoundError:
    pass

try:
    __version__ = get_distribution("turbo-broccoli").version
except DistributionNotFound:
    __version__ = "local"
