"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""

from pkg_resources import DistributionNotFound, get_distribution

from turbo_broccoli.context import Context
from turbo_broccoli.custom.secret import (
    LockedSecret,
    Secret,
    SecretDict,
    SecretFloat,
    SecretInt,
    SecretList,
    SecretStr,
)
from turbo_broccoli.guard import GuardedBlockHandler
from turbo_broccoli.native import load, save
from turbo_broccoli.turbo_broccoli import (
    from_json,
    load_json,
    save_json,
    to_json,
)

try:
    __version__ = get_distribution("turbo-broccoli").version
except DistributionNotFound:
    __version__ = "local"
