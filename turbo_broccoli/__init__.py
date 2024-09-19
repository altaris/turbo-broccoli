"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""

from .context import Context
from .custom.embedded import EmbeddedDict, EmbeddedList
from .custom.external import ExternalData
from .guard import GuardedBlockHandler
from .native import load, save
from .parallel import Parallel, delayed
from .turbo_broccoli import (
    from_json,
    load_json,
    save_json,
    to_json,
)
from .user import register_decoder, register_encoder

try:
    from .custom.secret import (
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
