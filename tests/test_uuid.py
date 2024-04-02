# pylint: disable=missing-function-docstring
"""bytes (de)serialization test suite"""

import uuid

from common import assert_to_from_json


def test_uuid():
    u1 = uuid.uuid1()
    assert_to_from_json({"a_uuid1": u1})
    u3 = uuid.uuid3(uuid.NAMESPACE_DNS, "python.org")
    assert_to_from_json({"a_uuid3": u3})
    u4 = uuid.uuid4()
    assert_to_from_json({"a_uuid4": u4})
    u5 = uuid.uuid5(uuid.NAMESPACE_DNS, "python.org")
    assert_to_from_json({"a_uuid5": u5})
