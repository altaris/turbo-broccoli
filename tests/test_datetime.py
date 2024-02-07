# pylint: disable=missing-function-docstring
"""datetime objects (de)serialization test suite"""

from datetime import datetime, timedelta, timezone

from common import assert_to_from_json


def test_datetime_datetime():
    assert_to_from_json(datetime.now())
    assert_to_from_json(datetime.utcnow())
    assert_to_from_json(datetime.now(tz=timezone(timedelta(hours=5))))


def test_datetime_time():
    assert_to_from_json(datetime.now().time())
    assert_to_from_json(datetime.utcnow().time())
    assert_to_from_json(datetime.now(tz=timezone(timedelta(hours=5))).time())


def test_datetime_timedelta():
    td = timedelta(
        days=50,
        seconds=27,
        microseconds=10,
        milliseconds=29000,
        minutes=5,
        hours=8,
        weeks=2,
    )
    assert_to_from_json(td)
    assert_to_from_json(timedelta(microseconds=-1))
