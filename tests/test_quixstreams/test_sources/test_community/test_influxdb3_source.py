"""
Broker-free tests for `InfluxDB3Source`.

The InfluxDB client is replaced with a small fake that serves
`SHOW MEASUREMENTS` and windowed data queries, and the Kafka producer is a mock,
so the tests assert about the windows that are queried and the records produced.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quixstreams.sources.community.influxdb3 import InfluxDB3Source

MODULE = "quixstreams.sources.community.influxdb3.influxdb3"

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


class FakeInfluxClient:
    """
    Stands in for `InfluxDBClient3`.

    `data` maps a measurement name to its rows; a data query returns the rows
    whose `time` falls into the requested `[start_time, end_time)` window.
    `fail_next_queries` makes the next N data queries raise.
    """

    def __init__(self, measurements, data=None, **_):
        self.measurements = measurements
        self.data = data or {}
        self.data_queries = []
        self.fail_next_queries = 0
        self.on_data_query = None
        self.closed = False

    def query(self, query, mode, language, query_parameters=None):
        assert mode == "pandas"
        assert language == "influxql"
        if query == "SHOW MEASUREMENTS":
            return pd.DataFrame({"name": self.measurements})

        start = datetime.fromisoformat(query_parameters["start_time"])
        end = datetime.fromisoformat(query_parameters["end_time"])
        measurement = query.split("FROM ")[1].split(" ")[0]
        self.data_queries.append((measurement, start, end))
        if self.on_data_query:
            self.on_data_query(len(self.data_queries))
        if self.fail_next_queries:
            self.fail_next_queries -= 1
            raise ConnectionError("query failed")

        rows = [
            row for row in self.data.get(measurement, []) if start <= row["time"] < end
        ]
        return pd.DataFrame(rows, columns=["time", "iox::measurement", "value"])

    def close(self):
        self.closed = True


@pytest.fixture()
def producer():
    mock = MagicMock()
    mock.flush.return_value = 0
    return mock


@pytest.fixture()
def source_factory(producer):
    """
    Builds a source wired to a `FakeInfluxClient` and a mocked producer.
    Returns `(source, client)`.
    """

    def factory(measurements=("cpu",), data=None, **kwargs):
        client = FakeInfluxClient(list(measurements), data)
        kwargs.setdefault("measurements", list(measurements))
        with patch(f"{MODULE}.InfluxDBClient3", return_value=client):
            source = InfluxDB3Source(
                host="http://localhost:8181",
                token="token",
                organization_id="org",
                database="db",
                **kwargs,
            )
        source.configure(topic=source.default_topic(), producer=producer)
        return source, client

    return factory


def _run(source, client):
    with patch(f"{MODULE}.InfluxDBClient3", return_value=client):
        source.start()


def _produced(producer):
    return [
        (call.kwargs["key"].decode(), json.loads(call.kwargs["value"]))
        for call in producer.produce.call_args_list
    ]


class TestStartDate:
    def test_default_start_date_is_resolved_when_source_is_created(
        self, source_factory
    ):
        before = datetime.now(tz=timezone.utc)
        source, _ = source_factory()
        after = datetime.now(tz=timezone.utc)

        assert before <= source._start_date <= after

    def test_default_start_date_is_not_shared_between_sources(self, source_factory):
        first, _ = source_factory()
        # Guarantee distinct timestamps even on a coarse clock
        while datetime.now(tz=timezone.utc) == first._start_date:
            pass
        second, _ = source_factory()

        assert second._start_date > first._start_date

    def test_explicit_start_date_is_used(self, source_factory):
        source, _ = source_factory(start_date=T0)

        assert source._start_date == T0


class TestWindows:
    def test_queries_tumbling_windows_until_end_date(self, source_factory, producer):
        source, client = source_factory(
            start_date=T0, end_date=T0 + timedelta(minutes=15), time_delta="5m"
        )

        _run(source, client)

        assert client.data_queries == [
            ("cpu", T0, T0 + timedelta(minutes=5)),
            ("cpu", T0 + timedelta(minutes=5), T0 + timedelta(minutes=10)),
            ("cpu", T0 + timedelta(minutes=10), T0 + timedelta(minutes=15)),
        ]
        assert client.closed

    def test_produces_records_of_each_window_with_measurement_name_as_key(
        self, source_factory, producer
    ):
        data = {
            "cpu": [
                {
                    "time": T0 + timedelta(minutes=1),
                    "iox::measurement": "cpu",
                    "value": 1,
                },
                {
                    "time": T0 + timedelta(minutes=7),
                    "iox::measurement": "cpu",
                    "value": 2,
                },
            ]
        }
        source, client = source_factory(
            data=data, start_date=T0, end_date=T0 + timedelta(minutes=10)
        )

        _run(source, client)

        produced = _produced(producer)
        assert [key for key, _ in produced] == ["cpu", "cpu"]
        assert [value["value"] for _, value in produced] == [1, 2]
        for _, value in produced:
            assert value["_measurement_name"] == "cpu"
            # The helper column InfluxDB adds is not part of the record
            assert "iox::measurement" not in value
        # Every batch is flushed to Kafka before the next window is queried
        assert producer.flush.call_count >= 2

    def test_measurements_are_processed_sequentially_when_end_date_is_set(
        self, source_factory
    ):
        source, client = source_factory(
            measurements=("cpu", "mem"),
            start_date=T0,
            end_date=T0 + timedelta(minutes=5),
        )

        _run(source, client)

        assert [(m, s, e) for m, s, e in client.data_queries] == [
            ("cpu", T0, T0 + timedelta(minutes=5)),
            ("mem", T0, T0 + timedelta(minutes=5)),
        ]

    def test_all_measurements_are_discovered_when_none_is_given(self, source_factory):
        source, client = source_factory(
            measurements=("cpu", "mem"),
            start_date=T0,
            end_date=T0 + timedelta(minutes=5),
        )
        source._measurements = []

        _run(source, client)

        assert {m for m, _, _ in client.data_queries} == {"cpu", "mem"}

    def test_stops_when_source_is_stopped(self, source_factory):
        # No end_date, start in the past: the source keeps querying until stopped
        source, client = source_factory(start_date=T0)
        client.on_data_query = lambda n: source.stop() if n == 3 else None

        _run(source, client)

        assert len(client.data_queries) == 3
        assert client.closed

    def test_custom_key_and_timestamp_setters(self, source_factory, producer):
        data = {
            "cpu": [
                {
                    "time": T0 + timedelta(minutes=1),
                    "iox::measurement": "cpu",
                    "value": 5,
                }
            ]
        }
        source, client = source_factory(
            data=data,
            start_date=T0,
            end_date=T0 + timedelta(minutes=5),
            key_setter=lambda record: f"host-{record['value']}",
            timestamp_setter=lambda record: 1234,
        )

        _run(source, client)

        (call,) = producer.produce.call_args_list
        assert call.kwargs["key"] == b"host-5"
        assert call.kwargs["timestamp"] == 1234


class TestRetries:
    def test_failed_query_is_retried_with_the_same_window(self, source_factory):
        source, client = source_factory(
            start_date=T0, end_date=T0 + timedelta(minutes=5), max_retries=2
        )
        client.fail_next_queries = 2

        with patch("time.sleep"):
            _run(source, client)

        window = ("cpu", T0, T0 + timedelta(minutes=5))
        assert client.data_queries == [window, window, window]

    def test_query_error_is_raised_when_retries_are_exhausted(self, source_factory):
        source, client = source_factory(
            start_date=T0, end_date=T0 + timedelta(minutes=5), max_retries=1
        )
        client.fail_next_queries = 5

        with patch("time.sleep"), pytest.raises(ConnectionError):
            _run(source, client)

        assert len(client.data_queries) == 2
        # The client is released even though the run failed
        assert client.closed
