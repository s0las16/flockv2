"""File-based memoization for Nominatim geocode lookups.

geocode() calls are rate-limited (~1 req/s) and results are stable —
places don't move between demo runs. This module stores each query's
candidate list as JSON under ``<dir>/<sha1(normalized query)>.geo.json``
so reruns skip the network entirely. Entries older than 14 days count as
misses; corrupt or unreadable entries are misses too. A stored empty
list is a legitimate hit — it means "Nominatim had no match".

Set ``SUNDIAL_NO_CACHE=1`` to bypass the cache entirely (every call is
forwarded to the function, nothing is read or written).

stdlib only: dataclasses / functools / hashlib / json / os / time.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import time

_DISABLE_ENV = "SUNDIAL_NO_CACHE"
TTL_DAYS = 14
TTL_S = TTL_DAYS * 24 * 3600

_default: GeocodeCache | None = None


def _disabled() -> bool:
    return os.environ.get(_DISABLE_ENV) == "1"


def _candidate_cls():
    """GeoCandidate class if the schema is importable, else None.

    Imported lazily so this module still loads (and the __main__ self-test
    still runs) outside package context.
    """
    try:
        from .schema import GeoCandidate
    except Exception:
        return None
    return GeoCandidate


def _default_cache() -> GeocodeCache:
    """Process-wide cache at ./.sundial_cache, created on first wrap."""
    global _default
    if _default is None:
        _default = GeocodeCache()
    return _default


class GeocodeCache:
    """Query-addressed JSON store for geocode candidate lists.

    ``key()`` is sha1 of the normalized query (stripped, lowercased), so
    "Paris", "  paris " and "PARIS" share one entry. ``get`` returns the
    stored list of dicts, or None on miss/stale/corrupt/disabled. ``put``
    writes atomically (tmp file + os.replace), so a crashed run never
    leaves a half-written entry.
    """

    def __init__(self, dir: str = ".sundial_cache") -> None:
        self.dir = dir
        self.hits = 0
        self.misses = 0
        self.writes = 0
        os.makedirs(dir, exist_ok=True)

    def key(self, query: str) -> str:
        """sha1(query.strip().lower()) as hex — always filename-safe."""
        norm = str(query or "").strip().lower()
        return hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()

    def get(self, query: str) -> list[dict] | None:
        """Return the cached candidates, or None on miss/stale/corrupt."""
        if _disabled():
            return None
        path = self._path(query)
        try:
            if time.time() - os.path.getmtime(path) > TTL_S:
                self.misses += 1
                return None  # older than the TTL: treat as a miss
            with open(path, "r", encoding="utf-8") as f:
                value = json.load(f)
        except (OSError, ValueError):  # missing file, bad JSON, bad encoding
            self.misses += 1
            return None
        if not isinstance(value, list) or not all(
            isinstance(v, dict) for v in value
        ):
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, query: str, candidates: list[dict]) -> bool:
        """Store a candidate list. Returns False if skipped/failed."""
        if _disabled() or not isinstance(candidates, list):
            return False
        try:
            blob = json.dumps(candidates)
        except (TypeError, ValueError):
            return False  # not JSON-able: never cached
        path = self._path(query)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(blob)
            os.replace(tmp, path)  # atomic: no half-written entries
        except OSError:
            return False
        self.writes += 1
        return True

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }

    def _path(self, query: str) -> str:
        # key() output is sha1 hex — no separators to sanitize.
        return os.path.join(self.dir, self.key(query) + ".geo.json")


def _to_dicts(result) -> list[dict] | None:
    """Normalize a geocoder result to a JSON-able list of dicts.

    GeoCandidate (and any other dataclass instance) is asdict()'d here in
    the wrapper so the cache only ever stores plain dicts. Returns None
    when the result isn't a clean list-of-dicts — those are never cached.
    """
    if not isinstance(result, list):
        return None
    rows = [
        dataclasses.asdict(x)
        if dataclasses.is_dataclass(x) and not isinstance(x, type)
        else x
        for x in result
    ]
    if not all(isinstance(r, dict) for r in rows):
        return None
    return rows


def _from_dicts(rows: list[dict]) -> list:
    """Rebuild GeoCandidate objects from cached dicts when possible.

    Only applies when every dict's keys fit GeoCandidate's fields — raw
    dicts cached from a non-dataclass geocoder pass through untouched.
    Falls back to plain dicts when the schema isn't importable (e.g.
    running this file standalone).
    """
    cls = _candidate_cls()
    if cls is None:
        return rows
    names = {f.name for f in dataclasses.fields(cls)}
    if not all(set(r) <= names for r in rows):
        return rows
    try:
        return [cls(**r) for r in rows]
    except (TypeError, ValueError):  # missing required fields, bad values
        return rows


def wrap_geocode(fn=None, cache: GeocodeCache | None = None):
    """Memoize a ``(query: str) -> list[GeoCandidate]``-style function.

    Usable as ``wrap_geocode(fn)``, ``wrap_geocode(fn, cache=...)`` or
    ``@wrap_geocode(cache=...)``. The wrapper consults the cache on the
    string argument: on a hit it returns rebuilt GeoCandidate objects
    (plain dicts if the schema isn't importable); on a miss it calls
    through and stores the result, asdict()'ing dataclasses to plain
    dicts first. With no explicit cache a process-wide one at
    ``./.sundial_cache`` is used — see ``stats_line()``. Honors
    ``SUNDIAL_NO_CACHE=1`` as a transparent pass-through.
    """
    c = cache if cache is not None else _default_cache()

    def _wrap(f):
        @functools.wraps(f)
        def wrapper(query: str, *args, **kwargs):
            if _disabled():
                return f(query, *args, **kwargs)
            cached = c.get(query)
            if cached is not None:
                return _from_dicts(cached)
            result = f(query, *args, **kwargs)
            rows = _to_dicts(result)
            if rows is not None:
                c.put(query, rows)
            return result

        wrapper.cache = c  # public so callers can read .stats()
        return wrapper

    return _wrap(fn) if fn is not None else _wrap


def stats_line(cache: GeocodeCache | None = None) -> str:
    """One-line summary, e.g. "geo cache: 12 hits, 3 writes".

    Reports on ``cache`` when given, else on the process-wide default
    used by ``wrap_geocode`` (zeros if nothing has been wrapped yet —
    no cache dir is created just by asking).
    """
    c = cache if cache is not None else _default
    hits, writes = (c.hits, c.writes) if c is not None else (0, 0)
    return f"geo cache: {hits} hits, {writes} writes"


if __name__ == "__main__":
    import tempfile

    @dataclasses.dataclass
    class _FakeCandidate:
        """Stand-in for GeoCandidate when run outside package context."""

        name: str
        lat: float
        lon: float
        country_code: str = ""
        display_name: str = ""
        score: float = 0.0

    def _plain(x):
        return dataclasses.asdict(x) if dataclasses.is_dataclass(x) else x

    with tempfile.TemporaryDirectory() as td:
        cache = GeocodeCache(dir=os.path.join(td, ".sundial_cache"))

        # write/read roundtrip — normalized queries share one entry
        rows = [
            {
                "name": "Big Ben",
                "lat": 51.5007,
                "lon": -0.1246,
                "country_code": "GB",
                "display_name": "Big Ben, Westminster, London",
                "score": 0.9,
            }
        ]
        assert cache.get("big ben") is None  # cold miss
        assert cache.put("  Big Ben  ", rows)
        assert cache.get("BIG BEN") == rows
        assert cache.get("big ben") == rows
        print("roundtrip: ok")

        # TTL expiry: backdate the entry past 14 days -> miss
        path = cache._path("big ben")
        old = time.time() - (TTL_DAYS + 1) * 86400
        os.utime(path, (old, old))
        assert cache.get("big ben") is None
        print("ttl expiry: ok")

        # wrapper: dataclass results are asdict()'d on write, rebuilt on hit
        calls = [0]

        @wrap_geocode(cache=cache)
        def fake_geocode(query: str):
            calls[0] += 1
            return [
                _FakeCandidate(
                    name="Paris",
                    lat=48.8566,
                    lon=2.3522,
                    country_code="FR",
                    display_name="Paris, France",
                    score=0.95,
                )
            ]

        first = fake_geocode("paris")
        second = fake_geocode("  PARIS ")
        assert calls[0] == 1, calls  # second call served from cache
        assert _plain(first[0]) == _plain(second[0])

        # a no-match result is cached too: second call doesn't re-invoke
        @wrap_geocode(cache=cache)
        def fake_empty(query: str):
            calls[0] += 1
            return []

        assert fake_empty("nowhere-land") == []
        assert fake_empty("nowhere-land") == []
        assert calls[0] == 2, calls
        print("wrap_geocode: ok —", stats_line(cache))

        # SUNDIAL_NO_CACHE=1: no reads, no writes, straight pass-through
        os.environ[_DISABLE_ENV] = "1"
        assert cache.get("paris") is None
        assert fake_geocode("paris")  # forwarded to the live call
        assert calls[0] == 3, calls
        del os.environ[_DISABLE_ENV]
        print("SUNDIAL_NO_CACHE bypass: ok")

        entries = sorted(os.listdir(cache.dir))
        print(f"cache dir: {cache.dir} ({len(entries)} entries)")
        print("stats:", cache.stats())
        print("self-test passed")
