"""File-based memoization for CosmosClient VLM calls.

VLM calls are the slow, non-deterministic part of a Sundial demo. This
module hashes each (frame-bytes, prompt) pair to a short key and stores
the JSON response under ``<root>/<key>.json`` so reruns are instant and
reproducible. Corrupt or unreadable entries are treated as misses.

Set ``SUNDIAL_NO_CACHE=1`` to bypass the cache entirely (every call is
forwarded to the client, nothing is read or written).

stdlib only: hashlib / json / os.
"""
from __future__ import annotations

import hashlib
import json
import os

_DISABLE_ENV = "SUNDIAL_NO_CACHE"

# CosmosClient methods that are pure (frame_path) -> JSON-able and safe to
# memoize. Keyed on the *prompt text* (not the method name) so edits to a
# prompt in .prompts naturally invalidate stale entries.
_METHODS = ("locate_shadow", "read_scene_text", "describe_conditions", "read_clock_overlay")


def _disabled() -> bool:
    return os.environ.get(_DISABLE_ENV) == "1"


def _method_prompts() -> dict:
    """Map CosmosClient method name -> prompt constant it sends.

    Imported lazily so this module still loads (and the __main__ demo still
    runs) outside package context; falls back to the method name itself,
    which is still deterministic and unique per method.
    """
    try:
        from .prompts import CLOCK_PROMPT, CONDITIONS_PROMPT, SHADOW_PROMPT, TEXT_PROMPT
    except Exception:
        return {name: name for name in _METHODS}
    return {
        "locate_shadow": SHADOW_PROMPT,
        "read_scene_text": TEXT_PROMPT,
        "describe_conditions": CONDITIONS_PROMPT,
        "read_clock_overlay": CLOCK_PROMPT,
    }


class VLMCache:
    """Content-addressed JSON store for VLM responses.

    ``key()`` mixes the frame's file bytes with the prompt text, so a
    re-encoded frame or an edited prompt invalidates old entries. ``get``
    returns the stored value or None on miss; note that a stored ``null``
    is therefore indistinguishable from a miss — ``wrap_cosmos_client``
    avoids this by never caching None results (so transient VLM failures
    simply retry on the next run).
    """

    def __init__(self, root: str = ".sundial_cache") -> None:
        self.root = root
        self.hits = 0
        self.misses = 0
        os.makedirs(root, exist_ok=True)

    def key(self, frame_path: str, prompt: str) -> str:
        """sha256(frame_bytes + prompt_text), first 12 hex chars."""
        h = hashlib.sha256()
        with open(frame_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\x00")  # unambiguous boundary between file bytes and prompt
        h.update(prompt.encode("utf-8", "replace"))
        return h.hexdigest()[:12]

    def get(self, key: str):
        """Return the cached value, or None on miss/corrupt/disabled."""
        if _disabled():
            return None
        try:
            with open(self._path(key), "r", encoding="utf-8") as f:
                value = json.load(f)
        except (OSError, ValueError):  # missing file, bad JSON, bad encoding
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value) -> bool:
        """Store a JSON-able value. Returns False if skipped/failed."""
        if _disabled() or value is None:
            return False
        try:
            blob = json.dumps(value)
        except (TypeError, ValueError):
            return False  # not JSON-able: never cached
        tmp = self._path(key) + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(blob)
            os.replace(tmp, self._path(key))  # atomic: no half-written entries
        except OSError:
            return False
        return True

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}

    def _path(self, key: str) -> str:
        # keys we generate are hex; sanitize anyway so a stray caller can't
        # smuggle path separators into the cache dir.
        safe = "".join(c if c.isalnum() else "_" for c in str(key)) or "_"
        return os.path.join(self.root, safe + ".json")


class _CachedClient:
    """Shallow proxy around CosmosClient.

    The four VLM methods consult the cache first; every other attribute
    (model, base_url, _ask, ...) is forwarded untouched.
    """

    def __init__(self, client, cache: VLMCache) -> None:
        self._client = client
        self.cache = cache  # public so callers can read .stats()
        prompts = _method_prompts()
        for name in _METHODS:
            setattr(self, name, self._bind(name, prompts.get(name, name)))

    def _bind(self, name: str, prompt: str):
        def call(frame_path: str):
            try:
                k = self.cache.key(frame_path, prompt)
            except OSError:
                # frame unreadable: the client's own methods degrade
                # gracefully (None/{}), so just forward and let it cope.
                return getattr(self._client, name)(frame_path)
            cached = self.cache.get(k)
            if cached is not None:
                return cached
            result = getattr(self._client, name)(frame_path)
            self.cache.put(k, result)
            return result

        call.__name__ = name
        return call

    def __getattr__(self, name: str):
        # Only reached for attributes not set above.
        return getattr(self._client, name)


def wrap_cosmos_client(client, cache: VLMCache | None = None):
    """Wrap a CosmosClient so its VLM calls are memoized on disk.

    Pass an existing ``VLMCache`` to share stats/root across clients; a
    default one at ``.sundial_cache`` is created otherwise. Honors
    ``SUNDIAL_NO_CACHE=1`` (transparent pass-through).
    """
    return _CachedClient(client, cache if cache is not None else VLMCache())


if __name__ == "__main__":
    import tempfile

    class _FakeCosmos:
        """Stand-in for CosmosClient: counts calls, returns fixed payloads."""

        def __init__(self) -> None:
            self.calls = 0

        def _respond(self, payload):
            self.calls += 1
            return payload

        def locate_shadow(self, frame_path):
            return self._respond(
                {
                    "object_class": "person",
                    "base_point": [12, 34],
                    "shadow_tip": [56, 78],
                    "confidence": 0.8,
                }
            )

        def read_scene_text(self, frame_path):
            return self._respond({"texts": ["MAIN ST"], "driving_side": "right"})

        def describe_conditions(self, frame_path):
            return self._respond({"sky": "clear", "lighting": "daylight"})

        def read_clock_overlay(self, frame_path):
            return self._respond({"overlay_timestamps": ["2024-06-01 15:04:12"]})

    with tempfile.TemporaryDirectory() as td:
        frame = os.path.join(td, "frame_0001.jpg")
        with open(frame, "wb") as f:
            f.write(b"\x89PNG-fake-frame" * 32)

        root = os.path.join(td, ".sundial_cache")

        fake = _FakeCosmos()
        client = wrap_cosmos_client(fake, VLMCache(root=root))
        print("pass 1 (cold cache):")
        print("  shadow:", client.locate_shadow(frame))
        print("  text:", client.read_scene_text(frame))
        print("  conditions:", client.describe_conditions(frame))
        print("  clock:", client.read_clock_overlay(frame))
        print("  stats:", client.cache.stats(), "| vlm calls:", fake.calls)

        # Fresh wrapper + fake over the same cache root: everything hits,
        # no VLM calls — exactly like a demo rerun.
        fake2 = _FakeCosmos()
        client2 = wrap_cosmos_client(fake2, VLMCache(root=root))
        print("pass 2 (warm cache, fresh client):")
        print("  shadow:", client2.locate_shadow(frame))
        print("  text:", client2.read_scene_text(frame))
        print("  conditions:", client2.describe_conditions(frame))
        print("  clock:", client2.read_clock_overlay(frame))
        print("  stats:", client2.cache.stats(), "| vlm calls:", fake2.calls)

        entries = sorted(os.listdir(root))
        print(f"cache dir: {root} ({len(entries)} entries)")
