"""VAST corpus cross-reference — are there corpus clips near the claim?

Sponsor story (VAST data platform + NVIDIA VSS): after a provenance run
produces a claimed place/time, this stage asks the video index "does the
corpus hold footage recorded around that moment?" A hit is corroborating
context — the corpus happened to be rolling at the claimed time — never
a verdict by itself, so the card stays INCONCLUSIVE with the lean parked
in ``data["signal"]``.

Segment objects are named
``segments/20261001_055249_scene1_p1c1_chunk_0000_segment_001_of_006.mp4``
— capture UTC, scene, camera, chunk and segment numbering ride in the key,
so a bucket listing alone answers the question with no HEAD requests.

Stdlib only at module level; boto3 (S3 listing) and the schema (Evidence)
import lazily inside the functions that need them. Every entry point
degrades to ``{}`` / ``[]`` / an INCONCLUSIVE card on failure — on a dev
box without the lab env (S3_ENDPOINT, ACCESS_KEY, SECRET_KEY,
S3_SEGMENTS_BUCKET) this module must never raise.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

DEFAULT_PREFIX = "segments/"
DEFAULT_LIMIT = 200
NEAR_MINUTES = 60
TOP_K = 5

# segments/<yyyymmdd>_<hhmmss>_scene<N>_<cam>_chunk_<N>_segment_<N>_of_<N>.<ext>
# A leading path (vault/segments/...) is tolerated; the ext is any media suffix.
_KEY_RE = re.compile(
    r"(?:^|/)segments/"
    r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"
    r"_scene(\d+)_([A-Za-z0-9]+)"
    r"_chunk_(\d+)_segment_(\d+)_of_(\d+)"
    r"\.[A-Za-z0-9]+$",
    re.IGNORECASE,
)


def segment_place_time(key: str) -> dict:
    """Parse one segment object key -> {utc, scene, camera, chunk, segment, of}.

    ``utc`` is ISO-8601 Zulu ("2026-10-01T05:52:49Z"). Returns {} when the
    key doesn't match the corpus naming scheme or carries an impossible
    date (month 13, hour 25, ...).
    """
    m = _KEY_RE.search(str(key or ""))
    if not m:
        return {}
    y, mo, d, hh, mi, ss, scene, cam, chunk, seg, of = m.groups()
    try:
        dt = datetime(
            int(y), int(mo), int(d), int(hh), int(mi), int(ss),
            tzinfo=timezone.utc,
        )
    except (ValueError, OverflowError):
        return {}
    return {
        "utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scene": int(scene),
        "camera": cam,
        "chunk": int(chunk),
        "segment": int(seg),
        "of": int(of),
    }


def _parse_iso(s: object) -> datetime | None:
    """ISO string -> aware datetime (UTC assumed when naive). None on failure."""
    try:
        txt = str(s or "").strip()
        if not txt:
            return None
        if txt[-1] in ("Z", "z"):
            txt = txt[:-1] + "+00:00"  # fromisoformat <3.11 rejects "Z"
        dt = datetime.fromisoformat(txt)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def nearest_segments(
    bucket_keys: list[str],
    target_utc_iso: str | None,
) -> list[dict]:
    """The TOP_K parsed keys closest to `target_utc_iso`.

    Each hit is the parsed fields plus ``key`` and ``delta_minutes``
    (|segment utc - target| in minutes, None when no usable target).
    With no target the first TOP_K parsed keys come back in input order.
    """
    parsed: list[dict] = []
    dts: list[datetime] = []
    for key in bucket_keys or []:
        p = segment_place_time(key)
        if not p:
            continue
        seg_dt = _parse_iso(p["utc"])
        if seg_dt is None:  # our own "Z" format, but stay defensive
            continue
        parsed.append({**p, "key": key})
        dts.append(seg_dt)

    target = _parse_iso(target_utc_iso)
    if target is None:
        for p in parsed:
            p["delta_minutes"] = None
        return parsed[:TOP_K]

    for p, seg_dt in zip(parsed, dts):
        p["delta_minutes"] = round(
            abs((seg_dt - target).total_seconds()) / 60.0, 2
        )
    parsed.sort(key=lambda p: p["delta_minutes"])
    return parsed[:TOP_K]


def list_segment_keys(
    bucket: str | None = None,
    prefix: str = DEFAULT_PREFIX,
    limit: int = DEFAULT_LIMIT,
) -> list[str]:
    """List object keys under `prefix` in the lab's S3 segment bucket.

    boto3 imports lazily; credentials come from ACCESS_KEY / SECRET_KEY /
    S3_ENDPOINT env vars (AWS_* names as fallback), and TLS verification
    is off because the lab endpoint is self-signed. Returns [] on ANY
    failure — missing boto3, missing env, unreachable cluster — this
    function never raises.
    """
    try:
        import boto3
    except Exception:
        return []
    try:  # self-signed endpoint would spam InsecureRequestWarning otherwise
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    bucket = (
        bucket
        or os.environ.get("S3_SEGMENTS_BUCKET")
        or os.environ.get("VASTDB_BUCKET")
    )
    if not bucket:
        return []
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT

    try:
        client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT"),
            verify=False,
            aws_access_key_id=os.environ.get("ACCESS_KEY")
            or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("SECRET_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION"),
        )
        keys: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                k = obj.get("Key")
                if k:
                    keys.append(k)
                    if len(keys) >= limit:
                        return keys[:limit]
        return keys[:limit]
    except Exception:
        return []


def vss_evidence(
    keys: list[str],
    claim_time_iso: str | None,
) -> "Evidence":
    """Cross-reference corpus segment keys against the claimed time.

    A segment within NEAR_MINUTES of the claim means the corpus holds
    footage from ~the claimed time — corroborating context, so the card
    stays INCONCLUSIVE (confidence 0.35) with ``data["signal"] =
    "supports"`` marking the lean. Otherwise plain INCONCLUSIVE with the
    nearest keys in ``data["nearest"]``. Never raises.
    """
    from .schema import ClaimVerdict, Evidence  # lazy: zero-dep module import

    nearest = nearest_segments(keys, claim_time_iso)
    target = _parse_iso(claim_time_iso)
    data = {
        "claimed_time_utc": target.isoformat() if target else None,
        "keys_supplied": len(keys or []),
        "nearest": nearest,
    }

    if not nearest:
        return Evidence(
            type="corpus",
            summary="no corpus segments available to cross-reference",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )
    if target is None:
        return Evidence(
            type="corpus",
            summary=(
                f"{len(nearest)} corpus segment(s) indexed but no usable "
                "claim time to compare against"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )

    best = nearest[0]
    delta = best.get("delta_minutes")
    if delta is not None and delta <= NEAR_MINUTES:
        return Evidence(
            type="corpus",
            summary=(
                f"corpus contains footage from ~claimed time: {best['key']} "
                f"({best.get('camera', '?')}, scene {best.get('scene', '?')}) "
                f"is {delta:.0f} min off"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.35,
            data={**data, "signal": "supports"},
        )

    return Evidence(
        type="corpus",
        summary=(
            f"nearest corpus footage is {delta:.0f} min from the claimed "
            "time — outside the 60 min window"
            if delta is not None
            else "corpus segments indexed but none comparable to claim time"
        ),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.2,
        data=data,
    )


if __name__ == "__main__":
    samples = [
        "segments/20261001_055249_scene1_p1c1_chunk_0000_segment_001_of_006.mp4",
        "segments/20261001_055749_scene1_p1c1_chunk_0000_segment_002_of_006.mp4",
        "segments/20261001_060249_scene1_p1c2_chunk_0000_segment_003_of_006.mp4",
        "segments/20261001_180249_scene2_p2c1_chunk_0001_segment_004_of_006.mp4",
        "segments/20261002_055249_scene1_p1c1_chunk_0000_segment_001_of_006.mp4",
        "segments/20261301_055249_scene1_p1c1_chunk_0000_segment_001_of_006.mp4",  # month 13
        "notes/readme.txt",  # not a segment key
    ]
    print("parse:")
    parsed_ok = 0
    for k in samples:
        print(f"  {k}\n    -> {segment_place_time(k)}")
        parsed_ok += bool(segment_place_time(k))
    assert parsed_ok == 5, "expected exactly 5 valid segment keys"
    assert segment_place_time(samples[0]) == {
        "utc": "2026-10-01T05:52:49Z",
        "scene": 1,
        "camera": "p1c1",
        "chunk": 0,
        "segment": 1,
        "of": 6,
    }
    assert segment_place_time("garbage") == {}
    assert segment_place_time("") == {}
    assert segment_place_time(None) == {}

    target = "2026-10-01T06:00:00Z"
    print(f"\nnearest to {target}:")
    hits = nearest_segments(samples, target)
    for h in hits:
        print(f"  {h['delta_minutes']:>8} min  {h['key']}")
    assert hits[0]["key"] == samples[1], "segment @05:57:49 is 2.18 min off"
    assert hits[0]["delta_minutes"] == 2.18
    assert hits[1]["key"] == samples[2]  # @06:02:49, 2.82 min the other way

    print("\nno target (first 5 parsed):")
    for h in nearest_segments(samples, None):
        print(f"  {h['key']}")

    print("\nlist_segment_keys (no lab env -> []):", list_segment_keys())
    print("\nself-test OK")
