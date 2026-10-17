"""Geo-family evidence dedupe — keep a pile of weak geo cards honest.

Sundial has many geo-flavored stages (geo, landmark, geoguess,
sign_style, vegetation, language, plate_style, osint...) and they are
all really answering the same question: *where was this filmed*. Five
weak cards saying roughly the same thing make the report look
contradictory when it isn't — so this module rolls the family up.

- ``geo_family()``           — is this evidence part of the geo-signal family?
- ``summarize_geo_family()`` — per-verdict counts + max confidence digest
- ``geo_rollup()``           — one "geo_rollup" Evidence for the pack (>=3 cards)
- ``compact_for_display()``  — collapse inconclusive geo cards into one UI row

Stdlib only; the schema is imported lazily so the module still loads
(and its ``__main__`` self-test still runs) with no package context.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotations only — the real import is lazy (see _schema)
    from .schema import Evidence

#: Evidence.type values that all speak to "where was this filmed" — one
#: redundant signal family as far as the report is concerned.
GEO_FAMILY_TYPES = frozenset({
    "geo", "geo_match", "geo_pred", "geotips", "landmark", "geoguess",
    "sign_style", "vegetation", "language", "plate_style", "plate_ocr",
    "osint", "osint_features", "osint_chain",
    "poi", "wiki_landmark", "wiki", "hemisphere", "timezone",
})

#: Rollup threshold — below this the cards speak for themselves.
_ROLLUP_MIN = 3

#: The rollup is a digest, not new evidence: fixed low confidence and a
#: fixed INCONCLUSIVE verdict. The adjudicator sees the underlying cards.
_ROLLUP_CONF = 0.25

_VERDICTS = ("supports", "contradicts", "inconclusive")


def _schema():
    """Lazy schema import — tolerates both package and direct-script use."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # `python sundial/dedupe_evidence.py` direct run
        from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


def _verdict_key(v: Any) -> str:
    """Normalize a ClaimVerdict / str verdict to supports|contradicts|inconclusive.

    Tolerates raw strings ("SUPPORTS"), enum members, and even the
    "ClaimVerdict.SUPPORTS" repr form; anything unrecognized buckets to
    inconclusive (same fallback as the report's pill renderer).
    """
    s = str(getattr(v, "value", v)).lower().rsplit(".", 1)[-1]
    return s if s in _VERDICTS else "inconclusive"


def _conf(ev: Any) -> float:
    try:
        return max(0.0, float(getattr(ev, "confidence", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def geo_family(e: Evidence) -> bool:
    """True when the evidence is one of the geo-signal types — the family
    whose cards tend to restate the same weak "where" signal."""
    t = str(getattr(e, "type", "") or "").strip().lower()
    return t in GEO_FAMILY_TYPES


def summarize_geo_family(evs: list[Evidence]) -> dict:
    """Digest stats over the geo-family cards in ``evs``.

    Counts per ClaimVerdict, the max confidence seen per verdict, and the
    sorted set of contributing evidence types. Non-geo cards are ignored.
    """
    geo = [e for e in (evs or []) if geo_family(e)]
    counts = {k: 0 for k in _VERDICTS}
    max_conf: dict[str, float] = {}
    for e in geo:
        k = _verdict_key(getattr(e, "verdict", None))
        counts[k] += 1
        max_conf[k] = max(max_conf.get(k, 0.0), _conf(e))
    return {
        "n_supports": counts["supports"],
        "n_contradicts": counts["contradicts"],
        "n_inconclusive": counts["inconclusive"],
        "max_conf_by_verdict": max_conf,
        "types": sorted(
            str(getattr(e, "type", "") or "").strip().lower() for e in geo
        ),
    }


def geo_rollup(evs: list[Evidence]) -> Evidence | None:
    """One digest card for the pack when >=3 geo-family cards exist.

    Always verdict=INCONCLUSIVE, confidence=0.25: the rollup is a digest
    so the report doesn't read as five disagreeing witnesses — the
    adjudicator still sees the underlying cards. Returns None below the
    threshold (or with no geo cards at all).
    """
    geo = [e for e in (evs or []) if geo_family(e)]
    if len(geo) < _ROLLUP_MIN:
        return None
    ClaimVerdict, Evidence = _schema()
    stats = summarize_geo_family(geo)
    summary = (
        f"geo signals: {stats['n_supports']} support, "
        f"{stats['n_contradicts']} contradict, "
        f"{stats['n_inconclusive']} inconclusive across {len(geo)} checks"
    )
    return Evidence(
        type="geo_rollup",
        summary=summary,
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=_ROLLUP_CONF,
        data=stats,
    )


def compact_for_display(evs: list[Evidence]) -> list[Evidence]:
    """New evidence list for the UI with the geo family compacted.

    Every INCONCLUSIVE geo-family card collapses into a single digest row
    sitting where the first collapsed card was; geo-family supports /
    contradicts cards and all non-geo cards pass through untouched, in
    order. The digest is a shallow copy (``copy.copy``) of the first
    collapsed card retyped "geo_rollup" — so it is not itself geo_family
    and re-running compact is idempotent — with ``data`` replaced by
    ``{"compact": True, "collapsed": n, **summarize_geo_family(...)}``.
    Originals are never mutated.
    """
    out: list = []
    collapsed: list = []
    digest_at: int | None = None  # slot in `out` reserved for the digest row
    for e in evs or []:
        if geo_family(e) and _verdict_key(getattr(e, "verdict", None)) == "inconclusive":
            if digest_at is None:
                digest_at = len(out)
                out.append(None)  # placeholder, filled once the group is known
            collapsed.append(e)
        else:
            out.append(e)

    if collapsed:
        stats = summarize_geo_family(collapsed)
        n = len(collapsed)
        digest = copy.copy(collapsed[0])
        digest.type = "geo_rollup"
        digest.summary = (
            f"geo signals: {n} inconclusive check{'s' if n != 1 else ''} "
            f"collapsed for display ({', '.join(stats['types'])})"
        )
        digest.confidence = max((_conf(e) for e in collapsed), default=0.0)
        digest.data = {"compact": True, "collapsed": n,
                       "channel": (getattr(collapsed[0], "data", None) or {}).get("channel", "derived"),
                       **stats}
        digest.frames = [
            f for e in collapsed for f in (getattr(e, "frames", None) or [])
        ][:4]
        out[digest_at] = digest
    return out


if __name__ == "__main__":
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # allow `python sundial/dedupe_evidence.py` directly
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from schema import ClaimVerdict, Evidence

    evs = [
        Evidence(type="geo", summary="sign text geolocates near claimed place",
                 verdict=ClaimVerdict.SUPPORTS, confidence=0.7),
        Evidence(type="landmark", summary="tower resembles claimed city",
                 verdict="supports", confidence=0.55),  # str verdict tolerated
        Evidence(type="plate_style", summary="plate format ambiguous",
                 verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.3),
        Evidence(type="language", summary="signage language inconclusive",
                 verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.4),
        Evidence(type="vegetation", summary="trees consistent with region",
                 verdict="inconclusive", confidence=0.2),
        Evidence(type="osint", summary="uploader hint contradicts claimed city",
                 verdict=ClaimVerdict.CONTRADICTS, confidence=0.45),
        Evidence(type="solar", summary="sun 12° vs expected 11°",
                 verdict=ClaimVerdict.SUPPORTS, confidence=0.8),
        Evidence(type="clock", summary='overlay reads "08:02"',
                 verdict=ClaimVerdict.SUPPORTS, confidence=0.85),
    ]

    # geo_family
    assert geo_family(evs[0]) and geo_family(evs[5])
    assert not geo_family(evs[6]) and not geo_family(evs[7])

    # summarize_geo_family — 6 geo cards: 2S / 1C / 3I
    stats = summarize_geo_family(evs)
    assert stats["n_supports"] == 2, stats
    assert stats["n_contradicts"] == 1, stats
    assert stats["n_inconclusive"] == 3, stats
    assert stats["max_conf_by_verdict"] == {
        "supports": 0.7, "contradicts": 0.45, "inconclusive": 0.4}, stats
    assert stats["types"] == [
        "geo", "landmark", "language", "osint", "plate_style", "vegetation"]

    # geo_rollup
    assert geo_rollup(evs[:2]) is None, "must return None below 3 geo cards"
    roll = geo_rollup(evs)
    assert roll is not None and roll.type == "geo_rollup"
    assert roll.verdict is ClaimVerdict.INCONCLUSIVE and roll.confidence == 0.25
    assert roll.summary == (
        "geo signals: 2 support, 1 contradict, 3 inconclusive across 6 checks")
    assert roll.data == stats
    assert not geo_family(roll), "rollup must not re-enter the family"

    # compact_for_display — 8 in -> 6 out: 2S+1C kept, 3I -> 1 digest, +2 non-geo
    compact = compact_for_display(evs)
    assert len(compact) == 6
    assert [e.type for e in compact] == [
        "geo", "landmark", "geo_rollup", "osint", "solar", "clock"]
    digest = compact[2]
    assert digest.data["compact"] is True and digest.data["collapsed"] == 3
    assert digest.data["n_inconclusive"] == 3
    assert digest.confidence == 0.4
    # originals untouched; idempotent on a second pass
    assert not any(getattr(e, "data", {}).get("compact") for e in evs)
    assert [e.type for e in compact_for_display(compact)] == [
        e.type for e in compact]

    print(f"{'type':12s} {'verdict':13s} {'conf':5s} summary")
    print("-" * 78)
    for e in compact:
        print(f"{e.type:12s} {str(getattr(e.verdict, 'value', e.verdict)):13s} "
              f"{e.confidence:<5.2f} {e.summary}")
    print("\nrollup:", roll.summary)
    print("digest data:", digest.data)
    print("\nOK — dedupe_evidence self-test passed")
