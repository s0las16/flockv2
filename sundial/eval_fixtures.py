"""Geo-eval fixture harvester — labeled frames from timestamped webcams.

geo_eval needs ground truth: a frame whose capture location is known.
foto-webcam.eu publishes just that — every JPEG carries a burned-in UTC
timestamp and each cam page embeds the camera's own latitude/longitude/
country in its metadata JSON. CAMS below is seeded from that site data
(all coords copied from the cams' pages, not guessed), plus room for
other public statics ("kind" distinguishes a single-frame "image" from a
"stream" endpoint, though harvest always stores one JPEG per fetch).

Stdlib only at import; all network I/O is lazy inside the functions.
Nothing here raises — offline or a dead cam just means fewer fixtures.

  python3 sundial/eval_fixtures.py                        # offline self-test
  SUNDIAL_NET=1 python3 sundial/eval_fixtures.py          # + 1 live fetch
  python -m sundial.eval_fixtures --out /tmp/geofixtures --per-cam 2
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

FOTO_BASE = "https://www.foto-webcam.eu/webcam"
TIMEOUT_S = 10
USER_AGENT = "sundial-hackathon/0.1 (video provenance verification)"
LABELS_FILE = "labels.json"
REQUIRED_KEYS = ("id", "name", "lat", "lon", "cc", "url", "kind")
KINDS = ("image", "stream")

__all__ = ["CAMS", "fetch_frame", "harvest", "labels_from_dir"]


def _foto(slug: str) -> str:
    """Latest large frame for a foto-webcam.eu cam (burned-in UTC stamp)."""
    return f"{FOTO_BASE}/{slug}/current/1200.jpg"


# ~15 known-good public cams. Coords copied from each cam's own
# foto-webcam.eu page metadata (verified); when a coord is only known
# approximately, the entry carries "approx": true. cc is ISO alpha-2.
CAMS = [
    # --- Germany ---
    {"id": "husum", "name": "Husum Hafenklappbrücke", "lat": 54.474171,
     "lon": 9.045982, "cc": "DE", "kind": "image",
     "url": _foto("husum-hafenklappbruecke")},
    {"id": "flensburg", "name": "Flensburg", "lat": 54.781286,
     "lon": 9.433995, "cc": "DE", "kind": "image",
     "url": _foto("flensburg")},
    {"id": "norderney", "name": "Norderney", "lat": 53.709701,
     "lon": 7.143195, "cc": "DE", "kind": "image",
     "url": _foto("norderney")},
    {"id": "darmstadt-ost", "name": "Darmstadt Ost", "lat": 49.875876,
     "lon": 8.662653, "cc": "DE", "kind": "image",
     "url": _foto("darmstadt-ost")},
    {"id": "feldberg-ts", "name": "Feldberg Taunus", "lat": 50.232203,
     "lon": 8.458489, "cc": "DE", "kind": "image",
     "url": _foto("feldberg-ts")},
    {"id": "bonn", "name": "Bonn", "lat": 50.760479, "lon": 7.085414,
     "cc": "DE", "kind": "image", "url": _foto("wetteronline-bonn")},
    {"id": "zugspitze", "name": "Zugspitze Gipfel", "lat": 47.42083,
     "lon": 10.98473, "cc": "DE", "kind": "image",
     "url": _foto("zugspitze")},
    {"id": "chiemsee", "name": "Bad Endorf (Chiemsee)", "lat": 47.89457,
     "lon": 12.30044, "cc": "DE", "kind": "image",
     "url": _foto("bad-endorf")},
    {"id": "eibsee", "name": "Eibsee Nord", "lat": 47.446384,
     "lon": 10.987823, "cc": "DE", "kind": "image",
     "url": _foto("eibsee-nord")},
    # --- Austria ---
    {"id": "dobratsch", "name": "Dobratsch (Villach)", "lat": 46.60355,
     "lon": 13.67034, "cc": "AT", "kind": "image",
     "url": _foto("dobratsch")},
    {"id": "wien", "name": "Wien Donaustadt", "lat": 48.233518,
     "lon": 16.411176, "cc": "AT", "kind": "image", "url": _foto("wien")},
    {"id": "bisamberg", "name": "Wien Bisamberg", "lat": 48.31067,
     "lon": 16.38251, "cc": "AT", "kind": "image",
     "url": _foto("bisamberg")},
    {"id": "innsbruck", "name": "Innsbruck", "lat": 47.305572,
     "lon": 11.377861, "cc": "AT", "kind": "image",
     "url": _foto("innsbruck")},
    {"id": "asitz", "name": "Leogang Asitz", "lat": 47.402468,
     "lon": 12.702289, "cc": "AT", "kind": "image", "url": _foto("asitz")},
    {"id": "dolomitenhuette", "name": "Dolomitenhütte", "lat": 46.789667,
     "lon": 12.78353, "cc": "AT", "kind": "image",
     "url": _foto("dolomitenhuette")},
    # --- Italy ---
    {"id": "eppan", "name": "Eppan Berg (near Bolzano)", "lat": 46.454609,
     "lon": 11.240582, "cc": "IT", "kind": "image",
     "url": _foto("eppan-berg")},
    {"id": "bruneck", "name": "Bruneck", "lat": 46.796379,
     "lon": 11.941309, "cc": "IT", "kind": "image",
     "url": _foto("bruneck")},
    # --- Switzerland ---
    {"id": "brunnen", "name": "Brunnen (Vierwaldstättersee)",
     "lat": 46.995571, "lon": 8.598627, "cc": "CH", "kind": "image",
     "url": _foto("brunnen")},
]


def _valid_cam(cam: dict) -> bool:
    """True when `cam` has the fixture keys with sane value ranges."""
    try:
        if not all(cam.get(k) for k in ("id", "url")):
            return False
        la, lo = float(cam["lat"]), float(cam["lon"])
        return -90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0
    except (KeyError, TypeError, ValueError):
        return False


def fetch_frame(cam: dict, out_dir: str) -> str | None:
    """GET `cam["url"]` and save it as ``<out_dir>/<id>.jpg``.

    Returns the saved path, or None on any failure — bad cam dict,
    network error, non-200, or a body that isn't actually a JPEG
    (a 200 HTML error page would otherwise poison the fixture set).
    """
    try:
        if not _valid_cam(cam):
            return None
        req = urllib.request.Request(
            str(cam["url"]), headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            data = r.read()
        if not (len(data) > 2 and data[:2] == b"\xff\xd8"):
            return None
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{cam['id']}.jpg")
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception:  # noqa: BLE001 — spec: never raise, None on failure
        return None


def harvest(
    out_dir: str,
    cams: list | None = None,
    per_cam: int = 2,
) -> dict:
    """Fetch `per_cam` frames per cam into `out_dir` plus a labels.json.

    Frames land as ``<id>_<i>.jpg``; labels.json is a list of
    {file, id, name, cc, lat, lon, kind} per saved frame and is always
    written (empty when nothing came down). Unreachable cams are skipped
    silently — offline runs still yield a valid, empty label set.
    Returns {"saved": int, "labels_path": str}.
    """
    labels = []
    saved = 0
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        pass  # labels_path is still reported; writes below fail the same way

    for cam in cams or CAMS:
        for i in range(max(0, int(per_cam))):
            tmp = fetch_frame(cam, out_dir)
            if tmp is None:
                break  # dead/unreachable cam — no point retrying i+1
            dest = os.path.join(out_dir, f"{cam['id']}_{i}.jpg")
            try:
                os.replace(tmp, dest)
            except OSError:
                break
            saved += 1
            labels.append(
                {
                    "file": os.path.basename(dest),
                    "id": cam["id"],
                    "name": cam.get("name"),
                    "cc": cam.get("cc"),
                    "lat": cam.get("lat"),
                    "lon": cam.get("lon"),
                    "kind": cam.get("kind", "image"),
                }
            )

    labels_path = os.path.join(out_dir, LABELS_FILE)
    try:
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return {"saved": saved, "labels_path": labels_path}


def labels_from_dir(dir: str) -> list:
    """Read ``<dir>/labels.json`` back into a list; [] on any problem."""
    try:
        with open(os.path.join(dir, LABELS_FILE), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001 — spec: never raise
        return []


def _selftest() -> int:
    """Offline checks; SUNDIAL_NET=1 adds one live fetch."""
    import tempfile

    ok = True

    def check(tag: str, good: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(good)
        print(f"  {tag:58} {'ok' if good else 'MISMATCH'} {extra}")

    print("CAMS schema:")
    ids = set()
    for cam in CAMS:
        missing = [k for k in REQUIRED_KEYS if k not in cam]
        check(f"  {cam.get('id')}: keys", not missing, f"missing={missing}")
        good = (
            not missing
            and isinstance(cam["lat"], (int, float))
            and isinstance(cam["lon"], (int, float))
            and -90.0 <= cam["lat"] <= 90.0
            and -180.0 <= cam["lon"] <= 180.0
            and isinstance(cam["cc"], str)
            and len(cam["cc"]) == 2
            and cam["kind"] in KINDS
            and str(cam["url"]).startswith("http")
            and cam["id"] not in ids
            and ("approx" not in cam
                 or isinstance(cam["approx"], bool))
        )
        check(f"  {cam.get('id')}: values", good,
              f"{cam.get('lat')},{cam.get('lon')} {cam.get('cc')}")
        ids.add(cam["id"])
    check(">=15 cams seeded", len(CAMS) >= 15, f"n={len(CAMS)}")

    print("labels round-trip:")
    with tempfile.TemporaryDirectory() as td:
        # Harvest a guaranteed-unreachable cam: 127.0.0.1:1 refuses fast,
        # no internet needed — exercises the graceful-skip path offline.
        dead = {
            "id": "unreachable", "name": "Nowhere", "lat": 0.0,
            "lon": 0.0, "cc": "DE", "kind": "image",
            "url": "http://127.0.0.1:1/none.jpg",
        }
        res = harvest(td, cams=[dead], per_cam=1)
        check("unreachable cam -> saved 0",
              res["saved"] == 0, f"res={res}")
        check("labels.json written anyway",
              os.path.exists(res["labels_path"]))
        check("labels_from_dir reads back empty list",
              labels_from_dir(td) == [])

        # Seed a labels.json by hand and check the parser's shape.
        want = [{"file": "a_0.jpg", "id": "a", "name": "A", "cc": "DE",
                 "lat": 50.0, "lon": 8.0, "kind": "image"}]
        with open(os.path.join(td, LABELS_FILE), "w",
                  encoding="utf-8") as f:
            json.dump(want, f)
        got = labels_from_dir(td)
        check("labels_from_dir round-trip", got == want)
        for k in ("file", "cc", "lat", "lon", "name"):
            check(f"label entry has '{k}'", k in (got[0] if got else {}))

        check("labels_from_dir(nonexistent) -> []",
              labels_from_dir(os.path.join(td, "nope")) == [])
        check("fetch_frame(bad cam) -> None",
              fetch_frame({"id": "x"}, td) is None)

    if os.environ.get("SUNDIAL_NET") == "1":
        print("live fetch (SUNDIAL_NET=1):")
        with tempfile.TemporaryDirectory() as td:
            cam = CAMS[0]
            path = fetch_frame(cam, td)
            if path is None:
                print(f"  fetch_frame({cam['id']}) -> None "
                      f"(offline? skipped, not a failure)")
            else:
                size = os.path.getsize(path)
                print(f"  fetch_frame({cam['id']}) -> {path} "
                      f"({size} bytes)")
                check("live frame is a real jpg", size > 1000)
    else:
        print("(offline — set SUNDIAL_NET=1 for one live fetch)")

    if not ok:
        print("SELF-TEST FAILED")
        return 1
    print("all checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="sundial.eval_fixtures", description=__doc__
    )
    ap.add_argument(
        "--out",
        help="fixture output dir; omitted -> run the offline self-test",
    )
    ap.add_argument(
        "--per-cam",
        type=int,
        default=2,
        help="frames to fetch per cam (default: 2)",
    )
    args = ap.parse_args(argv)

    if not args.out:
        return _selftest()

    res = harvest(args.out, per_cam=args.per_cam)
    print(f"saved {res['saved']} frame(s); labels -> {res['labels_path']}")
    if res["saved"] == 0:
        print("  (nothing fetched — offline? labels.json is empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
