"""Clip/frame source fetchers — where the test media comes from.

  - foto-webcam.eu  dated JPEG frames from European webcams, no key
  - TfL JamCam      London traffic-cam catalogue -> video URLs, no key
  - VAST S3 corpus  hackathon clip store on an S3-compatible endpoint

Third-party imports (requests, boto3, urllib3) are lazy inside the
functions that need them, so this module imports with zero dependencies.
requests falls back to stdlib urllib when unavailable.

  python -m sundial.sources --foto husum 2026-10-17 1230
  python -m sundial.sources --tfl --limit 10
  python -m sundial.sources --s3 bucket clip.mp4 out.mp4 \
      --endpoint-url http://lab.internal:9000 --insecure
"""
from __future__ import annotations

import argparse
import json
import os
import sys

FOTO_BASE = "https://www.foto-webcam.eu/webcam"
TFL_PLACES_URL = "https://api.tfl.gov.uk/Place/Type/JamCam"
TIMEOUT_S = 15
USER_AGENT = "sundial-hackathon/0.1 (video provenance verification)"

__all__ = ["foto_webcam_frame", "tfl_jamcams", "vast_s3_download"]


def _fetch_bytes(url: str) -> bytes:
    """GET `url` via requests, falling back to stdlib urllib.

    Raises FileNotFoundError on HTTP 404, RuntimeError on anything else.
    """
    try:
        import requests
    except ImportError:
        requests = None

    if requests is None:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(f"not found: {url}") from e
            raise RuntimeError(f"GET {url} -> HTTP {e.code}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"GET {url} failed: {e.reason}") from e

    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S
        )
    except requests.RequestException as e:
        raise RuntimeError(f"GET {url} failed: {e}") from e
    if resp.status_code == 404:
        raise FileNotFoundError(f"not found: {url}")
    if resp.status_code != 200:
        raise RuntimeError(
            f"GET {url} -> HTTP {resp.status_code}: {resp.text[:160]!r}"
        )
    return resp.content


def foto_webcam_frame(
    cam: str,
    ymd: str,
    hm: str,
    variant: str = "lm",
    out: str | None = None,
) -> str:
    """Download one dated frame from foto-webcam.eu. Returns the saved path.

    `ymd` is the site's "YYYY-MM-DD" date, `hm` the "HHMM" frame time
    ("12:30" is also accepted). `variant` is the site's image suffix —
    "lm" is the standard large frame. `out` defaults to
    "{cam}_{yyyymmdd}_{hm}_{variant}.jpg" in the cwd. Raises
    FileNotFoundError when that cam/date/time has no frame (HTTP 404).
    """
    parts = str(ymd).split("-")
    if len(parts) != 3:
        raise ValueError(f"ymd must look like 'YYYY-MM-DD', got {ymd!r}")
    y, m, d = parts[0], parts[1].zfill(2), parts[2].zfill(2)
    hm = str(hm).replace(":", "").strip()
    if len(hm) != 4 or not hm.isdigit():
        raise ValueError(f"hm must look like 'HHMM', got {hm!r}")

    url = f"{FOTO_BASE}/{cam}/{y}/{m}/{d}/{hm}_{variant}.jpg"
    data = _fetch_bytes(url)

    if not out:
        out = f"{cam}_{y}{m}{d}_{hm}_{variant}.jpg"
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)
    return out


def _tfl_place(place: dict) -> dict | None:
    """Flatten one TfL Place into {id, lat, lon, video_url, common_name}."""
    try:
        lat = float(place["lat"])
        lon = float(place["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    video_url = None
    for prop in place.get("additionalProperties") or []:
        if str(prop.get("key", "")).lower() == "videourl":
            video_url = prop.get("value")
            break
    return {
        "id": place.get("id"),
        "lat": lat,
        "lon": lon,
        "video_url": video_url,
        "common_name": place.get("commonName"),
    }


def tfl_jamcams(limit: int = 50) -> list[dict]:
    """List TfL JamCam traffic cameras -> [{id, lat, lon, video_url, common_name}].

    The Place/Type catalogue is served without auth; TFL_APP_ID /
    TFL_APP_KEY env vars are sent when set (they raise rate limits).
    `video_url` is None for the rare place without a videoUrl property.
    """
    import urllib.parse

    qs = urllib.parse.urlencode(
        {
            "app_id": os.environ.get("TFL_APP_ID", ""),
            "app_key": os.environ.get("TFL_APP_KEY", ""),
        }
    )
    payload = _fetch_bytes(f"{TFL_PLACES_URL}?{qs}")
    try:
        places = json.loads(payload)
    except ValueError as e:
        raise RuntimeError("TfL JamCam endpoint returned non-JSON") from e
    if not isinstance(places, list):
        raise RuntimeError(
            f"unexpected TfL payload: {str(places)[:200]}"
        )
    cams = [c for c in (_tfl_place(p) for p in places) if c is not None]
    return cams[: max(0, int(limit))]


def vast_s3_download(
    bucket: str,
    key: str,
    dest: str,
    endpoint_url: str | None = None,
    insecure: bool = False,
) -> str:
    """Download s3://bucket/key to `dest` via boto3. Returns `dest`.

    Credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
    AWS_DEFAULT_REGION env vars (boto3's chain still applies as fallback).
    `endpoint_url` selects a non-AWS S3-compatible store (the lab's VAST
    cluster); `insecure=True` disables TLS verification for its self-signed
    certs and silences the urllib3 warnings that would spam every request.
    """
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError(
            "vast_s3_download needs boto3 (pip install boto3)"
        ) from e

    if insecure:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        verify=False if insecure else None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    try:
        client.download_file(bucket, key, dest)
    except Exception as e:
        raise RuntimeError(f"s3://{bucket}/{key} -> {dest} failed: {e}") from e
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sundial.sources", description=__doc__)
    ap.add_argument(
        "--foto",
        nargs=3,
        metavar=("CAM", "YMD", "HM"),
        help="foto-webcam frame, e.g. --foto husum 2026-10-17 1230",
    )
    ap.add_argument(
        "--variant",
        default="lm",
        help="foto-webcam image suffix (default: lm)",
    )
    ap.add_argument("--out", help="save path for --foto")
    ap.add_argument(
        "--tfl", action="store_true", help="list TfL JamCam cameras"
    )
    ap.add_argument(
        "--limit", type=int, default=50, help="max cameras for --tfl"
    )
    ap.add_argument(
        "--s3",
        nargs=3,
        metavar=("BUCKET", "KEY", "DEST"),
        help="download an S3 object, e.g. --s3 corpus clip.mp4 out.mp4",
    )
    ap.add_argument(
        "--endpoint-url", help="S3-compatible endpoint URL for --s3"
    )
    ap.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS verification on --s3 (self-signed lab endpoints)",
    )
    args = ap.parse_args(argv)

    if not (args.foto or args.tfl or args.s3):
        ap.error("nothing to do — pass --foto, --tfl and/or --s3")

    rc = 0
    if args.foto:
        cam, ymd, hm = args.foto
        try:
            print(
                foto_webcam_frame(
                    cam, ymd, hm, variant=args.variant, out=args.out
                )
            )
        except Exception as e:
            print(f"foto-webcam: {e}", file=sys.stderr)
            rc = 1
    if args.tfl:
        try:
            cams = tfl_jamcams(limit=args.limit)
            print(f"{len(cams)} JamCams")
            for c in cams:
                print(
                    f"  {c['id']}  {c['lat']:.5f},{c['lon']:.5f}  "
                    f"{c['common_name']}  {c['video_url'] or '-'}"
                )
        except Exception as e:
            print(f"tfl: {e}", file=sys.stderr)
            rc = 1
    if args.s3:
        bucket, key, dest = args.s3
        try:
            print(
                vast_s3_download(
                    bucket,
                    key,
                    dest,
                    endpoint_url=args.endpoint_url,
                    insecure=args.insecure,
                )
            )
        except Exception as e:
            print(f"s3: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
