"""Persistent job store for the sundial demo server.

serve.py keeps live jobs in the in-memory ``_JOBS`` dict, which dies with the
process — every restart turns saved ``/result/<jobid>`` links into the
"expired" page. This module mirrors each job to disk so results survive:

  POST /verify → _start_job() seeds _JOBS[jobid] and spawns _worker()
  _worker()    → pipeline.run() + report.render_html(), job flips "done"/"error"
  save()       → writes <STORE_DIR>/<jobid>.json (atomic tmp+rename) plus a
                 copy of the report at <STORE_DIR>/<jobid>.html
  load()       → rehydrates the dict; "report" is repointed at the stored
                 .html so _result() can serve it after a restart
  load_all()   → {jobid: job} for every stored entry; corrupt files skipped
  sweep()      → deletes stored jobs older than the TTL, returns count removed
  status_of()  → cheap {status,verdict,confidence,claim_time,claim_place} read

Every function is total: bad input, missing files, corrupt JSON, and I/O
errors return None/False/{}/0 rather than raising — the demo must degrade,
never 500. Non-JSON-serializable values are coerced with ``default=str``.

Storage dir resolves lazily: ``$SUNDIAL_JOB_DIR`` or ``~/.sundial_jobs/``.
Note ``from sundial.job_store import STORE_DIR`` snapshots the value at import
time; ``job_store.STORE_DIR`` stays lazy. Stdlib only.
"""
from __future__ import annotations

import json
import os
import shutil
import time

_DEFAULT_TTL_S = 7200            # match serve._JOB_TTL_S
_JSON_EXT = ".json"
_HTML_EXT = ".html"


# ---------------------------------------------------------------------------
# storage dir — resolved lazily so env overrides work even after import

def _store_dir() -> str:
    env = (os.environ.get("SUNDIAL_JOB_DIR") or "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.expanduser("~"), ".sundial_jobs")


def __getattr__(name: str):      # PEP 562: lazy module-level STORE_DIR
    if name == "STORE_DIR":
        return _store_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# internals

def _paths(jobid: str) -> tuple[str, str] | None:
    """(json_path, html_path) for a jobid, or None if the id is unsafe."""
    if (not jobid or jobid.startswith(".")
            or os.sep in jobid
            or (os.altsep is not None and os.altsep in jobid)):
        return None
    base = _store_dir()
    return (os.path.join(base, jobid + _JSON_EXT),
            os.path.join(base, jobid + _HTML_EXT))


def _atomic_write(path: str, data: bytes) -> bool:
    """tmp+rename so a reader never sees a half-written file."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------------
# public API — none of these ever raise

def save(jobid: str, job_dict: dict) -> bool:
    """Persist job metadata and (if present) its report HTML. Never raises."""
    try:
        paths = _paths(str(jobid))
        if paths is None or not isinstance(job_dict, dict):
            return False
        json_path, html_path = paths
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        try:
            report = job_dict.get("report")
            if isinstance(report, str) and os.path.isfile(report):
                tmp = html_path + ".tmp"
                shutil.copyfile(report, tmp)
                os.replace(tmp, html_path)
        except Exception:
            pass                        # metadata still persists w/o html
        payload = json.dumps(job_dict, default=str, indent=2, sort_keys=True)
        return _atomic_write(json_path, payload.encode("utf-8"))
    except Exception:
        return False


def load(jobid: str) -> dict | None:
    """Read a stored job back; "report" points at the stored .html. Or None."""
    try:
        paths = _paths(str(jobid))
        if paths is None:
            return None
        json_path, html_path = paths
        with open(json_path, "r", encoding="utf-8") as f:
            job = json.load(f)
        if not isinstance(job, dict):
            return None
        if os.path.isfile(html_path):
            job["report"] = html_path   # repoint at the persisted copy
        return job
    except Exception:
        return None


def load_all() -> dict:
    """{jobid: job_dict} for every stored entry; corrupt files are skipped."""
    out: dict[str, dict] = {}
    try:
        names = os.listdir(_store_dir())
    except OSError:
        return out
    for name in names:
        if not name.endswith(_JSON_EXT):
            continue
        jobid = name[: -len(_JSON_EXT)]
        job = load(jobid)
        if job is not None:
            out[jobid] = job
    return out


def sweep(ttl_s: float = _DEFAULT_TTL_S) -> int:
    """Delete stored jobs older than ttl_s; returns the number removed.

    Age is taken from the job's own "created" timestamp (survives restarts),
    falling back to the file's mtime when the json lacks or loses that key.
    Each job removes both its .json and its .html; failures are skipped.
    """
    removed = 0
    now = time.time()
    try:
        names = os.listdir(_store_dir())
    except OSError:
        return 0
    for name in names:
        if not name.endswith(_JSON_EXT):
            continue
        jobid = name[: -len(_JSON_EXT)]
        json_path = os.path.join(_store_dir(), name)
        html_path = os.path.join(_store_dir(), jobid + _HTML_EXT)
        try:
            created = None
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    created = json.load(f).get("created")
            except Exception:
                pass                    # corrupt json → fall back to mtime
            if not isinstance(created, (int, float)):
                created = os.path.getmtime(json_path)
            if now - float(created) <= ttl_s:
                continue
            try:
                os.unlink(json_path)
            except OSError:
                continue
            try:
                os.unlink(html_path)
            except OSError:
                pass
            removed += 1
        except Exception:
            continue
    return removed


def status_of(jobid: str) -> dict | None:
    """Cheap {status,verdict,confidence,claim_time,claim_place} read, or None."""
    job = load(jobid)
    if job is None:
        return None
    return {
        "status": job.get("status"),
        "verdict": job.get("verdict"),
        "confidence": job.get("confidence"),
        "claim_time": job.get("claim_time"),
        "claim_place": job.get("claim_place"),
    }


# ---------------------------------------------------------------------------
# self-test: python3 sundial/job_store.py

if __name__ == "__main__":
    import sys
    import tempfile

    tmp_store = tempfile.mkdtemp(prefix="sundial_store_")
    tmp_src = tempfile.mkdtemp(prefix="sundial_src_")
    os.environ["SUNDIAL_JOB_DIR"] = tmp_store

    mod = sys.modules[__name__]
    assert mod.STORE_DIR == _store_dir() == tmp_store, "lazy STORE_DIR"

    # -- round-trip save/load -------------------------------------------------
    report_src = os.path.join(tmp_src, "report.html")
    with open(report_src, "w", encoding="utf-8") as f:
        f.write("<html>verdict report</html>")

    job = {
        "status": "done",
        "created": time.time(),
        "dir": tmp_src,
        "report": report_src,
        "verdict": "consistent",
        "confidence": 0.83,
        "error": None,
        "claim_time": "2024-06-01 12:00",
        "claim_place": "Lisbon",
        "weird": object(),              # non-serializable → default=str
    }
    assert save("abc123", job) is True, "save returned False"
    assert os.path.isfile(os.path.join(tmp_store, "abc123.json"))
    assert os.path.isfile(os.path.join(tmp_store, "abc123.html"))

    back = load("abc123")
    assert back is not None and back["verdict"] == "consistent"
    assert back["report"] == os.path.join(tmp_store, "abc123.html")
    assert isinstance(back["weird"], str), "default=str sanitize failed"
    print("ok  save/load round-trip + report copy + default=str")

    # -- status_of ------------------------------------------------------------
    st = status_of("abc123")
    assert st == {"status": "done", "verdict": "consistent",
                  "confidence": 0.83, "claim_time": "2024-06-01 12:00",
                  "claim_place": "Lisbon"}, st
    print("ok  status_of")

    # -- load_all + corrupt file skipped --------------------------------------
    with open(os.path.join(tmp_store, "broken.json"), "w") as f:
        f.write("{not valid json")
    assert load("broken") is None, "corrupt json should load as None"
    all_jobs = load_all()
    assert "abc123" in all_jobs and "broken" not in all_jobs, all_jobs.keys()
    print("ok  load_all skips corrupt files")

    # -- missing job → None ----------------------------------------------------
    assert load("nope") is None and status_of("nope") is None
    assert save("bad/name", job) is False, "path traversal must be refused"
    print("ok  missing job → None; unsafe jobid refused")

    # -- sweep removes expired, keeps fresh ------------------------------------
    expired = dict(job, created=time.time() - 10_000)
    assert save("oldjob", expired) is True
    n = sweep(ttl_s=7200)
    assert n == 1 and load("oldjob") is None and load("abc123") is not None
    # html of the expired job must be gone too
    assert not os.path.exists(os.path.join(tmp_store, "oldjob.html"))
    n2 = sweep(ttl_s=0)                  # 0-ttl clears everything left
    assert n2 >= 1 and load_all() == {}
    print("ok  sweep removed expired entries (json + html)")

    shutil.rmtree(tmp_store, ignore_errors=True)
    shutil.rmtree(tmp_src, ignore_errors=True)
    print("self-test passed")
