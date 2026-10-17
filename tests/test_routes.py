"""Route-level tests for the sundial demo server.

System under test::

    python3 -m sundial.serve --port <scratch>

Runnable two ways — pytest is NOT required::

    python3 -m pytest tests/test_routes.py      # pytest collects TestCases
    python3 tests/test_routes.py                # stdlib unittest runner

The server is spawned once per module (setUpModule) on a free scratch port
and killed in tearDownModule. Stdlib only: http.client, subprocess, threading.

The upload-flow test posts a real (tiny, valid) PNG — the pipeline is
failure-isolated, so a still image with no claim reaches ``status=done``
offline in a few seconds and produces a report.html we can fetch.
"""
from __future__ import annotations

import base64
import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(tempfile.gettempdir(),
                        f"sundial_test_routes_{os.getpid()}.log")

# A real 64x48 PNG (161 bytes) — valid image bytes so the still-image path
# in the pipeline is exercised end to end.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwCAIAAAAuKetIAAAAaElEQVR4nNXOQREAI"
    "AzAsFI1aEITohGxB9coyLpnUyZxEidxEidxEidxEidxEidxEidxEidxEidxEidxE"
    "idxEidxEidxEidxEidxEidxEidxEidxEidxEidxEufvwNQDGLsBbp2YtrsAAAAA"
    "SUVORK5CYII="
)
TINY_PNG = base64.b64decode(_PNG_B64)

_PROC: subprocess.Popen | None = None
_PORT: int = 0
_BASE: str = ""
_LOG_FH = None
_STATUS_DEADLINE_S = 180.0   # generous: first job pays module-import cost

# Hosts legitimately referenced from a rendered report (map tiles + the SVG
# xmlns namespace identifier, which is a URI constant, not a fetch).
_ALLOWED_EXTERNAL = (
    "basemaps.cartocdn.com",
    "tile.openstreetmap.org",
    "www.w3.org",
)


# ---------------------------------------------------------------------------
# server lifecycle

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ensure_server() -> None:
    global _PROC, _PORT, _BASE, _LOG_FH
    if _PROC is not None and _PROC.poll() is None:
        return                      # already running (pytest may call twice)
    err = None
    for _ in range(3):              # retry — a freed port can be re-taken
        _PORT = _free_port()
        _BASE = f"http://127.0.0.1:{_PORT}"
        _LOG_FH = open(LOG_PATH, "wb")
        _PROC = subprocess.Popen(
            [sys.executable, "-m", "sundial.serve", "--port", str(_PORT)],
            cwd=ROOT, stdout=_LOG_FH, stderr=subprocess.STDOUT,
        )
        err = _wait_ready(30)
        if err is None:
            return
        _stop_server()
    raise RuntimeError(err or "sundial server did not start")


def _wait_ready(timeout_s: float) -> str | None:
    """Poll until GET / answers 200; -> error string or None."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _PROC.poll() is not None:
            return (f"sundial server exited early (rc={_PROC.returncode})\n"
                    + _log_tail())
        try:
            status, _, _ = _req("GET", "/")
            if status == 200:
                return None
        except OSError:
            pass
        time.sleep(0.2)
    return "sundial server did not answer within 30s\n" + _log_tail()


def _stop_server() -> None:
    global _PROC, _LOG_FH
    if _PROC is None:
        return
    if _PROC.poll() is None:
        # SIGINT first — the server's KeyboardInterrupt path runs atexit,
        # which wipes the sundial_* job tempdirs.
        try:
            _PROC.send_signal(2)  # signal.SIGINT, portable literal
            _PROC.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            _PROC.terminate()
            try:
                _PROC.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _PROC.kill()
                _PROC.wait(timeout=5)
    _PROC = None
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def setUpModule() -> None:
    _ensure_server()


def tearDownModule() -> None:
    _stop_server()


# nose/pytest-style aliases — pytest calls these too; both are idempotent.
def setup_module() -> None:
    _ensure_server()


def teardown_module() -> None:
    _stop_server()


def _log_tail(n: int = 40) -> str:
    try:
        if _LOG_FH is not None:
            _LOG_FH.flush()
        with open(LOG_PATH, "rb") as f:
            lines = f.read().decode("utf-8", "replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError as e:
        return f"<could not read server log: {e}>"


# ---------------------------------------------------------------------------
# http helpers — http.client gives raw status/headers, never follows redirects

def _req(method: str, path: str, body: bytes | None = None,
         headers: dict | None = None, timeout: float = 30.0):
    """-> (status:int, headers:dict, body:bytes). Never raises on 4xx/5xx."""
    conn = http.client.HTTPConnection("127.0.0.1", _PORT, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, hdrs, data
    finally:
        conn.close()


def _get(path: str, **kw):
    return _req("GET", path, **kw)


def _multipart(fields: list[tuple[str, str]],
               files: list[tuple[str, str, str, bytes]]):
    """Build a multipart/form-data body.

    fields: [(name, value)]
    files:  [(name, filename, content_type, data)]
    -> (body_bytes, content_type_header)
    """
    boundary = "----sundialtest-" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields:
        out += (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n").encode("utf-8")
    for name, fname, ctype, data in files:
        out += (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{fname}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n").encode("utf-8")
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode("utf-8")
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _post_verify(fields=None, files=None, extra_headers=None):
    body, ctype = _multipart(fields or [], files or [])
    headers = {"Content-Type": ctype, "Accept": "application/json"}
    headers.update(extra_headers or {})
    return _req("POST", "/verify", body=body, headers=headers, timeout=60)


def _upload_png(fields=None):
    """POST a valid PNG upload; -> (status, headers, parsed_json_or_None)."""
    f = [("claim", "")] if fields is None else fields
    st, hd, body = _post_verify(
        fields=f,
        files=[("video", "clip.png", "image/png", TINY_PNG)])
    try:
        return st, hd, json.loads(body)
    except (ValueError, TypeError):
        return st, hd, None


def _poll_status(jobid: str, deadline_s: float = _STATUS_DEADLINE_S) -> dict:
    """Poll /status/<jobid> until done|error or deadline; -> last payload."""
    deadline = time.time() + deadline_s
    last: dict = {}
    while time.time() < deadline:
        st, _, body = _get(f"/status/{jobid}")
        if st == 200:
            try:
                last = json.loads(body)
            except ValueError:
                last = {}
            if last.get("status") in ("done", "error"):
                return last
        time.sleep(1.0)
    return last


def _fresh_jobid() -> str:
    """A well-formed (16 hex chars) but nonexistent jobid."""
    return secrets.token_hex(8)


# ---------------------------------------------------------------------------
# tests

class TestStaticRoutes(unittest.TestCase):
    """GET /, the JS asset, unknown paths, HEAD."""

    def test_index_200_upload_form(self):
        st, hd, body = _get("/")
        self.assertEqual(200, st)
        self.assertIn("text/html", hd.get("content-type", ""))
        # upload form posts to /verify as multipart
        self.assertIn(b'<form id="vf" method="post" action="/verify"', body)
        self.assertIn(b"multipart/form-data", body)
        # claim field present
        self.assertIn(b'name="claim"', body)
        # file input + drop zone (pb-field has the drag/drop handlers)
        self.assertIn(b'type="file"', body)
        self.assertIn(b'id="pbf"', body)
        self.assertIn(b"drop a file", body)
        self.assertIn(b"dataTransfer", body)   # drop handler wired up

    def test_index_html_alias(self):
        st, _, body = _get("/index.html")
        self.assertEqual(200, st)
        self.assertIn(b'action="/verify"', body)

    def test_sliced_waves_js(self):
        st, hd, body = _get("/sliced-waves.js")
        self.assertEqual(200, st)
        self.assertIn("javascript", hd.get("content-type", ""))
        self.assertGreater(len(body), 10 * 1024, "JS asset should be >10KB")
        head = body.lstrip()[:400]
        # starts with a comment or code, not an error page
        self.assertFalse(head.startswith(b"<"), "JS body looks like HTML")
        self.assertIn(b"SlicedWaves", body[:4096])

    def test_unknown_path_404_page(self):
        st, hd, body = _get("/nonsense")
        self.assertEqual(404, st)
        self.assertIn("text/html", hd.get("content-type", ""))
        # fully rendered page (has doctype/body), not a bare error
        self.assertIn(b"<!DOCTYPE html>", body)
        # back-link to the upload page
        self.assertIn(b'href="/"', body)
        self.assertIn(b"Back to upload", body)

    def test_head_index(self):
        st, hd, body = _req("HEAD", "/")
        self.assertEqual(200, st)
        self.assertEqual(b"", body)
        self.assertIn("content-length", hd)


class TestJobLookupRoutes(unittest.TestCase):
    """/status/<id> and /result/<id> for unknown/malformed ids."""

    def test_status_unknown_404_json(self):
        st, hd, body = _get(f"/status/{_fresh_jobid()}")
        self.assertEqual(404, st)
        self.assertIn("application/json", hd.get("content-type", ""))
        payload = json.loads(body)
        self.assertEqual("error", payload.get("status"))
        self.assertTrue(payload.get("error"))

    def test_status_unknown_trailing_slash(self):
        st, _, body = _get(f"/status/{_fresh_jobid()}/")
        self.assertEqual(404, st)
        self.assertEqual("error", json.loads(body).get("status"))

    def test_result_unknown_graceful(self):
        st, hd, body = _get(f"/result/{_fresh_jobid()}")
        self.assertEqual(404, st)
        self.assertIn("text/html", hd.get("content-type", ""))
        self.assertIn(b"expired", body.lower())
        self.assertIn(b'href="/"', body)          # link back to upload
        self.assertNotIn(b"Traceback", body)

    def test_result_unknown_trailing_slash(self):
        st, hd, body = _get(f"/result/{_fresh_jobid()}/")
        self.assertEqual(404, st)
        self.assertIn(b"expired", body.lower())

    def test_result_malformed_ids_no_crash(self):
        # every one of these must produce a clean 4xx, never a 500/traceback
        paths = [
            "/result//",
            "/result/%20",
            "/result/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "/result/%00%00%00%00%00%00%00%00",
            "/result/" + "a" * 200,
            "/result/XYZWXYZWXYZWXYZW",          # 16 chars but not hex
            "/result/abcdef0123456789/extra",    # valid id + extra segment
            "/status//",
            "/status/%20",
            "/status/" + "f" * 15,               # one char short
        ]
        for p in paths:
            with self.subTest(path=p):
                st, _, body = _req("GET", p)
                self.assertIn(st, (400, 404),
                              f"{p} -> {st} (expected clean 4xx)")
                self.assertNotIn(b"Traceback", body)
                self.assertNotIn(b"File \"/", body)  # no leaked source paths

    def test_result_traversal_path(self):
        # ../ in the raw request path (http.client does not normalize it)
        st, _, body = _req("GET", "/result/../etc/passwd")
        self.assertIn(st, (400, 404))
        self.assertNotIn(b"Traceback", body)
        self.assertNotIn(b"root:", body)         # no file content leaked


class TestPostRoutes(unittest.TestCase):
    """POST /verify validation and the 404 JSON for POSTs elsewhere."""

    def test_post_to_get_only_routes_json_404(self):
        for p in ("/", "/nonsense", "/sliced-waves.js",
                  f"/status/{_fresh_jobid()}", f"/result/{_fresh_jobid()}"):
            with self.subTest(path=p):
                st, hd, body = _req("POST", p, body=b"x=1",
                                    headers={"Content-Type":
                                             "application/x-www-form-urlencoded"})
                self.assertEqual(404, st)
                self.assertIn("application/json", hd.get("content-type", ""))
                self.assertIn("error", json.loads(body))

    def test_verify_empty_body(self):
        st, _, body = _req("POST", "/verify", body=b"",
                           headers={"Content-Type": "multipart/form-data"})
        self.assertEqual(400, st)
        self.assertIn("error", json.loads(body))

    def test_verify_no_file_field(self):
        st, _, body = _post_verify(fields=[("claim", "test")], files=[])
        self.assertEqual(400, st)
        payload = json.loads(body)
        self.assertIn("error", payload)
        self.assertIn("video", payload["error"])

    def test_verify_empty_file(self):
        # a file part that exists but carries zero bytes
        st, _, body = _post_verify(
            fields=[], files=[("video", "clip.png", "image/png", b"")])
        self.assertEqual(400, st)
        self.assertIn("error", json.loads(body))

    def test_verify_wrong_content_type_urlencoded(self):
        st, hd, body = _req(
            "POST", "/verify", body=b"claim=hello&x=1",
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(400, st)
        self.assertIn("application/json", hd.get("content-type", ""))
        self.assertIn("error", json.loads(body))

    def test_verify_wrong_content_type_json(self):
        st, _, body = _req("POST", "/verify", body=b'{"claim":"x"}',
                           headers={"Content-Type": "application/json"})
        self.assertEqual(400, st)
        self.assertIn("error", json.loads(body))

    def test_verify_multipart_missing_boundary(self):
        st, _, body = _req(
            "POST", "/verify", body=b"--garbage--",
            headers={"Content-Type": "multipart/form-data"})  # no boundary=
        self.assertEqual(400, st)
        self.assertIn("error", json.loads(body))

    def test_verify_html_accept_redirects(self):
        # A plain (non-fetch) browser form post gets a 303 to the result page.
        st, hd, _ = _post_verify(
            fields=[("claim", "")],
            files=[("video", "clip.png", "image/png", TINY_PNG)],
            extra_headers={"Accept": "text/html"})
        self.assertEqual(303, st)
        loc = hd.get("location", "")
        self.assertRegex(loc, r"^/result/[0-9a-f]{16}$")


class TestVerifyFlow(unittest.TestCase):
    """The whole upload -> status -> result pipeline, incl. trailing slash."""

    def _new_job(self) -> str:
        st, _, payload = _upload_png()
        self.assertEqual(202, st, f"verify -> {st}")
        self.assertIsNotNone(payload, "verify response is not JSON")
        jobid = payload.get("jobid", "")
        self.assertRegex(jobid, r"^[0-9a-f]{16}$")
        self.assertEqual(f"/status/{jobid}", payload.get("status_url"))
        self.assertEqual(f"/result/{jobid}", payload.get("result_url"))
        return jobid

    def test_upload_status_result_roundtrip(self):
        jobid = self._new_job()

        # running job: status shape + the non-JS "analyzing" result page
        st, _, body = _get(f"/status/{jobid}")
        self.assertEqual(200, st)
        snap = json.loads(body)
        for key in ("jobid", "status", "elapsed_s", "claim", "result_url"):
            self.assertIn(key, snap)
        if snap["status"] == "running":          # may already be done
            st2, _, body2 = _get(f"/result/{jobid}")
            self.assertEqual(200, st2)           # graceful "analyzing" page
            self.assertIn(b"<!DOCTYPE html>", body2)

        final = _poll_status(jobid)
        self.assertEqual("done", final.get("status"),
                         f"job never finished: {final}")
        self.assertIn("verdict", final)

        # the trailing-slash regression: BOTH spellings must serve the report
        st_a, hd_a, body_a = _get(f"/result/{jobid}")
        st_b, hd_b, body_b = _get(f"/result/{jobid}/")
        self.assertEqual(200, st_a, "GET /result/<id> failed")
        self.assertEqual(200, st_b, "GET /result/<id>/ (trailing slash) failed")
        self.assertEqual(body_a, body_b, "trailing-slash body differs")

        # self-contained HTML
        self.assertIn(b"<!DOCTYPE", body_a[:200])
        self.assertIn(b"</html>", body_a)
        import re
        # flag only well-formed absolute refs — the slippy-map fragment
        # builds tile URLs via JS string concat (src="https://' + SUB ...)
        # which must not be mistaken for a static external reference.
        hosts = set(re.findall(
            rb'(?:src|href)\s*=\s*"https?://([a-zA-Z0-9.\-]+)', body_a))
        for host in hosts:
            self.assertTrue(
                any(host.endswith(h.encode()) for h in _ALLOWED_EXTERNAL),
                f"external (non-tile) reference in report, host {host!r}")
        # no CDN script/link tags at all
        self.assertNotIn(b'<script src="http', body_a)
        self.assertNotIn(b'<link rel="stylesheet" href="http', body_a)

    def test_result_while_running_then_done(self):
        jobid = self._new_job()
        # immediately after POST the result URL must not error out
        st, _, body = _get(f"/result/{jobid}")
        self.assertIn(st, (200,), f"early /result -> {st}")
        self.assertNotIn(b"Traceback", body)
        final = _poll_status(jobid)
        self.assertEqual("done", final.get("status"), f"final: {final}")


class TestConcurrency(unittest.TestCase):
    """Simultaneous uploads must not collide on jobids."""

    def test_three_simultaneous_verify_distinct_jobids(self):
        barrier = threading.Barrier(3)
        results: list[dict] = []

        def worker():
            barrier.wait(timeout=15)
            st, _, payload = _upload_png()
            results.append({"status": st, "payload": payload})

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        self.assertEqual(3, len(results))
        jobids = []
        for r in results:
            self.assertEqual(202, r["status"], f"verify -> {r['status']}")
            self.assertIsNotNone(r["payload"])
            jobids.append(r["payload"].get("jobid"))
        self.assertEqual(3, len(set(jobids)),
                         f"jobids not distinct: {jobids}")
        # drain the three jobs so teardown doesn't race running workers
        for j in jobids:
            _poll_status(j)


class TestZzServerLog(unittest.TestCase):
    """Runs last alphabetically — the server must not have crashed anywhere."""

    def test_server_log_has_no_tracebacks(self):
        self.assertIsNotNone(_PROC)
        self.assertIsNone(_PROC.poll(), "server process died during tests")
        log = _log_tail(n=10000)
        self.assertNotIn("Traceback (most recent call last)", log,
                         "server logged a traceback:\n" + _log_tail(60))


# ---------------------------------------------------------------------------
# direct run

if __name__ == "__main__":
    unittest.main(verbosity=2)
