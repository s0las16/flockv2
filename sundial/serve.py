"""Sundial demo server — upload a clip + claim, get back a verdict card.

  python3 -m sundial.serve --port 8765     # then open http://localhost:8765

Stdlib only. pipeline/report are imported lazily inside the job worker, so
`import sundial.serve` never touches third-party deps. Multipart bodies are
parsed with the `email` package (cgi is gone in Python 3.13).

Flow: POST /verify stores the clip in a tempdir and answers {"jobid": ...}
immediately; the page polls GET /status/<jobid> while pipeline.run() works in
a daemon thread (a real run takes minutes), then the browser lands on
GET /result/<jobid>, which serves the generated report.
"""
from __future__ import annotations

import argparse
import atexit
import html
import json
import math
import os
import secrets
import shutil
import sys
import tempfile
import threading
import time
from email.parser import BytesParser
from email.policy import default as _EMAIL_POLICY
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_MAX_UPLOAD = 1 << 30          # 1 GiB request cap
_JOB_TTL_S = 2 * 60 * 60       # results live this long, then tempdirs are wiped
_JOB_ID_LEN = 16               # secrets.token_hex(8)

_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# pages — same aesthetic as report.py: dark navy #070b14, dotted drift, mono

_CSS = """
*{box-sizing:border-box}
body{font-family:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace;
 background:#000;color:#f5f5f5;margin:0;min-height:100vh;
 display:flex;flex-direction:column}
#bg{position:fixed;inset:0;z-index:0;pointer-events:none}
.wrap{flex:1;width:100%;max-width:520px;margin:0 auto;
 padding:0 24px;display:flex;flex-direction:column;justify-content:center;
 position:relative;z-index:1;min-height:100vh}
.brand{font-size:15px;letter-spacing:.6em;color:#fff;text-transform:uppercase;
 margin-bottom:56px}
.card{position:relative}
h1{display:none}
.sub{display:none}
input[type=file]{position:absolute;width:1px;height:1px;opacity:0;
 overflow:hidden;clip:rect(0 0 0 0)}
a{color:#999}

/* prompt-bar */
.pb{width:100%}
.pb-field{background:#141414;border:1px solid #262626;border-radius:16px;
 padding:14px 14px 10px;transition:border-color .2s;cursor:text}
.pb-field:focus-within{border-color:#4a4a4a}
.pb-field.over{border-color:#777;border-style:dashed}
.pb-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:2px}
.pb-chips:empty{display:none}
.pb-chip{display:inline-flex;align-items:center;gap:7px;height:26px;
 padding:0 4px 0 9px;border-radius:8px;background:#262626;font-size:12px;
 color:#d4d4d4}
.pb-chip svg{flex:none;opacity:.7}
.pb-chip-name{max-width:200px;overflow:hidden;white-space:nowrap;
 text-overflow:ellipsis}
.pb-chip-x{display:inline-grid;place-items:center;width:18px;height:18px;
 padding:0;border:0;border-radius:5px;background:transparent;color:inherit;
 opacity:.6;cursor:pointer}
.pb-chip-x:hover{opacity:1;background:#3a3a3a}
.pb-in{width:100%;border:0;background:transparent;color:#f5f5f5;
 font:inherit;font-size:17px;line-height:24px;resize:none;outline:none;
 min-height:26px;max-height:120px}
.pb-in::placeholder{color:#5c5c5c}
.pb-bar{display:flex;align-items:center;gap:6px;margin-top:8px}
.pb-tool{display:inline-grid;place-items:center;width:30px;height:30px;
 padding:0;border:0;border-radius:8px;background:transparent;color:#8a8a8a;
 cursor:pointer;transition:background .15s,color .15s}
.pb-tool:hover{background:#262626;color:#f5f5f5}
.pb-hint{font-size:12px;color:#4f4f4f;letter-spacing:.03em}
.pb-spacer{flex:1}
.pb-send{display:inline-grid;place-items:center;width:30px;height:30px;
 padding:0;border:0;border-radius:8px;background:#262626;color:#666;
 cursor:default;transition:background .18s,color .18s,transform .15s}
.pb-send[data-armed]{background:#f5f5f5;color:#000;cursor:pointer}
.pb-send[data-armed]:active{transform:scale(.94)}

/* progress */
.prog{display:none;margin-top:22px;text-align:center}
.spin{width:32px;height:32px;border-radius:50%;margin:0 auto 14px;
 border:2px solid #1f1f1f;border-top-color:#f5f5f5;
 animation:rot .8s linear infinite}
@keyframes rot{to{transform:rotate(360deg)}}
.ptxt{font-size:13px;color:#999;letter-spacing:.05em}
.ptxt b{color:#f5f5f5;font-variant-numeric:tabular-nums;font-weight:400}
.step{font-size:11px;color:#555;margin-top:7px;letter-spacing:.12em}
.err{color:#b66;font-size:13px;margin-top:14px;min-height:1em}
button{font:inherit}
"""

_UPLOAD_BODY = """
<div class="card pb">
  <form id="vf" method="post" action="/verify" enctype="multipart/form-data">
    <input type="file" id="video" name="video" accept="video/*,image/*">
    <div class="pb-field" id="pbf">
      <div class="pb-chips" id="chips"></div>
      <textarea class="pb-in" id="claim" name="claim" rows="1"
        placeholder="filmed tuesday 3pm in london &hellip; or ask where it was"
        autocomplete="off"></textarea>
      <div class="pb-bar">
        <button type="button" class="pb-tool" id="attach"
          aria-label="attach a file">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" stroke-width="2" stroke-linecap="round"
            stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0
            1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0
            1-2.83-2.83l8.49-8.48"/></svg>
        </button>
        <span class="pb-hint" id="hint">drop a file</span>
        <span class="pb-spacer"></span>
        <button type="submit" class="pb-send" id="go" aria-label="verify">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
            stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/>
            <polyline points="5 12 12 5 19 12"/></svg>
        </button>
      </div>
    </div>
    <div class="prog" id="prog">
      <div class="spin"></div>
      <div class="ptxt"><b id="elapsed">0:00</b></div>
      <div class="step" id="step"></div>
    </div>
    <div class="err" id="err"></div>
  </form>
</div>
<script>
(function() {
  var form = document.getElementById("vf"), btn = document.getElementById("go"),
      prog = document.getElementById("prog"), err = document.getElementById("err"),
      elapsed = document.getElementById("elapsed"),
      step = document.getElementById("step"),
      field = document.getElementById("pbf"),
      fi = document.getElementById("video"),
      chips = document.getElementById("chips"),
      hint = document.getElementById("hint"),
      input = document.getElementById("claim");
  var STEPS = ["extracting frames", "asking the VLM about shadows & sign text",
               "cataloguing OSINT scene cues", "scoring candidate countries",
               "reading burned-in clock overlays", "geocoding sign text",
               "checking landmarks, weather & climate archives",
               "computing sun position", "adjudicating the evidence"];

  // textarea auto-grow, prompt-bar style
  var grow = function() {
    input.style.height = "0px";
    var h = Math.min(input.scrollHeight, 120);
    input.style.height = h + "px";
    input.style.overflowY = input.scrollHeight > 120 ? "auto" : "hidden";
  };
  input.addEventListener("input", grow);

  // send tile arms only with a file attached
  var arm = function() {
    if (fi.files && fi.files.length) btn.setAttribute("data-armed", "");
    else btn.removeAttribute("data-armed");
  };
  arm();

  var showFile = function() {
    chips.innerHTML = "";
    if (fi.files && fi.files.length) {
      var chip = document.createElement("span");
      chip.className = "pb-chip";
      chip.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" '
        + 'fill="none" stroke="currentColor" stroke-width="2"><path d="M13 '
        + '2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>'
        + '<polyline points="13 2 13 9 20 9"/></svg>'
        + '<span class="pb-chip-name"></span>'
        + '<button type="button" class="pb-chip-x">&times;</button>';
      chip.querySelector(".pb-chip-name").textContent = fi.files[0].name;
      chip.querySelector(".pb-chip-x").addEventListener("click", function(e) {
        e.stopPropagation();
        fi.value = ""; showFile(); arm();
      });
      chips.appendChild(chip);
      hint.textContent = "file ready";
    } else {
      hint.textContent = "drop a file";
    }
  };
  document.getElementById("attach").addEventListener("click", function() {
    fi.click();
  });
  fi.addEventListener("change", function() { showFile(); arm(); });

  // drag & drop anywhere on the field
  ["dragover","dragenter"].forEach(function(ev) {
    field.addEventListener(ev, function(e) {
      e.preventDefault(); field.classList.add("over");
    });
  });
  ["dragleave","drop"].forEach(function(ev) {
    field.addEventListener(ev, function(e) {
      e.preventDefault(); field.classList.remove("over");
    });
  });
  field.addEventListener("drop", function(e) {
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      fi.files = e.dataTransfer.files;
      fi.dispatchEvent(new Event("change"));
    }
  });
  field.addEventListener("click", function(e) {
    if (e.target === field) input.focus();
  });

  form.addEventListener("submit", function(e) {
    e.preventDefault();
    err.textContent = "";
    if (!fi.files || !fi.files.length) {
      err.textContent = "attach a file first";
      return;
    }
    btn.disabled = true;
    prog.style.display = "block";
    var t0 = Date.now();
    var tick = setInterval(function() {
      var s = Math.floor((Date.now() - t0) / 1000);
      elapsed.textContent =
          Math.floor(s / 60) + ":" + ("0" + (s % 60)).slice(-2);
      step.textContent =
          STEPS[Math.min(Math.floor(s / 18), STEPS.length - 1)] + "\\u2026";
    }, 1000);
    var fail = function(msg) {
      clearInterval(tick);
      btn.disabled = false;
      prog.style.display = "none";
      err.textContent = msg;
    };
    fetch("/verify", {method: "POST", body: new FormData(form)})
      .then(function(r) {
        return r.json().then(function(j) { return {ok: r.ok, j: j}; });
      })
      .then(function(res) {
        if (!res.ok) { fail(res.j.error || ("HTTP " + res.status)); return; }
        var poll = setInterval(function() {
          fetch("/status/" + res.j.jobid)
            .then(function(r) { return r.json(); })
            .then(function(s) {
              if (s.status === "done") {
                clearInterval(poll);
                clearInterval(tick);
                window.location = s.result_url;
              } else if (s.status === "error") {
                clearInterval(poll);
                fail(s.error || "analysis failed");
              }
            })
            .catch(function() { /* transient — keep polling */ });
        }, 2500);
      })
      .catch(function(e) { fail(String(e)); });
  });
})();
</script>
"""

_WAIT_BODY = """
<div class="card" style="text-align:center">
  <div class="spin"></div>
  <h1 style="font-size:16px">analysis in progress</h1>
  <p class="sub">job <b>@JOBID@</b> is running &mdash; this page refreshes
  itself when the verdict is ready.</p>
  <div class="step" id="wel" style="margin-top:10px"></div>
  <div class="err" id="werr"></div>
</div>
<script>
setInterval(function() {
  fetch("/status/@JOBID@")
    .then(function(r) { return r.json(); })
    .then(function(s) {
      var el = document.getElementById("wel");
      if (typeof s.elapsed_s === "number")
        el.textContent = "elapsed " + Math.round(s.elapsed_s) + "s";
      if (s.status === "done") location.reload();
      else if (s.status === "error")
        document.getElementById("werr").textContent =
            "analysis failed: " + (s.error || "unknown error");
    })
    .catch(function() {});
}, 3000);
</script>
"""

_LOST_BODY = """
<div class="card" style="text-align:center">
  <h1 style="font-size:16px">job unknown or expired</h1>
  <p class="sub">results live in memory for a couple of hours and vanish with
  the server. <a href="/">Submit a new clip &rarr;</a></p>
</div>
"""


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


# SlicedWaves background — sundial palette (slate / sun-amber / sky). The
# waves sit low on the page so they read as a horizon behind the form.
_WAVES_INIT = """
<script src="/sliced-waves.js"></script>
<script>
(function() {
  if (!window.SlicedWaves) return;
  var waves = window.SlicedWaves.init(document.getElementById("bg"), {
    color1: "#2a2a2a", color2: "#e5e5e5", color3: "#6b6b6b",
    ribbons: 3, spacing: 34, perspective: 0.5, curve: 0.55,
    blur: 0.8, glow: 0.9, brightness: 0.55, position: 0.82,
    amplitude: 0.1, frequency: 0.6, spread: 0.1,
    grain: 0.05, fade: 0.4, opacity: 0.7, mouseInteraction: true
  });
  if (!waves) { document.getElementById("bg").style.display = "none"; return; }

  // typing couples the waves: drift slows and the lift/focus pulls under
  // the prompt field — the bars bend toward where you're working
  var input = document.getElementById("claim"),
      field = document.getElementById("pbf");
  if (!input || !field) return;
  var pull = function() {
    var r = field.getBoundingClientRect();
    waves.setPointer(r.left + r.width / 2, r.top + r.height / 2, true);
    waves.setSpeed(0.22);
  };
  var release = function() {
    if (document.activeElement === input) return;
    waves.clearPointer();
    waves.setSpeed(1);
  };
  input.addEventListener("focus", pull);
  input.addEventListener("input", pull);
  input.addEventListener("blur", release);
  field.addEventListener("pointerenter", pull);
  field.addEventListener("pointerleave", release);
})();
</script>
"""


def _page(title: str, body: str) -> bytes:
    doc = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<link rel="icon" href="data:,">'
        f"<title>Sundial — {_e(title)}</title>"
        f'<style>{_CSS}</style></head><body>'
        '<div id="bg"></div><div class="wrap">'
        '<div class="brand">sundial</div>'
        f"{body}"
        "</div>"
        f"{_WAVES_INIT}"
        "</body></html>"
    )
    return doc.encode("utf-8")


_INDEX = _page("verify a clip", _UPLOAD_BODY)


# ---------------------------------------------------------------------------
# multipart/form-data without cgi: wrap the raw body in a synthetic MIME
# message and let the stdlib email package split the parts.

def _parse_multipart(body: bytes, content_type: str) -> tuple[dict, dict]:
    """-> (fields, files); files maps field name -> (filename, bytes)."""
    head = ("MIME-Version: 1.0\r\nContent-Type: " + content_type +
            "\r\n\r\n").encode("latin-1", "replace")
    msg = BytesParser(policy=_EMAIL_POLICY).parsebytes(head + body)
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    if not msg.is_multipart():
        return fields, files
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()  # "" when a file input was left empty
        if filename:
            files[name] = (filename, payload)
        else:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, files


# ---------------------------------------------------------------------------
# job lifecycle

def _valid_jobid(jobid: str) -> bool:
    return (len(jobid) == _JOB_ID_LEN
            and all(c in "0123456789abcdef" for c in jobid))


def _get_job(jobid: str) -> dict | None:
    if not _valid_jobid(jobid):
        return None
    with _LOCK:
        job = _JOBS.get(jobid)
    if job is not None:
        return job
    try:  # persisted job from a previous server lifetime
        from . import job_store
        job = job_store.load(jobid)
        if job:
            with _LOCK:
                _JOBS[jobid] = job
            return job
    except Exception:
        pass
    return None


def _sweep() -> None:
    """Drop finished jobs older than the TTL and wipe their tempdirs."""
    now = time.time()
    with _LOCK:
        stale = [j for j, jb in _JOBS.items()
                 if now - jb["created"] > _JOB_TTL_S]
        for j in stale:
            jb = _JOBS.pop(j)
            shutil.rmtree(jb.get("dir", ""), ignore_errors=True)
    try:
        from . import job_store
        job_store.sweep(_JOB_TTL_S)
    except Exception:
        pass


def _cleanup_all() -> None:
    with _LOCK:
        jobs = list(_JOBS.values())
        _JOBS.clear()
    for jb in jobs:
        shutil.rmtree(jb.get("dir", ""), ignore_errors=True)


atexit.register(_cleanup_all)


def _report_extra(adj) -> dict:
    """Build the geo rail payload: location summary, dark static map, chain.

    Everything is best-effort — the report renders fine with any piece
    missing, so no failure here may break the job.
    """
    extra: dict = {}
    try:
        from . import locate_summary

        ev = list(getattr(getattr(adj, "pack", None), "evidence", []) or [])
        anchor = getattr(adj, "anchor", None)
        claim_place = getattr(adj, "claim_place", None)
        summ = locate_summary.summarize(ev, anchor, claim_place)
        if isinstance(summ, dict) and any(
                summ.get(k) for k in ("anchor", "predicted", "markers",
                                      "ranked", "best_cc", "best_name")):
            extra["loc_summary"] = summ
            # centre on where we think it was filmed, falling back to the claim
            center = summ.get("predicted") or summ.get("anchor")
            placeable = bool(center or summ.get("markers"))
            radius = (summ.get("predicted") or {}).get("radius_km") or (
                summ.get("predicted") or {}).get("r_km")
            zoom = 12 if not radius else max(
                2, min(12, round(8 - math.log2(max(radius, 30) / 300))))
            # Generate the offline vector SVG first so it can serve both as an
            # in-map "Vector" tab / offline fallback inside map_viz AND as a
            # standalone fallback if map_viz fails.
            svg_frag = ""
            if placeable:
                try:
                    from . import vec_map

                    svg_frag = vec_map.vec_fragment(summ, height="100%") or ""
                except Exception:
                    svg_frag = ""
                try:  # interactive slippy map fills the whole right side
                    from . import map_viz

                    frag = map_viz.map_fragment(
                        center, markers=summ.get("markers") or [],
                        bands=[], zoom=zoom, height="100%",
                        radius_km=radius, ring_center=summ.get("predicted"),
                        svg_fallback=svg_frag)
                    if frag:
                        extra["map_fragment"] = frag
                except Exception:
                    pass
                if "map_fragment" not in extra and svg_frag:
                    extra["map_svg"] = svg_frag
            if placeable and "map_svg" not in extra and "map_fragment" not in extra:
                try:  # static PNG last resort (needs PIL + tile network)
                    from . import map_tiles

                    img = map_tiles.static_map_b64(
                        center, markers=summ.get("markers") or [],
                        bands=summ.get("bands") or [], zoom=zoom)
                    if img:
                        extra["map_static"] = img
                except Exception:
                    pass
    except Exception:
        pass
    return extra


def _worker(jobid: str, video_path: str, claim_time: str | None,
            claim_place: str | None) -> None:
    """Run the pipeline off the request thread; results land in _JOBS."""
    job = _JOBS[jobid]
    try:
        from .pipeline import run          # lazy: heavy third-party deps
        from .report import render_html    # live in here, not at import time

        adj = run(
            video_path,
            claim_time=claim_time,
            claim_place=claim_place,
            workdir=os.path.join(job["dir"], "frames"),
        )
        render_html(adj, job["report"], extra=_report_extra(adj))
        with _LOCK:
            job["verdict"] = getattr(adj.verdict, "value", str(adj.verdict))
            job["confidence"] = float(adj.confidence)
            job["status"] = "done"
    except Exception as e:  # a demo must fail loud in JSON, not silently die
        with _LOCK:
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
    finally:  # persist result + report so restarts don't wipe results
        try:
            from . import job_store
            job_store.save(jobid, dict(job))
        except Exception:
            pass


def _start_job(data: bytes, filename: str, claim_time: str | None,
               claim_place: str | None) -> str:
    jobid = secrets.token_hex(8)
    jobdir = tempfile.mkdtemp(prefix=f"sundial_{jobid}_")
    ext = "".join(c for c in os.path.splitext(filename or "")[1]
                  if c.isalnum())[:8]
    video_path = os.path.join(jobdir, "clip" + ("." + ext if ext else ".mp4"))
    with open(video_path, "wb") as f:
        f.write(data)
    with _LOCK:
        _JOBS[jobid] = {
            "status": "running",
            "created": time.time(),
            "dir": jobdir,
            "report": os.path.join(jobdir, "report.html"),
            "verdict": None,
            "confidence": None,
            "error": None,
            "claim_time": claim_time,
            "claim_place": claim_place,
        }
    threading.Thread(
        target=_worker,
        args=(jobid, video_path, claim_time, claim_place),
        daemon=True,
        name=f"sundial-job-{jobid}",
    ).start()
    return jobid


# ---------------------------------------------------------------------------
# HTTP handler

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SundialDemo/1.0"

    def log_message(self, fmt, *args) -> None:
        sys.stderr.write("[sundial] %s\n" % (fmt % args))

    # -- plumbing -----------------------------------------------------------
    def _send(self, status: int, body=b"",
              ctype: str = "text/html; charset=utf-8",
              extra: dict | None = None) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            if self.command != "HEAD" and body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"),
                   "application/json; charset=utf-8",
                   {"Cache-Control": "no-store"})

    # -- GET -----------------------------------------------------------------
    def do_GET(self) -> None:
        _sweep()
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, _INDEX)
        elif path == "/sliced-waves.js":
            js_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "sliced_waves.js")
            try:
                with open(js_path, "rb") as f:
                    self._send(200, f.read(),
                               "application/javascript; charset=utf-8",
                               {"Cache-Control": "no-cache"})
            except OSError:
                self._send(404, b"", "application/javascript")
        elif path.startswith("/status/"):
            self._status(path[len("/status/"):].rstrip("/"))
        elif path.startswith("/result/"):
            self._result(path[len("/result/"):].rstrip("/"))
        else:
            self._send(404, _page(
                "not found",
                '<div class="card"><p class="err">404 — unknown path. '
                '<a href="/">Back to upload &rarr;</a></p></div>'))

    def do_HEAD(self) -> None:
        self.do_GET()

    def _status(self, jobid: str) -> None:
        job = _get_job(jobid)
        if job is None:
            self._json(404, {"status": "error",
                             "error": "unknown or expired job"})
            return
        payload = {
            "jobid": jobid,
            "status": job["status"],
            "elapsed_s": round(time.time() - job["created"], 1),
            "claim": {"time": job.get("claim_time"),
                      "place": job.get("claim_place")},
            "result_url": f"/result/{jobid}",
        }
        if job["status"] == "done":
            payload["verdict"] = job.get("verdict")
            payload["confidence"] = job.get("confidence")
        if job["status"] == "error":
            payload["error"] = job.get("error") or "analysis failed"
        self._json(200, payload)

    def _result(self, jobid: str) -> None:
        job = _get_job(jobid)
        if job is None:
            self._send(404, _page("expired", _LOST_BODY))
            return
        if job["status"] == "error":
            self._send(500, _page(
                "error",
                '<div class="card"><h1 style="font-size:16px">analysis '
                f'failed</h1><p class="err">{_e(job.get("error"))}</p>'
                '<p class="sub"><a href="/">Try another clip &rarr;</a></p>'
                "</div>"))
            return
        report = job.get("report") or ""
        if job["status"] != "done" or not os.path.exists(report):
            # Fell through to a non-JS client, or a reload mid-run.
            self._send(200, _page("analyzing",
                                  _WAIT_BODY.replace("@JOBID@", jobid)),
                       extra={"Cache-Control": "no-store"})
            return
        try:
            with open(report, "rb") as f:
                self._send(200, f.read())
        except OSError as e:
            self._send(500, _page("error",
                                  f'<div class="card"><p class="err">'
                                  f"could not read report: {_e(e)}</p></div>"))

    # -- POST -----------------------------------------------------------------
    def do_POST(self) -> None:
        _sweep()
        path = urlparse(self.path).path
        if path != "/verify":
            self._json(404, {"error": "unknown path"})
            return
        ctype = self.headers.get("Content-Type", "")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._json(400, {"error": "empty request body"})
            return
        if length > _MAX_UPLOAD:
            self._json(413, {"error": "clip too large (1 GiB max)"})
            return
        body = self.rfile.read(length)

        if "multipart/form-data" in ctype:
            fields, files = _parse_multipart(body, ctype)
        else:  # tolerate urlencoded probes; they'll fail on the missing file
            fields = {k: v[0] for k, v in
                      parse_qs(body.decode("utf-8", "replace")).items()}
            files = {}

        vid = files.get("video") or next(iter(files.values()), None)
        if not vid or not vid[1]:
            self._json(400,
                       {"error": "no video file uploaded (field 'video')"})
            return
        filename, data = vid
        claim_time = (fields.get("claim_time") or "").strip() or None
        claim_place = (fields.get("claim_place") or "").strip() or None
        claim_text = (fields.get("claim") or "").strip()
        if claim_text and (not claim_time or not claim_place):
            try:  # plain-words claim — explicit fields win when both set
                from .claim_parser import claim_to_args
                parsed_time, parsed_place = claim_to_args(claim_text)
                claim_time = claim_time or parsed_time or None
                claim_place = claim_place or parsed_place or None
            except Exception:
                pass  # a claim that won't parse just adds no hint
        jobid = _start_job(
            data,
            filename or "clip",
            claim_time,
            claim_place,
        )
        payload = {"jobid": jobid,
                   "status_url": f"/status/{jobid}",
                   "result_url": f"/result/{jobid}"}
        accept = self.headers.get("Accept", "")
        # A fetch() from the upload page wants JSON and polls /status itself.
        # A plain form navigation (Accept: text/html, no fetch) gets a 303 to
        # /result/<jobid>, which renders a self-refreshing "analyzing" page.
        if "application/json" not in accept and "text/html" in accept:
            self._send(303, b"", "text/plain",
                       {"Location": f"/result/{jobid}"})
        else:
            self._json(202, payload)


# ---------------------------------------------------------------------------
# entry points

def serve(port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Start the demo server (blocks until Ctrl+C). Returns the server."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True  # don't hang shutdown on a running job
    shown = host if host else "0.0.0.0"
    print(f"[sundial] demo server → http://{shown}:{httpd.server_port}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[sundial] stopping", flush=True)
    finally:
        httpd.server_close()
    return httpd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sundial.serve", description=__doc__)
    ap.add_argument("--port", type=int, default=8765,
                    help="listen port (default 8765)")
    ap.add_argument("--host", default="127.0.0.1",
                    help='bind address (default 127.0.0.1; use "" or '
                         "0.0.0.0 to expose on the LAN)")
    args = ap.parse_args(argv)
    serve(port=args.port, host=args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
