#!/usr/bin/env python3
"""
Roadrover Studio Server — local HTTP API + web UI for session management.

Start in a ROS 2 sourced terminal so processing jobs inherit the ROS environment:
    python3 studio_server.py [--bags-dir ~/roadrover_bags] [--port 8765]

Then open http://localhost:8765 in a browser for the session manager UI.
In Lichtblick: File > Import layout > roadrover_layout.json for the visualization panels.

To open a processed bag:
  • Click "→ MCAP" in the Session Manager, then click the URL to copy it
  • In Lichtblick: File > Open > Remote file > paste URL

Install deps (once):
    pip install fastapi uvicorn
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_PACIFIC = ZoneInfo("America/Los_Angeles")
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

TOOLS_DIR    = Path(__file__).parent
PIPELINE_DIR = TOOLS_DIR.parent / "src" / "roadrover_perception" / "scripts"
DEFAULT_BAGS_DIR = Path.home() / "roadrover_bags"

app = FastAPI(title="Roadrover Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set by main() before uvicorn.run()
bags_dir: Path = DEFAULT_BAGS_DIR


# ── Job management ────────────────────────────────────────────────────────────

class Job:
    def __init__(self, job_id: str, label: str, cmd: List[str]) -> None:
        self.job_id:   str = job_id
        self.label:    str = label
        self.cmd:      List[str] = cmd
        self.status:   str = "running"
        self.log:      List[str] = []
        self.started:  float = time.time()
        self.finished: Optional[float] = None


jobs: Dict[str, Job] = {}


async def _run_job(job: Job) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *job.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            job.log.append(raw.decode(errors="replace").rstrip())
        await proc.wait()
        job.status = "done" if proc.returncode == 0 else "error"
    except Exception as exc:
        job.log.append(f"[server] {exc}")
        job.status = "error"
    finally:
        job.finished = time.time()


def start_job(label: str, cmd: List[str]) -> str:
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, label, cmd)
    jobs[job_id] = job
    asyncio.create_task(_run_job(job))
    return job_id


# ── Session discovery ─────────────────────────────────────────────────────────

_OUTPUT_SUFFIXES = ("_map", "_chunks", "_processed", "_scenario", "_mcap")


def _is_raw_session(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any(path.name.endswith(s) for s in _OUTPUT_SUFFIXES):
        return False
    return (path / "metadata.yaml").exists() or bool(list(path.glob("*.db3")))


def _find_mcap_file(bag_dir: Path) -> Optional[Path]:
    """First .mcap data file inside a rosbag2 directory, or None."""
    if not bag_dir.exists():
        return None
    return next(bag_dir.glob("*.mcap"), None)


def _mcap_url(mcap_path: Optional[Path]) -> Optional[str]:
    if mcap_path is None:
        return None
    try:
        return "/files/" + mcap_path.relative_to(bags_dir).as_posix()
    except ValueError:
        return None


def _parse_created_at(name: str) -> Optional[str]:
    m = re.match(r"session_(\d{8})_(\d{6})", name)
    if not m:
        return None
    try:
        dt_utc = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(_PACIFIC).strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return None


def _processed_at(chunks_dir: Path) -> Optional[str]:
    proc_dirs = [d for d in chunks_dir.iterdir() if d.name.endswith("_processed")] if chunks_dir.exists() else []
    if not proc_dirs:
        return None
    latest = max(proc_dirs, key=lambda d: d.stat().st_mtime)
    return datetime.fromtimestamp(latest.stat().st_mtime, tz=_PACIFIC).strftime("%Y-%m-%d %H:%M %Z")


def _session_info(session_dir: Path) -> dict:
    name = session_dir.name
    chunks_dir = bags_dir / (name + "_chunks")
    map_dir    = bags_dir / (name + "_map")

    chunks: List[dict] = []
    if chunks_dir.exists():
        for cdir in sorted(
            d for d in chunks_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("chunk_")
            and not any(d.name.endswith(s) for s in ("_processed", "_scenario", "_mcap"))
        ):
            proc_dir      = chunks_dir / (cdir.name + "_processed")
            scen_dir      = chunks_dir / (cdir.name + "_scenario")
            raw_mcap_dir  = chunks_dir / (cdir.name + "_mcap")
            proc_mcap_dir = chunks_dir / (cdir.name + "_processed_mcap")

            raw_mcap  = _find_mcap_file(raw_mcap_dir)
            proc_mcap = _find_mcap_file(proc_mcap_dir)
            xosc      = next(scen_dir.glob("*.xosc"), None) if scen_dir.exists() else None

            chunks.append({
                "name":               cdir.name,
                "processed":          proc_dir.exists(),
                "has_scenario":       xosc is not None,
                "raw_mcap":           raw_mcap is not None,
                "processed_mcap":     proc_mcap is not None,
                "raw_mcap_url":       _mcap_url(raw_mcap),
                "processed_mcap_url": _mcap_url(proc_mcap),
                "scenario_xosc":      str(xosc) if xosc else None,
            })

    return {
        "id":           name,
        "name":         name,
        "created_at":   _parse_created_at(name),
        "processed_at": _processed_at(chunks_dir),
        "map_ready":    (map_dir / "map_graph.pkl").exists(),
        "chunked":      chunks_dir.exists(),
        "processed":    any(c["processed"] for c in chunks),
        "has_scenario": any(c["has_scenario"] for c in chunks),
        "chunks":       chunks,
    }


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
def api_sessions() -> List[dict]:
    if not bags_dir.exists():
        return []
    return [
        _session_info(d)
        for d in sorted(bags_dir.iterdir(), reverse=True)
        if _is_raw_session(d)
    ]


@app.get("/api/sessions/{session_id}")
def api_session(session_id: str) -> dict:
    d = bags_dir / session_id
    if not d.exists():
        raise HTTPException(404, "Session not found")
    return _session_info(d)


@app.post("/api/sessions/{session_id}/split")
async def api_split(session_id: str) -> dict:
    """Download map + split into chunks without running perception."""
    d = bags_dir / session_id
    if not d.exists():
        raise HTTPException(404, "Session not found")
    cmd = [sys.executable, str(PIPELINE_DIR / "pipeline_session.py"), str(d), "--yes", "--split-only"]
    return {"job_id": start_job(f"split:{session_id}", cmd)}


@app.post("/api/sessions/{session_id}/process")
async def api_process(session_id: str) -> dict:
    """Download map + split + run perception on all moving chunks."""
    d = bags_dir / session_id
    if not d.exists():
        raise HTTPException(404, "Session not found")
    cmd = [sys.executable, str(PIPELINE_DIR / "pipeline_session.py"), str(d), "--yes"]
    yolo = PIPELINE_DIR / "yolov8s.pt"
    if yolo.exists():
        cmd += ["--yolo-weights", str(yolo)]
    return {"job_id": start_job(f"process:{session_id}", cmd)}


@app.post("/api/sessions/{session_id}/chunks/{chunk_name}/process")
async def api_chunk_process(session_id: str, chunk_name: str) -> dict:
    """Run perception pipeline on a single chunk."""
    chunks_dir = bags_dir / (session_id + "_chunks")
    chunk_dir  = chunks_dir / chunk_name
    map_graph  = bags_dir / (session_id + "_map") / "map_graph.pkl"
    if not chunk_dir.exists():
        raise HTTPException(404, "Chunk not found — run Split first")
    if not map_graph.exists():
        raise HTTPException(400, "Map not ready — run Split first to download map")
    processed = chunks_dir / (chunk_name + "_processed")
    cmd = [
        sys.executable, str(PIPELINE_DIR / "process_bag.py"),
        str(chunk_dir),
        "--output", str(processed),
        "--map-graph", str(map_graph),
    ]
    return {"job_id": start_job(f"process:{session_id}/{chunk_name}", cmd)}


@app.post("/api/sessions/{session_id}/extract")
async def api_extract(session_id: str) -> dict:
    chunks_dir = bags_dir / (session_id + "_chunks")
    map_graph  = bags_dir / (session_id + "_map") / "map_graph.pkl"
    if not chunks_dir.exists():
        raise HTTPException(400, "Session not processed yet — run Process first")
    if not map_graph.exists():
        raise HTTPException(400, "Map graph not found — re-run Process to download it")
    cmd = [
        sys.executable, str(TOOLS_DIR / "run_extract_all.py"),
        str(chunks_dir), "--map-graph", str(map_graph),
    ]
    return {"job_id": start_job(f"extract:{session_id}", cmd)}


@app.post("/api/sessions/{session_id}/convert-mcap")
async def api_convert(
    session_id: str,
    type: str = Query(default="processed", description="raw | processed | both"),
) -> dict:
    chunks_dir = bags_dir / (session_id + "_chunks")
    if not chunks_dir.exists():
        raise HTTPException(400, "Session not split yet — run Split first")
    cmd = [sys.executable, str(TOOLS_DIR / "convert_bags.py"), str(chunks_dir)]
    if type == "raw":
        cmd.append("--raw")
    elif type == "both":
        cmd.append("--both")
    label = f"→ MCAP ({type}): {session_id}"
    return {"job_id": start_job(label, cmd)}


@app.post("/api/sessions/{session_id}/chunks/{chunk_name}/convert-mcap")
async def api_chunk_convert(session_id: str, chunk_name: str) -> dict:
    """Convert a single chunk (raw or processed) to MCAP."""
    chunks_dir = bags_dir / (session_id + "_chunks")
    chunk_dir  = chunks_dir / chunk_name
    if not chunk_dir.exists():
        raise HTTPException(404, "Chunk not found")
    cmd = [
        sys.executable, str(TOOLS_DIR / "convert_bags.py"),
        str(chunks_dir), "--chunk", chunk_name, "--force",
    ]
    label = f"→ MCAP: {session_id}/{chunk_name}"
    return {"job_id": start_job(label, cmd)}


@app.post("/api/sessions/{session_id}/esmini")
async def api_esmini(
    session_id: str,
    chunk: Optional[str] = Query(default=None),
) -> dict:
    chunks_dir = bags_dir / (session_id + "_chunks")
    if not chunks_dir.exists():
        raise HTTPException(400, "Session not processed")

    if chunk:
        xosc_files = list((chunks_dir / (chunk + "_scenario")).glob("*.xosc"))
    else:
        xosc_files = sorted(chunks_dir.glob("*_scenario/*.xosc"))

    if not xosc_files:
        raise HTTPException(400, "No .xosc files found — run Extract Scenario first")

    xosc = xosc_files[0]
    cmd  = ["esmini", "--osc", str(xosc), "--window", "60", "60", "800", "400"]
    return {"job_id": start_job(f"esmini:{session_id}", cmd), "xosc": str(xosc)}


@app.get("/api/jobs")
def api_jobs() -> List[dict]:
    return [
        {
            "job_id":  j.job_id,
            "label":   j.label,
            "status":  j.status,
            "elapsed": round((j.finished or time.time()) - j.started, 1),
        }
        for j in sorted(jobs.values(), key=lambda j: j.started, reverse=True)
    ]


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id":   job.job_id,
        "label":    job.label,
        "status":   job.status,
        "log_tail": job.log[-100:],
        "elapsed":  round((job.finished or time.time()) - job.started, 1),
    }


@app.get("/", response_class=HTMLResponse)
def web_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Roadrover</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; font-size: 12px; background: #1a1a1a; color: #e0e0e0;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  #toolbar { display: flex; align-items: center; gap: 6px; padding: 6px 10px;
             background: #222; border-bottom: 1px solid #333; flex-shrink: 0; }
  #toolbar span { color: #7ec8e3; font-weight: bold; }
  #toolbar button { padding: 2px 8px; border: 1px solid #555; background: #2a2a2a;
                    color: #ccc; border-radius: 3px; cursor: pointer; font: 11px monospace; }
  #toolbar button:hover { border-color: #7ec8e3; }
  #body { flex: 1; overflow-y: auto; padding: 8px; }
  #error { color: #eb5757; padding: 6px 0; font-size: 11px; }
  .session { background: #252525; border: 1px solid #333; border-radius: 4px; margin-bottom: 6px; }
  .session-header { display: flex; align-items: center; flex-wrap: wrap; gap: 5px;
                    padding: 6px 8px; cursor: pointer; user-select: none; }
  .session-header:hover { background: #2a2a2a; }
  .sname { font-weight: bold; }
  .smeta { font-size: 10px; color: #666; margin-top: 1px; }
  .sname-col { flex: 1; display: flex; flex-direction: column; }
  .chunks { padding: 4px 8px 8px 20px; border-top: 1px solid #2a2a2a; }
  .chunk { display: flex; align-items: center; flex-wrap: wrap; gap: 4px;
           padding: 4px 0; border-bottom: 1px solid #282828; }
  .cname { width: 130px; color: #999; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badge { padding: 1px 5px; border-radius: 3px; font-size: 10px; }
  .ok  { background: #1a3d24; color: #6fcf97; }
  .bad { background: #3a1515; color: #eb5757; }
  .btn { padding: 2px 7px; border: 1px solid #555; background: #2a2a2a; color: #ccc;
         border-radius: 3px; cursor: pointer; font: 11px monospace; }
  .btn:hover { border-color: #aaa; }
  .btn.primary { border-color: #7ec8e3; color: #7ec8e3; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .section-title { font-size: 11px; color: #666; margin: 10px 0 4px; }
  .job-card { background: #222; border: 1px solid #333; border-radius: 4px; margin-bottom: 6px; }
  .job-header { display: flex; align-items: center; gap: 6px; padding: 5px 8px; }
  .job-label { flex: 1; color: #ccc; font-weight: bold; }
  .job-elapsed { color: #555; font-size: 10px; }
  .job-step { font-size: 10px; color: #7ec8e3; padding: 0 8px 4px; }
  .s-running { color: #f2c94c; } .s-done { color: #6fcf97; } .s-error { color: #eb5757; }
  .prog-track { height: 3px; background: #333; margin: 0 8px 6px; border-radius: 2px; overflow: hidden; }
  .prog-fill { height: 100%; background: #7ec8e3; border-radius: 2px; transition: width 0.4s; }
  @keyframes pulse { 0%{transform:translateX(-100%)} 100%{transform:translateX(400%)} }
  .prog-pulse { height: 100%; width: 25%; background: #7ec8e3; border-radius: 2px;
                animation: pulse 1.2s ease-in-out infinite; }
  .log-box { font-size: 10px; color: #888; background: #111; padding: 5px 8px;
             max-height: 180px; overflow-y: auto; white-space: pre-wrap; word-break: break-all;
             border-radius: 0 0 3px 3px; border-top: 1px solid #2a2a2a; }
  .log-toggle { font-size: 10px; color: #555; cursor: pointer; padding: 0 8px 5px;
                display: inline-block; }
  .log-toggle:hover { color: #999; }
  #toast { position: fixed; bottom: 12px; right: 12px; background: #333; color: #7ec8e3;
           padding: 6px 12px; border-radius: 4px; font-size: 11px; z-index: 999;
           max-width: 320px; display: none; }
  .url-copy { display: flex; align-items: center; gap: 3px; }
  .url-copy input { background: #111; border: 1px solid #444; color: #aaa;
                    padding: 1px 4px; border-radius: 3px; font: 10px monospace;
                    width: 160px; cursor: text; }
</style>
</head>
<body>
<div id="toolbar">
  <span>&#x1F5FA; Roadrover</span>
  <button onclick="loadSessions()">&#x21BA; Refresh</button>
</div>
<div id="body">
  <div id="error" style="display:none"></div>
  <div id="jobs-section" style="display:none">
    <div class="section-title">Active Jobs</div>
    <div id="jobs"></div>
  </div>
  <div id="sessions"></div>
</div>
<div id="toast"></div>

<script>
const BASE = window.location.origin;
const expanded = {};
const streams = {};   // job_id -> { es, lines[], label, status, showLog }
let allJobs = [];

function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.style.display = 'none', 2500);
}

async function apiFetch(method, path, qs) {
  let url = BASE + path;
  if (qs) url += '?' + new URLSearchParams(qs);
  const r = await fetch(url, { method });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function postAction(path, label, qs) {
  try {
    const r = await apiFetch('POST', path, qs);
    toast(label + ' started');
    openStream(r.job_id, label);
  } catch(e) { toast('Error: ' + e); }
}

// ── Progress parsing ───────────────────────────────────────────────────────
function parseProgress(lines) {
  let step = '', pct = null, frames = null;
  for (const line of lines) {
    if (/Checking session velocity/.test(line))  { step = 'Checking velocity…'; pct = 5; }
    else if (/Step 1:/.test(line))               { step = 'Step 1: Downloading OSM map…'; pct = 10; }
    else if (/Step 2:/.test(line))               { step = 'Step 2: Splitting chunks…'; pct = 25; }
    else if (/Loading YOLOv8/.test(line))        { step = 'Loading YOLO model…'; pct = 5; }
    else if (/Model ready/.test(line))           { step = 'Running perception…'; pct = 20; }
    else if (/Map matching enabled/.test(line))  { step = 'Running perception…'; pct = 25; }
    else if (/All done/.test(line))              { step = 'Done'; pct = 100; }
    else if (/^Done —/.test(line.trim()))   { step = 'Done'; pct = 100; }
    else if (/Split-only mode/.test(line))       { step = 'Split complete'; pct = 100; }
    const cm = line.match(/\\((\\d+)\\/(\\d+)\\)/);
    if (cm) {
      const k = parseInt(cm[1]), n = parseInt(cm[2]);
      step = 'Processing chunk ' + k + ' of ' + n;
      pct = Math.round(30 + (k / n) * 65);
    }
    const fm = line.match(/(\\d+) frames processed/);
    if (fm) frames = parseInt(fm[1]);
  }
  return { step, pct, frames };
}

// ── SSE stream management ──────────────────────────────────────────────────
function openStream(jobId, label) {
  if (streams[jobId]) return;
  streams[jobId] = { es: null, lines: [], label, status: 'running', showLog: true, _start: Date.now() };
  renderJobsSection();
  renderJobCard(jobId);
  document.getElementById('jobs-section').scrollIntoView({ behavior: 'smooth', block: 'start' });

  const es = new EventSource(BASE + '/api/jobs/' + jobId + '/stream');
  streams[jobId].es = es;

  es.onmessage = (e) => {
    streams[jobId].lines.push(JSON.parse(e.data));
    renderJobCard(jobId);
  };
  es.addEventListener('done', (e) => {
    streams[jobId].status = JSON.parse(e.data);
    es.close();
    renderJobCard(jobId);
    loadSessions();
  });
  es.onerror = () => { es.close(); };
}

function toggleJobLog(jobId) {
  if (!streams[jobId]) return;
  streams[jobId].showLog = !streams[jobId].showLog;
  renderJobCard(jobId);
}

function renderJobCard(jobId) {
  const s = streams[jobId];
  if (!s) return;
  const el = document.getElementById('job-' + jobId);
  if (!el) return;

  const { step, pct, frames } = parseProgress(s.lines);
  const running = s.status === 'running';
  const elapsed = s._start ? Math.round((Date.now() - s._start) / 1000) + 's' : '';

  let progHtml = '';
  if (running) {
    progHtml = (pct !== null)
      ? '<div class="prog-track"><div class="prog-fill" style="width:' + pct + '%"></div></div>'
      : '<div class="prog-track"><div class="prog-pulse"></div></div>';
  } else if (s.status === 'done') {
    progHtml = '<div class="prog-track"><div class="prog-fill" style="width:100%"></div></div>';
  }

  const stepText = step + (frames && running ? ' · ' + frames + ' frames' : '');
  const stepHtml = stepText ? '<div class="job-step">' + stepText + '</div>' : '';
  const toggleLbl = s.showLog ? '▲ hide log' : '▼ show log';
  const logHtml = s.showLog
    ? '<div class="log-box" id="logbox-' + jobId + '">' + (s.lines.slice(-300).join('\\n') || '(waiting for output…)') + '</div>'
    : '';

  el.innerHTML =
    '<div class="job-header">' +
      '<span class="job-label">' + s.label + '</span>' +
      '<span class="s-' + s.status + '">' + s.status + '</span>' +
      (elapsed ? '<span class="job-elapsed">' + elapsed + '</span>' : '') +
    '</div>' +
    stepHtml + progHtml +
    `<span class="log-toggle" onclick="toggleJobLog('${jobId}')">${toggleLbl}</span>` +
    logHtml;

  if (s.showLog) {
    const lb = document.getElementById('logbox-' + jobId);
    if (lb) lb.scrollTop = lb.scrollHeight;
  }
}

function renderJobsSection() {
  const ids = Object.keys(streams);
  const sec = document.getElementById('jobs-section');
  if (!ids.length) { sec.style.display = 'none'; return; }
  sec.style.display = 'block';
  const jobsEl = document.getElementById('jobs');
  // add cards for any new job ids
  ids.forEach(jobId => {
    if (!document.getElementById('job-' + jobId)) {
      const div = document.createElement('div');
      div.className = 'job-card';
      div.id = 'job-' + jobId;
      jobsEl.prepend(div);
    }
  });
}

function badge(ok, trueLabel, falseLabel) {
  return `<span class="badge ${ok?'ok':'bad'}">${ok?trueLabel:falseLabel}</span>`;
}
function okBadge(ok, label) {
  return ok ? `<span class="badge ok">${label}</span>` : '';
}

function copyUrl(url) {
  navigator.clipboard.writeText(url).then(
    () => toast('Copied! In Lichtblick: File ▶ Open ▶ Remote file ▶ paste URL'),
    () => {
      prompt('Copy this URL:', url);
    }
  );
}

function renderChunk(sid, c) {
  const processBtn = c.processed
    ? `<button class="btn" title="Overwrites existing processed bag" onclick="postAction('/api/sessions/${sid}/chunks/${c.name}/process','Reprocess ${c.name}')">Reprocess</button>`
    : `<button class="btn primary" onclick="postAction('/api/sessions/${sid}/chunks/${c.name}/process','Process ${c.name}')">Process</button>`;

  // Raw MCAP: convert button always available (overwrites); copy-URL button once it exists
  const rawConvert = `<button class="btn" title="${c.raw_mcap ? 'Reconvert (overwrites existing MCAP)' : 'Convert raw db3 to MCAP'}" onclick="postAction('/api/sessions/${sid}/chunks/${c.name}/convert-mcap','→ MCAP ${c.name}')">→ MCAP (raw)</button>`;
  const rawCopy = c.raw_mcap_url
    ? `<button class="btn" onclick="copyUrl('${BASE}${c.raw_mcap_url}')">Copy raw MCAP</button>` : '';

  // Processed MCAP: only available once the chunk is processed
  const procConvert = c.processed
    ? `<button class="btn" title="${c.processed_mcap ? 'Reconvert (overwrites existing MCAP)' : 'Convert processed db3 to MCAP'}" onclick="postAction('/api/sessions/${sid}/chunks/${c.name + '_processed'}/convert-mcap','→ MCAP ${c.name}_processed')">→ MCAP (proc)</button>`
    : '';
  const procCopy = c.processed_mcap_url
    ? `<button class="btn" onclick="copyUrl('${BASE}${c.processed_mcap_url}')">Copy proc MCAP</button>` : '';

  const esmini = c.has_scenario
    ? `<button class="btn primary" onclick="postAction('/api/sessions/${sid}/esmini','ESMini',{chunk:'${c.name}'})">&#x25B6; ESMini</button>` : '';
  return `<div class="chunk">
    <span class="cname" title="${c.name}">${c.name}</span>
    ${badge(c.processed,'proc','raw')}
    ${okBadge(c.has_scenario,'scenario')}
    ${okBadge(c.raw_mcap,'raw mcap')}
    ${okBadge(c.processed_mcap,'proc mcap')}
    ${processBtn}${rawConvert}${rawCopy}${procConvert}${procCopy}${esmini}
  </div>`;
}

function renderSession(s) {
  const isExp = !!expanded[s.id];
  const chunks = isExp && s.chunks.length > 0
    ? `<div class="chunks">${s.chunks.map(c=>renderChunk(s.id,c)).join('')}</div>`
    : isExp ? `<div class="chunks" style="color:#555">No chunks found.</div>` : '';

  const createdLine = s.created_at ? `recorded ${s.created_at}` : '';
  const processedLine = s.processed_at ? `processed ${s.processed_at}` : '';
  const metaLine = [createdLine, processedLine].filter(Boolean).join(' &nbsp;·&nbsp; ');

  return `<div class="session" id="sess-${s.id}">
    <div class="session-header" onclick="toggleExpand('${s.id}')">
      <div class="sname-col">
        <span class="sname">${s.name}</span>
        ${metaLine ? `<span class="smeta">${metaLine}</span>` : ''}
      </div>
      ${badge(s.map_ready,'map ✓','no map')}
      ${badge(s.processed,'processed','raw')}
      ${okBadge(s.has_scenario,'scenario')}
      <span onclick="event.stopPropagation()" style="display:flex;gap:4px">
        <button class="btn" onclick="postAction('/api/sessions/${s.id}/split','Split')">Split</button>
        <button class="btn primary" onclick="postAction('/api/sessions/${s.id}/process','Process all')">Process all</button>
        <button class="btn" onclick="postAction('/api/sessions/${s.id}/extract','Extract')">Extract scenario</button>
        <button class="btn" onclick="postAction('/api/sessions/${s.id}/convert-mcap','→ MCAP raw',{type:'raw'})">&#x2192; MCAP raw</button>
        <button class="btn" onclick="postAction('/api/sessions/${s.id}/convert-mcap','→ MCAP proc',{type:'processed'})">&#x2192; MCAP proc</button>
      </span>
    </div>
    ${chunks}
  </div>`;
}

function toggleExpand(id) {
  expanded[id] = !expanded[id];
  renderAll(window._sessions || []);
}

function renderAll(sessions) {
  window._sessions = sessions;
  const el = document.getElementById('sessions');
  if (!sessions.length) {
    el.innerHTML = '<div style="color:#555;padding:8px 0">No sessions found in ~/roadrover_bags</div>';
    return;
  }
  el.innerHTML = sessions.map(renderSession).join('');
}

async function loadSessions() {
  document.getElementById('error').style.display = 'none';
  try {
    const data = await apiFetch('GET', '/api/sessions');
    renderAll(data);
  } catch(e) {
    const err = document.getElementById('error');
    err.style.display = 'block';
    err.innerHTML = 'Could not reach server: ' + e + '<br><span style="color:#888">Start: <code>python3 tools/studio_server.py</code></span>';
  }
}

// On page load, pick up any already-running jobs from the server
async function loadExistingJobs() {
  try {
    const jobs = await apiFetch('GET', '/api/jobs');
    for (const j of jobs) {
      if (j.status === 'running' && !streams[j.job_id]) {
        openStream(j.job_id, j.label);
      }
    }
  } catch(_) {}
}

loadSessions();
loadExistingJobs();
</script>
</body>
</html>"""


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_log(job_id: str):
    """SSE endpoint — streams log lines in real time while a job is running."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def generate():
        sent = 0
        while True:
            new_lines = job.log[sent:]
            for line in new_lines:
                yield f"data: {json.dumps(line)}\n\n"
            sent = len(job.log)
            if job.status != "running":
                yield f"event: done\ndata: {json.dumps(job.status)}\n\n"
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/files/{path:path}")
def serve_file(path: str) -> FileResponse:
    """Serve MCAP files over HTTP so Lichtblick can open them via 'Remote file' URL."""
    full = (bags_dir / path).resolve()
    try:
        full.relative_to(bags_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Forbidden")
    if not full.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(full))


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    global bags_dir

    ap = argparse.ArgumentParser(description="Roadrover Studio Server")
    ap.add_argument("--bags-dir", default=str(DEFAULT_BAGS_DIR),
                    help=f"Root directory containing session folders (default: {DEFAULT_BAGS_DIR})")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    bags_dir = Path(args.bags_dir).expanduser().resolve()
    print(f"Bags dir : {bags_dir}")
    print(f"Web UI   : http://localhost:{args.port}/")
    print(f"API      : http://localhost:{args.port}/api/sessions")
    print()
    print("NOTE: start in a ROS 2 sourced terminal so processing jobs work.")
    print("In Lichtblick: File > Import layout > roadrover_layout.json  (one-time setup)")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
