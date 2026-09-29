/* ==========================================================================
   Model Scanner — front-end controller
   Talks to: POST /api/scans/upload, POST /api/scans/hf,
             GET  /api/scans/:id, /api/scans/:id/report[/html], GET /api/rules
   ========================================================================== */
"use strict";

/* ---------------- constants ---------------- */

const SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"];
const SEV_RANK = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, INFO: 4 };
const SEV_LABEL = { CRITICAL: "Critical", HIGH: "High", MEDIUM: "Medium", LOW: "Low", INFO: "Info" };

const STATUS_COPY = {
  QUEUED:    ["Scan queued", "Waiting for an available scanner worker…"],
  ACQUIRING: ["Acquiring artifacts", "Fetching model files from the remote repository…"],
  SCANNING:  ["Analysing", "Running static analysis — inspecting structure, headers and opcode streams…"],
  COMPLETED: ["Scan complete", "Building the report…"],
  ERROR:     ["Scan failed", "The scan terminated before completing."],
  CANCELLED: ["Scan cancelled", "This scan was cancelled and its report discarded."],
};

/* Polling: fast at first, then backs off. Hard ceiling so a wedged
   job can never spin the tab forever (this was the "status stuck" bug). */
const POLL_START_MS = 700;
const POLL_MAX_MS = 4000;
const POLL_BACKOFF = 1.25;
const POLL_TIMEOUT_MS = 15 * 60 * 1000;
const POLL_MAX_CONSECUTIVE_ERRORS = 4;

/* ---------------- state ---------------- */

const state = {
  scanId: null,
  result: null,
  artifactIndex: 0,
  selectedFile: null,
  selectedFolder: null,   // FileList from a webkitdirectory pick
  rules: [],
  rulesLoaded: false,
  ruleSev: "ALL",
  findingSev: "ALL",
  findingQuery: "",
  poll: null,          // { generation, timer, controller, startedAt, delay, errors }
  generation: 0,       // invalidates in-flight work from a superseded scan
  lastSubmit: null,    // for the retry button
  submitting: false,   // guards against double-submits piling up the queue
};

/* ---------------- tiny helpers ---------------- */

const $ = (id) => document.getElementById(id);
const show = (el) => el && el.classList.remove("is-hidden");
const hide = (el) => el && el.classList.add("is-hidden");
const setText = (id, v) => { const el = $(id); if (el) el.textContent = String(v); };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function formatBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), units.length - 1);
  const val = n / Math.pow(1024, i);
  return `${val >= 100 || i === 0 ? Math.round(val) : val.toFixed(val >= 10 ? 1 : 2)} ${units[i]}`;
}

/* Human-readable scan time. The JSON report keeps ISO 8601. */
function formatTimestamp(iso) {
  if (!iso) return "—";
  const d = new Date(String(iso).replace(" ", "T"));
  if (isNaN(d.getTime())) return String(iso);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getDate())}/${p(d.getMonth() + 1)}/${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function formatDuration(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n)) return "—";
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60000) return `${(n / 1000).toFixed(1)} s`;
  return `${Math.floor(n / 60000)}m ${Math.round((n % 60000) / 1000)}s`;
}

const pretty = (v) => String(v ?? "").replace(/_/g, " ");

function formatTimestampLocal(iso) {
  if (!iso) return "—";
  const d = new Date(String(iso).replace(" ", "T"));
  if (isNaN(d.getTime())) return String(iso);
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "local";
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getDate())}/${p(d.getMonth() + 1)}/${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())} (${tz})`;
}

function formatOutcome(outcome, severityCounts) {
  if (!outcome || outcome === "COMPLETE") return pretty(outcome || "—");
  if (String(outcome).toUpperCase() === "WARNING") {
    const counts = severityCounts || {};
    if ((counts.CRITICAL || 0) > 0) return "Findings Detected (Critical)";
    if ((counts.HIGH || 0) > 0) return "Findings Detected (High)";
    if ((counts.MEDIUM || 0) > 0) return "Findings Detected (Medium)";
    return "Findings Detected";
  }
  return pretty(outcome);
}

function safeSev(raw) {
  const s = String(raw || "").toUpperCase();
  return SEVERITIES.includes(s) ? s : "INFO";
}

function scrollToEl(node) {
  if (!node) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  node.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
}

/* ---------------- boot ---------------- */

function boot() {
  initTheme();
  initTabs();
  initDropzone();
  initExamples();
  initScanActions();
  initFindingFilters();
  initRulesModal();
  preloadRuleCount();
  refreshCapacity(false);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}

/* ---------------- theme ---------------- */

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("ms-theme"); } catch (_) { /* private mode */ }
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  const btn = $("btn-theme");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const current = document.documentElement.getAttribute("data-theme") || (systemDark ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("ms-theme", next); } catch (_) { /* ignore */ }
  });
}

/* ---------------- source tabs ---------------- */

function initTabs() {
  const seg = document.querySelector(".seg");
  const buttons = Array.from(document.querySelectorAll(".seg-btn"));
  if (!seg || !buttons.length) return;
  seg.dataset.active = "upload";

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      seg.dataset.active = tab;
      buttons.forEach((b) => {
        const on = b === btn;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      document.querySelectorAll(".pane").forEach((pane) => {
        const on = pane.id === `tab-${tab}`;
        pane.classList.toggle("is-active", on);
        pane.hidden = !on;
      });
      hide($("upload-error"));
      hide($("hf-error"));
    });
  });
}

function activeTab() {
  const seg = document.querySelector(".seg");
  return (seg && seg.dataset.active) || "upload";
}

/* ---------------- dropzone ----------------
   The old markup stacked a full-bleed <input type=file> on top of the
   dropzone content, so the "browse" control was unreachable and the label
   text swallowed clicks. The input is now visually hidden and the whole
   dropzone is its <label>, which is what makes click + keyboard work.      */

function initDropzone() {
  const zone = $("dropzone");
  const input = $("file-input");
  if (!zone || !input) return;

  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (file) selectFile(file);
  });

  let depth = 0; // dragenter/dragleave fire for children too — count them
  const stop = (e) => { e.preventDefault(); e.stopPropagation(); };

  ["dragenter", "dragover", "dragleave", "drop"].forEach((evt) => {
    zone.addEventListener(evt, stop);
  });
  zone.addEventListener("dragenter", () => { depth += 1; zone.classList.add("is-drag"); });
  zone.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) zone.classList.remove("is-drag"); });
  zone.addEventListener("dragover", (e) => { if (e.dataTransfer) e.dataTransfer.dropEffect = "copy"; });

  zone.addEventListener("drop", (e) => {
    depth = 0;
    zone.classList.remove("is-drag");
    const dt = e.dataTransfer;
    if (!dt) return;
    let file = dt.files && dt.files[0];
    if (!file && dt.items && dt.items.length) {
      const item = Array.from(dt.items).find((i) => i.kind === "file");
      if (item) file = item.getAsFile();
    }
    if (file) {
      selectFile(file);
      // Keep the native input in sync where the browser allows it.
      try { input.files = dt.files; } catch (_) { /* read-only in some browsers */ }
    } else {
      showFieldError("upload-error", "That drop did not contain a file. Try dragging the model file itself.");
    }
  });

  // Dropping anywhere else on the page should not make the browser
  // navigate away from the app and lose the session.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // ---- folder upload ----
  // A single file can never trigger the repository analyzer, so the SC-series
  // supply-chain rules and the C001-C003 correlations are unreachable from a
  // one-file scan. Uploading the folder with webkitRelativePath preserved is
  // what unlocks them.
  const folderInput = $("folder-input");
  const folderBtn = $("folder-btn");
  if (folderBtn && folderInput) {
    folderBtn.addEventListener("click", () => folderInput.click());
    folderInput.addEventListener("change", () => {
      const files = Array.from(folderInput.files || []);
      if (files.length) selectFolder(files);
    });
  }

  const clear = $("file-clear");
  if (clear) {
    clear.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      clearSelection();
    });
  }
}

function clearSelection() {
  state.selectedFile = null;
  state.selectedFolder = null;
  const fi = $("file-input"); if (fi) fi.value = "";
  const fo = $("folder-input"); if (fo) fo.value = "";
  hide($("file-card"));
  show($("dropzone"));
  show($("folder-btn"));
  show(document.querySelector(".orsplit"));
}

const FOLDER_SKIP = /(^|\/)(\.git|\.github|__pycache__|node_modules|\.DS_Store|\.ipynb_checkpoints)(\/|$)/i;

function selectFolder(files) {
  // Drop VCS and cache noise: it inflates the upload and is not part of the model.
  const keep = files.filter((f) => !FOLDER_SKIP.test(f.webkitRelativePath || f.name));
  if (!keep.length) {
    showFieldError("upload-error", "That folder has no scannable files in it.");
    return;
  }
  const skipped = files.length - keep.length;
  const root = (keep[0].webkitRelativePath || "").split("/")[0] || "selected folder";
  const bytes = keep.reduce((sum, f) => sum + (f.size || 0), 0);

  state.selectedFolder = keep;
  state.selectedFile = null;

  setText("file-name", root);
  setText("file-size",
    `${keep.length} ${plural(keep.length, "file", "files")} · ${formatBytes(bytes)}` +
    (skipped ? ` · ${skipped} skipped` : ""));

  const cardEl = $("file-card");
  if (cardEl) {
    let badge = cardEl.querySelector(".filecard-count");
    if (!badge) {
      badge = el("span", "filecard-count");
      cardEl.insertBefore(badge, cardEl.querySelector("#file-clear"));
    }
    badge.textContent = "FOLDER";
  }
  show(cardEl);
  hide($("dropzone"));
  hide($("folder-btn"));
  hide(document.querySelector(".orsplit"));
  hide($("upload-error"));
}

function selectFile(file) {
  state.selectedFile = file;
  state.selectedFolder = null;
  const badge = document.querySelector(".filecard-count");
  if (badge) badge.remove();
  hide($("folder-btn"));
  hide(document.querySelector(".orsplit"));
  setText("file-name", file.name);
  setText("file-size", `${formatBytes(file.size)} · ${file.type || "binary"}`);
  show($("file-card"));
  hide($("dropzone"));
  hide($("upload-error"));
}

function initExamples() {
  document.querySelectorAll(".example").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = $("hf-url-input");
      if (input) { input.value = btn.dataset.url || ""; input.focus(); }
      hide($("hf-error"));
    });
  });
}

/* ---------------- scan submission ---------------- */

function initScanActions() {
  const start = $("start-scan-btn");
  if (start) start.addEventListener("click", submitScan);

  const hfInput = $("hf-url-input");
  if (hfInput) {
    hfInput.addEventListener("keydown", (e) => { if (e.key === "Enter") submitScan(); });
    hfInput.addEventListener("input", () => hide($("hf-error")));
  }

  const newScan = $("btn-new-scan");
  if (newScan) {
    newScan.addEventListener("click", () => {
      stopPolling();
      hide($("results-card"));
      hide($("status-card"));
      hide($("queue-recovery"));
      clearSelection();
      scrollToEl($("input-card"));
    });
  }

  const cancel = $("btn-cancel-scan");
  if (cancel) cancel.addEventListener("click", cancelCurrentScan);

  const cancelAll = $("btn-cancel-all");
  if (cancelAll) cancelAll.addEventListener("click", cancelAllWaiting);

  const refresh = $("btn-refresh-capacity");
  if (refresh) refresh.addEventListener("click", () => refreshCapacity(true));

  const retry = $("btn-retry");
  if (retry) retry.addEventListener("click", () => { if (state.lastSubmit) state.lastSubmit(); });

  const json = $("btn-download-json");
  if (json) json.addEventListener("click", () => downloadReport(""));

  const html = $("btn-download-html");
  if (html) html.addEventListener("click", () => downloadReport("/html"));
}

/* The old code set window.location.href for downloads. If the server
   answered with a JSON error instead of a file, the whole SPA navigated
   away and the results vanished. Fetch first, verify, then save. */
async function downloadReport(suffix) {
  if (!state.scanId) return;
  const url = `/api/scans/${encodeURIComponent(state.scanId)}/report${suffix}`;
  const btn = suffix ? $("btn-download-html") : $("btn-download-json");
  if (btn) btn.disabled = true;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      const detail = await readError(res);
      alertBanner(`Report unavailable — ${detail}`);
      return;
    }
    const blob = await res.blob();
    const href = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = href;
    a.download = `scan-report-${state.scanId}${suffix ? ".html" : ".json"}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(href), 4000);
  } catch (err) {
    alertBanner(`Could not download the report: ${err && err.message ? err.message : err}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function alertBanner(message) {
  const box = $("upload-error");
  if (!box) return;
  box.className = "alert alert-bad";
  box.textContent = message;
  show(box);
  setTimeout(() => hide(box), 8000);
}

async function readError(res) {
  try {
    const data = await res.json();
    return data.message || data.error || `HTTP ${res.status}`;
  } catch (_) {
    return `HTTP ${res.status}`;
  }
}

async function submitScan() {
  // Rapid clicks used to enqueue a separate job each, which is how the queue
  // filled up in the first place. Refuse while a submit is in flight AND while
  // a scan is still being polled.
  if (state.submitting || state.poll) return;

  const tab = activeTab();
  const strict = !!($("strict-mode-chk") && $("strict-mode-chk").checked);

  state.generation += 1;
  stopPolling();
  hide($("results-card"));
  hide($("queue-recovery"));

  state.submitting = true;
  const btn = $("start-scan-btn");
  if (btn) btn.disabled = true;

  try {
    if (tab === "upload") {
      const form = new FormData();
      form.append("strict_mode", strict ? "true" : "false");

      if (state.selectedFolder) {
        const files = state.selectedFolder;
        const bytes = files.reduce((sum, f) => sum + (f.size || 0), 0);
        const root = (files[0].webkitRelativePath || "").split("/")[0] || "folder";
        state.lastSubmit = submitScan;
        setStatus("QUEUED", "Uploading folder",
          `Sending ${files.length} ${plural(files.length, "file", "files")} ` +
          `(${formatBytes(bytes)}) from ${root}…`);
        // The third argument becomes the multipart filename; sending the
        // relative path is what lets the server rebuild the folder tree.
        for (const f of files) {
          form.append("file", f, f.webkitRelativePath || f.name);
        }
      } else {
        const input = $("file-input");
        const file = state.selectedFile || (input && input.files && input.files[0]);
        if (!file) {
          showFieldError("upload-error", "Choose a model file or a model folder first.");
          return;
        }
        if (file.size === 0) {
          showFieldError("upload-error", "That file is empty — there is nothing to analyse.");
          return;
        }
        state.lastSubmit = submitScan;
        setStatus("QUEUED", "Uploading", `Sending ${file.name} (${formatBytes(file.size)}) to the scanner…`);
        form.append("file", file, file.name);
      }

      await postScan("/api/scans/upload", { body: form });
    } else {
      const input = $("hf-url-input");
      const url = input ? input.value.trim() : "";
      if (!url) {
        showFieldError("hf-error", "Enter a Hugging Face repository or file URL first.");
        return;
      }
      if (!/^https?:\/\/huggingface\.co\//i.test(url)) {
        showFieldError("hf-error", "Only huggingface.co URLs are supported, e.g. https://huggingface.co/org/model.");
        return;
      }
      if (/\/datasets\//i.test(url)) {
        showFieldError("hf-error", "Dataset repositories are not supported — provide a model repository instead.");
        return;
      }
      state.lastSubmit = submitScan;
      setStatus("QUEUED", "Submitting", "Queueing the repository scan…");

      await postScan("/api/scans/hf", {
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, strict_mode: strict }),
      });
    }
  } finally {
    state.submitting = false;
    if (btn) btn.disabled = false;
  }
}

async function postScan(endpoint, init) {
  const generation = state.generation;
  try {
    const res = await fetch(endpoint, { method: "POST", ...init });
    if (generation !== state.generation) return;
    if (!res.ok) {
      if (res.status === 429) {
        let payload = null;
        try { payload = await res.json(); } catch (_) { /* fall through */ }
        failStatus(
          (payload && payload.message) || "The scan queue is full.",
          payload && payload.error,
          true
        );
        renderCapacity(payload && payload.waiting !== undefined ? payload : null);
        if (!payload || payload.waiting === undefined) refreshCapacity(false);
        return;
      }
      failStatus(await readError(res));
      return;
    }
    const data = await res.json();
    if (generation !== state.generation) return;
    if (!data || !data.scan_id) {
      failStatus("The server accepted the request but returned no scan id.");
      return;
    }
    state.scanId = data.scan_id;
    startPolling(data.scan_id, generation);
  } catch (err) {
    if (generation !== state.generation) return;
    failStatus(`Could not reach the scanner: ${err && err.message ? err.message : err}`);
  }
}

/* ---------------- cancellation ----------------
   The executor caps concurrent scans. Without a cancel path, a slow or wedged
   scan holds a worker, everything behind it piles up in QUEUED, and the server
   answers SCAN_QUEUE_FULL until the process is restarted. These controls are
   the escape hatch.                                                           */

async function cancelCurrentScan() {
  const scanId = state.scanId;
  if (!scanId) return;
  const btn = $("btn-cancel-scan");
  if (btn) btn.disabled = true;

  try {
    const res = await fetch(`/api/scans/${encodeURIComponent(scanId)}`, { method: "DELETE" });

    if (res.status === 404) {
      stopPolling();
      setStatus("CANCELLED", "Scan gone", "That scan is no longer on the server.");
      afterCancel();
      return;
    }
    if (res.status === 409) {
      // Already finished between the click and the request landing.
      return;
    }
    if (!res.ok) {
      alertBanner(`Could not cancel the scan — ${await readError(res)}`);
      return;
    }

    const outcome = await res.json();
    stopPolling();
    state.generation += 1;               // ignore anything still in flight

    if (outcome.was_running) {
      setStatus(
        "CANCELLED",
        "Detached from the scan",
        "It had already started, so the worker keeps unwinding in the background and its report will be discarded. Its slot frees when it stops."
      );
      const card = $("status-card");
      if (card) { card.classList.remove("is-error"); card.classList.add("is-detached"); }
    } else {
      setStatus("CANCELLED", "Scan cancelled", "It had not started yet, so its slot is free again.");
    }
    afterCancel(outcome.capacity);
  } catch (err) {
    alertBanner(`Could not reach the server to cancel: ${err && err.message ? err.message : err}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function cancelAllWaiting() {
  const btn = $("btn-cancel-all");
  if (btn) btn.disabled = true;
  try {
    const res = await fetch("/api/scans/cancel-all", { method: "POST" });
    if (!res.ok) {
      alertBanner(`Could not clear the queue — ${await readError(res)}`);
      return;
    }
    const outcome = await res.json();
    stopPolling();
    state.generation += 1;
    setStatus(
      "CANCELLED",
      "Queue cleared",
      outcome.cancelled_count
        ? `Cancelled ${outcome.cancelled_count} waiting ${outcome.cancelled_count === 1 ? "scan" : "scans"}. You can start a new scan now.`
        : "Nothing was waiting. Any remaining scans are already running — they finish on their own."
    );
    afterCancel(outcome.capacity);
  } catch (err) {
    alertBanner(`Could not reach the server: ${err && err.message ? err.message : err}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function afterCancel(capacity) {
  hide($("btn-cancel-scan"));
  show($("btn-retry"));
  if (capacity) renderCapacity(capacity);
  else refreshCapacity(false);
}

async function refreshCapacity(announce) {
  try {
    const res = await fetch("/api/scans", { cache: "no-store" });
    if (!res.ok) return;
    renderCapacity(await res.json(), announce);
  } catch (_) { /* the panel just stays as it is */ }
}

function renderCapacity(capacity, announce) {
  const panel = $("queue-recovery");
  if (!panel || !capacity) return;

  if (capacity.accepting) {
    hide(panel);
    if (announce) {
      setStatus("CANCELLED", "Queue has room again",
        `${capacity.waiting} waiting, ${capacity.running} running. Start a new scan whenever you are ready.`);
    }
    return;
  }

  setText("queue-recovery-detail",
    `${capacity.waiting} of ${capacity.max_queued} slots are taken by scans waiting to start, ` +
    `and ${capacity.running} of ${capacity.max_active} workers are busy. ` +
    `Cancelling the waiting scans frees space immediately.`);
  show(panel);
}

/* ---------------- polling ---------------- */

function startPolling(scanId, generation) {
  stopPolling();
  const poll = {
    generation,
    timer: null,
    startedAt: Date.now(),
    delay: POLL_START_MS,
    errors: 0,
  };
  state.poll = poll;
  show($("btn-cancel-scan"));
  hide($("btn-retry"));
  hide($("queue-recovery"));

  const tick = async () => {
    if (!state.poll || state.poll !== poll || generation !== state.generation) return;

    if (Date.now() - poll.startedAt > POLL_TIMEOUT_MS) {
      stopPolling();
      failStatus("The scan is taking unusually long and the page stopped waiting. It may still be running on the server.");
      return;
    }

    try {
      const res = await fetch(`/api/scans/${encodeURIComponent(scanId)}`, { cache: "no-store" });
      if (!state.poll || state.poll !== poll || generation !== state.generation) return;

      if (res.status === 404) {
        stopPolling();
        failStatus("That scan is no longer available — results expire after an hour.");
        return;
      }
      if (!res.ok) throw new Error(await readError(res));

      const job = await res.json();
      if (!state.poll || state.poll !== poll || generation !== state.generation) return;
      poll.errors = 0;

      const status = String(job.status || "").toUpperCase();

      if (status === "COMPLETED") {
        stopPolling();
        if (!job.result) {
          failStatus("The scan finished but the server returned no report.");
          return;
        }
        state.result = job.result;
        state.artifactIndex = 0;
        hide($("status-card"));
        renderResults(job.result);
        return;
      }

      if (status === "CANCELLED") {
        stopPolling();
        setStatus("CANCELLED", "Scan cancelled", "This scan was cancelled and its report discarded.");
        hide($("btn-cancel-scan"));
        show($("btn-retry"));
        return;
      }

      if (status === "ERROR") {
        stopPolling();
        failStatus(job.error_message || "The scan terminated with an unexpected error.", job.error_code);
        return;
      }

      setStatus(status || "SCANNING");
      poll.delay = Math.min(POLL_MAX_MS, poll.delay * POLL_BACKOFF);
      poll.timer = setTimeout(tick, poll.delay);
    } catch (err) {
      if (!state.poll || state.poll !== poll || generation !== state.generation) return;
      poll.errors += 1;
      if (poll.errors >= POLL_MAX_CONSECUTIVE_ERRORS) {
        stopPolling();
        failStatus(`Lost contact with the scanner: ${err && err.message ? err.message : err}`);
        return;
      }
      poll.delay = Math.min(POLL_MAX_MS, poll.delay * 2);
      poll.timer = setTimeout(tick, poll.delay);
    }
  };

  poll.timer = setTimeout(tick, POLL_START_MS);
}

function stopPolling() {
  if (state.poll && state.poll.timer) clearTimeout(state.poll.timer);
  state.poll = null;
  hide($("btn-cancel-scan"));
}

/* ---------------- status card ---------------- */

function setStatus(status, titleOverride, messageOverride) {
  const card = $("status-card");
  const key = String(status || "SCANNING").toUpperCase();
  const [defTitle, defMsg] = STATUS_COPY[key] || STATUS_COPY.SCANNING;

  show(card);
  if (card) {
    card.classList.toggle("is-error", key === "ERROR");
    if (key !== "CANCELLED") card.classList.remove("is-detached");
  }

  setText("status-title", titleOverride || defTitle);
  setText("status-message", messageOverride || defMsg);

  const badge = $("status-badge");
  if (badge) {
    badge.textContent = key;
    badge.className = `badge badge-${key.toLowerCase()}`;
  }

  const order = ["QUEUED", "ACQUIRING", "SCANNING", "COMPLETED"];
  const idx = order.indexOf(key);
  document.querySelectorAll(".step").forEach((step) => {
    const pos = order.indexOf(step.dataset.step);
    step.classList.remove("is-done", "is-current");
    if (key === "ERROR" || key === "CANCELLED") {
      if (pos === 0) step.classList.add("is-current");
      return;
    }
    if (idx < 0) return;
    if (pos < idx) step.classList.add("is-done");
    else if (pos === idx) step.classList.add("is-current");
  });

  if (key !== "ERROR") hide($("status-tech"));
}

function failStatus(rawMessage, errorCode, useRaw) {
  stopPolling();
  // Some server messages already explain the situation better than any
  // client-side rewrite could (queue capacity, for one) - pass those through.
  const friendly = useRaw ? String(rawMessage) : humaniseError(rawMessage);
  setStatus("ERROR", "Scan failed", friendly);

  const raw = String(rawMessage || "");
  const tech = $("status-tech");
  if (tech && raw && raw !== friendly) {
    tech.textContent = errorCode ? `${errorCode}: ${raw}` : raw;
    show(tech);
  } else {
    hide(tech);
  }
  show($("btn-retry"));
  scrollToEl($("status-card"));
}

function humaniseError(raw) {
  if (!raw) return "An unexpected error occurred. Please try again.";
  const r = String(raw).toLowerCase();
  const rules = [
    [/dataset/, "Dataset repositories are not supported — provide a model repository URL."],
    [/401|unauthori[sz]ed|gated/, "That repository needs authorisation. Only public Hugging Face repositories can be scanned."],
    [/403|forbidden/, "Access denied — the repository may be private, gated or restricted."],
    [/404|not found/, "Repository or file not found. Check the URL and try again."],
    [/429|rate limit|too many/, "Hugging Face is rate-limiting requests. Wait a moment and retry."],
    [/timed? ?out|timeout/, "The connection timed out. Check your network and try again."],
    [/network|urlopen|connection|tunnel|socket|dns|resolve/, "The scanner could not reach the network. Check connectivity or any proxy settings."],
    [/no supported|no model artifact/, "No supported model files were found at that location."],
    [/payload too large|413/, "That file is larger than the 1 GB upload limit."],
    [/queue|capacity/, "The scan queue is full right now. Wait for running scans to finish and retry."],
    [/multipart|content-type/, "The upload was rejected as malformed. Reselect the file and try again."],
  ];
  for (const [pattern, message] of rules) if (pattern.test(r)) return message;
  const s = String(raw);
  return s.length > 220 ? `${s.slice(0, 220)}…` : s;
}

function showFieldError(id, message) {
  const box = $(id);
  if (!box) return;
  box.className = "alert alert-warn";
  box.textContent = message;
  show(box);
  setTimeout(() => hide(box), 7000);
}

/* ---------------- verdict ---------------- */

const VERDICT_ICONS = {
  PASS: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
  WARN: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v5"/><path d="M12 17.5h.01"/><path d="M10.3 3.9 1.9 18a2 2 0 0 0 1.7 3h16.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  FAIL: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M15 9l-6 6M9 9l6 6"/></svg>',
  ERROR: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16.5h.01"/></svg>',
  UNSUPPORTED: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12h8"/></svg>',
  INCONCLUSIVE: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><circle cx="12" cy="16" r=".6" fill="currentColor" stroke="none"/></svg>',
};

/* Verdict is computed across every artifact, not just the first one.
   Precedence: FAIL > ERROR > INCONCLUSIVE > WARN > UNSUPPORTED > PASS. */
function deriveVerdict(result) {
  const summary = result.summary || {};
  const artifacts = result.artifacts || [];
  const sev = summary.findings_by_severity || {};
  const critical = Number(sev.CRITICAL || 0);
  const high = Number(sev.HIGH || 0);
  const summaryOutcome = String(summary.overall_outcome || "").toUpperCase();

  // Count all findings, but exclude format-classification INFO findings from
  // the security-findings total used for the WARN verdict.  Coverage-gap
  // artifacts (TFJS_WEIGHT_SHARD) produce only G002 INFO and are counted
  // separately under coverage_gap_artifacts, not as security findings.
  const coverageGaps = Number(summary.coverage_gap_artifacts || 0);
  const totalFindings = SEVERITIES.reduce((sum, k) => sum + Number(sev[k] || 0), 0);
  // Security findings = all findings except INFO from pure coverage-gap artifacts.
  // Simplification: if overall_outcome is INCONCLUSIVE and no HIGH/CRITICAL/MEDIUM/LOW
  // findings exist, all INFO findings are format-gap coverage notices.
  const securityFindings = (summaryOutcome === "INCONCLUSIVE" && critical === 0 && high === 0)
    ? Number(sev.MEDIUM || 0) + Number(sev.LOW || 0)
    : totalFindings;

  if (!artifacts.length) {
    return {
      label: "UNSUPPORTED", cls: "verdict-unsupported",
      desc: "No supported model artifacts were found at this target, so nothing could be analysed.",
    };
  }

  const outcomes = artifacts.map((a) => String(a.scan_outcome || "").toUpperCase());
  const errored = outcomes.filter((o) => o === "ERROR").length;
  const unsupported = outcomes.filter((o) => o === "UNSUPPORTED").length;

  if (critical > 0 || high > 0) {
    return {
      label: "FAIL", cls: "verdict-fail",
      desc: `${critical + high} critical or high-severity ${plural(critical + high, "finding", "findings")} detected across ${artifacts.length} ${plural(artifacts.length, "artifact", "artifacts")}. Do not load this model.`,
    };
  }
  if (errored > 0) {
    return {
      label: "ERROR", cls: "verdict-error",
      desc: `${errored} of ${artifacts.length} ${plural(artifacts.length, "artifact", "artifacts")} could not be parsed. Coverage is incomplete — treat the result as inconclusive.`,
    };
  }
  if (summaryOutcome === "INCONCLUSIVE" && securityFindings === 0) {
    return {
      label: "INCONCLUSIVE", cls: "verdict-inconclusive",
      desc: `${coverageGaps} of ${artifacts.length} ${plural(artifacts.length, "artifact", "artifacts")} could not be analyzed — format not supported by this scanner. Coverage is incomplete; treat the result as inconclusive.`,
    };
  }
  if (securityFindings > 0) {
    return {
      label: "WARN", cls: "verdict-warn",
      desc: `${securityFindings} medium or lower-severity ${plural(securityFindings, "finding", "findings")} detected. Review before deployment.`,
    };
  }
  if (unsupported === artifacts.length) {
    return {
      label: "UNSUPPORTED", cls: "verdict-unsupported",
      desc: "Every artifact is outside the scanner's supported formats, so no analysis was performed.",
    };
  }
  return {
    label: "PASS", cls: "verdict-pass",
    desc: `No security findings in static analysis of ${artifacts.length} ${plural(artifacts.length, "artifact", "artifacts")}.`,
  };
}

const plural = (n, one, many) => (Number(n) === 1 ? one : many);

/* ---------------- results ---------------- */

function renderResults(result) {
  const card = $("results-card");
  if (!card) return;

  const summary = result.summary || {};
  const artifacts = Array.isArray(result.artifacts) ? result.artifacts : [];
  show(card);

  /* verdict */
  const verdict = deriveVerdict(result);
  const banner = $("verdict-banner");
  if (banner) banner.className = `verdict ${verdict.cls}`;
  const icon = $("verdict-icon");
  if (icon) icon.innerHTML = VERDICT_ICONS[verdict.label] || "";
  setText("verdict-text", verdict.label);
  setText("verdict-desc", verdict.desc);

  /* score gauge — worst score across all artifacts */
  const score = Number(
    summary.max_risk_score !== undefined && summary.max_risk_score !== null
      ? summary.max_risk_score
      : artifacts.reduce((m, a) => Math.max(m, Number((a.risk_score || {}).final_score || 0)), 0)
  );
  setText("res-score", Math.round(score));
  const gauge = $("gauge");
  if (gauge) {
    gauge.style.setProperty("--pct", "0");
    requestAnimationFrame(() => gauge.style.setProperty("--pct", String(Math.max(0, Math.min(100, score)))));
  }

  /* KPIs */
  const worstBand = artifacts.reduce((worst, a) => {
    const band = String((a.risk_score || {}).risk_band || "");
    const order = ["CLEAN", "UNANALYZED", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "CRITICAL_RISK"];
    return order.indexOf(band) > order.indexOf(worst) ? band : worst;
  }, "CLEAN");
  setText("res-band", artifacts.length ? pretty(worstBand) : "—");
  setText("res-outcome", formatOutcome(summary.overall_outcome, summary.findings_by_severity));
  setText("res-exit-code", summary.exit_code !== undefined ? summary.exit_code : "—");
  setText("res-duration", formatDuration(result.total_duration_ms));

  const supported = artifacts.filter((a) => String(a.scan_outcome).toUpperCase() !== "UNSUPPORTED").length;
  setText("res-artifact-count", artifacts.length ? `${artifacts.length}${artifacts.length !== supported ? ` (${supported} analysed)` : ""}` : "0");

  /* posture across the whole scan */
  renderPosture(summary.findings_by_severity || {});
  setText("posture-scope", artifacts.length > 1 ? `Aggregated across ${artifacts.length} artifacts` : "This artifact");

  /* artifact picker — the old UI silently dropped artifacts[1..n] */
  renderArtifactPicker(artifacts);

  state.artifactIndex = pickWorstArtifact(artifacts);
  renderArtifact(state.artifactIndex);

  scrollToEl(card);
}

function pickWorstArtifact(artifacts) {
  let best = 0;
  let bestKey = [-1, -1];
  artifacts.forEach((a, i) => {
    const findings = a.findings || [];
    const worstSev = findings.reduce((m, f) => Math.min(m, SEV_RANK[safeSev(f.severity)]), 9);
    const key = [-worstSev, Number((a.risk_score || {}).final_score || 0)];
    if (key[0] > bestKey[0] || (key[0] === bestKey[0] && key[1] > bestKey[1])) {
      bestKey = key;
      best = i;
    }
  });
  return best;
}

function renderPosture(counts) {
  const host = $("posture");
  if (!host) return;
  host.textContent = "";
  const max = Math.max(1, ...SEVERITIES.map((s) => Number(counts[s] || 0)));

  SEVERITIES.forEach((sev) => {
    const n = Number(counts[sev] || 0);
    const row = el("div", `prow${n === 0 ? " is-zero" : ""}`);

    const label = el("span", "plabel", SEV_LABEL[sev].toUpperCase());
    label.style.color = `var(--sev-${sev.toLowerCase()})`;

    const track = el("div", "ptrack");
    const fill = el("div", "pfill");
    fill.style.background = `var(--sev-${sev.toLowerCase()})`;
    track.appendChild(fill);

    row.append(label, track, el("span", "pcount", n));
    host.appendChild(row);

    requestAnimationFrame(() => {
      fill.style.width = n === 0 ? "0%" : `${Math.max(4, Math.round((n / max) * 100))}%`;
    });
  });
}

function renderArtifactPicker(artifacts) {
  const card = $("artifact-picker-card");
  const list = $("artifact-list");
  if (!card || !list) return;

  if (artifacts.length < 2) { hide(card); return; }
  show(card);
  setText("artifact-picker-sub", `${artifacts.length} files scanned — select one to inspect`);
  list.textContent = "";

  artifacts.forEach((a, i) => {
    const findings = a.findings || [];
    const worst = findings.reduce((m, f) => Math.min(m, SEV_RANK[safeSev(f.severity)]), 9);
    const worstSev = worst < 9 ? SEVERITIES[worst] : null;
    const outcome = String(a.scan_outcome || "").toUpperCase();

    const item = el("button", "aitem");
    item.type = "button";
    item.setAttribute("role", "tab");

    const dot = el("span", "aitem-dot");
    dot.style.background = worstSev
      ? `var(--sev-${worstSev.toLowerCase()})`
      : outcome === "ERROR" ? "var(--bad)"
      : outcome === "UNSUPPORTED" ? "var(--muted)" : "var(--ok)";

    const main = el("span", "aitem-main");
    main.append(
      el("span", "aitem-name", a.name || a.source_origin || `Artifact ${i + 1}`),
      el("span", "aitem-meta",
        `${pretty(a.detected_format || "UNKNOWN")} · ${formatBytes(a.size_bytes)} · ${findings.length} ${plural(findings.length, "finding", "findings")}`)
    );

    const tags = el("span", "aitem-tags");
    if (worstSev) tags.appendChild(el("span", `badge badge-${worstSev.toLowerCase()}`, SEV_LABEL[worstSev]));
    if (outcome === "ERROR") tags.appendChild(el("span", "badge badge-error", "ERROR"));
    if (outcome === "UNSUPPORTED") tags.appendChild(el("span", "badge badge-rule", "UNSUPPORTED"));

    item.append(dot, main, tags);
    item.addEventListener("click", () => {
      state.artifactIndex = i;
      renderArtifact(i);
      scrollToEl($("findings-list") ? $("findings-list").closest(".card") : null);
    });
    list.appendChild(item);
  });
}

function renderArtifact(index) {
  const artifacts = (state.result && state.result.artifacts) || [];
  const artifact = artifacts[index];

  document.querySelectorAll(".aitem").forEach((node, i) => {
    node.classList.toggle("is-active", i === index);
    node.setAttribute("aria-selected", i === index ? "true" : "false");
  });

  renderIdentity(artifact, artifacts.length);
  renderDiagnostics(artifact);
  state.findingSev = "ALL";
  state.findingQuery = "";
  const search = $("findings-search");
  if (search) search.value = "";
  renderSevFilter(artifact);
  renderFindings();
  renderCorrelation(artifact);
}

function renderIdentity(artifact, total) {
  const host = $("ident");
  if (!host) return;
  host.textContent = "";
  if (!artifact) {
    host.appendChild(identRow("Artifact", "No artifact was returned by the scan", false, true));
    return;
  }

  const rows = [
    ["Name", artifact.name || "—", false],
    ["Format", pretty(artifact.detected_format || "UNKNOWN"), false],
    ["Size", formatBytes(artifact.size_bytes), false],
    ["Outcome", pretty(artifact.scan_outcome || "—"), false],
    ["Risk band", pretty((artifact.risk_score || {}).risk_band || "—"), false],
    ["Risk score", `${(artifact.risk_score || {}).final_score ?? "—"} / 100`, false],
    ["Source", pretty(artifact.input_source || "—"), false],
    ["Analysis time", formatDuration(artifact.duration_ms), false],
    ["Scanned at", formatTimestampLocal((state.result || {}).timestamp_utc), false],
  ];
  rows.forEach(([k, v]) => host.appendChild(identRow(k, v, false)));

  if (artifact.source_origin && artifact.source_origin !== artifact.name) {
    host.appendChild(identRow("Origin", artifact.source_origin, true, true));
  }
  host.appendChild(identRow("SHA-256", artifact.sha256 || "—", true, true));

  setText("findings-scope", total > 1 ? `In ${artifact.name || "the selected artifact"}` : "");
}

function identRow(key, value, mono, full) {
  const row = el("div", `ident-row${full ? " ident-full" : ""}`);
  row.append(el("span", "ident-k", key), el("span", `ident-v${mono ? " mono" : ""}`, value));
  return row;
}

/* Warnings and errors used to be dropped on the floor — this is where the
   "why is this UNSUPPORTED" explanation actually lives. */
function renderDiagnostics(artifact) {
  const host = $("diagnostics");
  if (!host) return;
  host.textContent = "";
  if (!artifact) return;

  (artifact.errors || []).forEach((e) => {
    host.appendChild(diagnostic("diag-error", "!", e.error_code || "Error", e.message, e.exception_type || e.traceback_summary));
  });
  (artifact.warnings || []).forEach((w) => {
    host.appendChild(diagnostic("diag-warn", "!", w.warning_code || "Warning", w.message, w.context));
  });
}

function diagnostic(cls, glyph, title, message, context) {
  const box = el("div", `diag ${cls}`);
  box.appendChild(el("span", "diag-icon", glyph));
  const body = el("div", "diag-body");
  body.appendChild(el("div", "diag-title", pretty(title)));
  if (message) body.appendChild(el("div", "diag-msg", message));
  if (context) body.appendChild(el("div", "diag-ctx", context));
  box.appendChild(body);
  return box;
}

/* ---------------- findings ---------------- */

function initFindingFilters() {
  const search = $("findings-search");
  if (search) {
    search.addEventListener("input", () => {
      state.findingQuery = search.value.trim().toLowerCase();
      renderFindings();
    });
  }
}

function renderSevFilter(artifact) {
  const host = $("findings-sevfilter");
  if (!host) return;
  host.textContent = "";
  const findings = (artifact && artifact.findings) || [];
  const counts = {};
  findings.forEach((f) => { const s = safeSev(f.severity); counts[s] = (counts[s] || 0) + 1; });

  const make = (sev, label, n, disabled) => {
    const btn = el("button", `sevbtn${state.findingSev === sev ? " is-active" : ""}`, `${label} ${n}`);
    btn.type = "button";
    btn.dataset.sev = sev;
    if (disabled) btn.disabled = true;
    btn.addEventListener("click", () => {
      state.findingSev = sev;
      host.querySelectorAll(".sevbtn").forEach((b) => b.classList.toggle("is-active", b === btn));
      renderFindings();
    });
    return btn;
  };

  host.appendChild(make("ALL", "All", findings.length, false));
  SEVERITIES.forEach((sev) => {
    const n = counts[sev] || 0;
    if (n > 0) host.appendChild(make(sev, SEV_LABEL[sev], n, false));
  });
  if (!findings.length) hide(host); else show(host);
}

function renderFindings() {
  const artifacts = (state.result && state.result.artifacts) || [];
  const artifact = artifacts[state.artifactIndex];
  const host = $("findings-list");
  const empty = $("findings-empty");
  if (!host) return;

  host.textContent = "";
  hide(empty);

  const all = ((artifact && artifact.findings) || []).slice().sort((a, b) => {
    const d = SEV_RANK[safeSev(a.severity)] - SEV_RANK[safeSev(b.severity)];
    if (d !== 0) return d;
    return String(a.rule_id || "").localeCompare(String(b.rule_id || ""));
  });

  setText("findings-total-count", all.length);

  const q = state.findingQuery;
  const visible = all.filter((f) => {
    if (state.findingSev !== "ALL" && safeSev(f.severity) !== state.findingSev) return false;
    if (!q) return true;
    return [f.rule_id, f.rule_name, f.category, f.description, f.remediation,
            (f.evidence || {}).matched_value, (f.evidence || {}).opcode]
      .some((v) => String(v || "").toLowerCase().includes(q));
  });

  if (!all.length) {
    if (empty) {
      const outcome = String((artifact || {}).scan_outcome || "").toUpperCase();
      if (!artifact) {
        empty.className = "empty";
        empty.textContent = "No artifacts were returned for this target.";
      } else if (outcome === "UNSUPPORTED") {
        empty.className = "empty";
        empty.textContent = "This format is outside the scanner's coverage, so no rules were evaluated. See the diagnostics above.";
      } else if (outcome === "ERROR") {
        empty.className = "empty";
        empty.textContent = "The artifact could not be parsed, so findings are incomplete. See the diagnostics above.";
      } else {
        empty.className = "empty empty-ok";
        empty.textContent = "No security findings — every rule evaluated clean against this artifact.";
      }
      show(empty);
    }
    return;
  }

  if (!visible.length) {
    if (empty) {
      empty.className = "empty";
      empty.textContent = "No findings match the current filter.";
      show(empty);
    }
    return;
  }

  visible.forEach((f) => host.appendChild(findingCard(f)));
}

function findingCard(finding) {
  const sev = safeSev(finding.severity);
  const card = el("article", "finding");
  card.dataset.sev = sev;

  /* header */
  const head = el("div", "f-head");
  const tags = el("div", "f-tags");
  tags.appendChild(el("span", "badge badge-rule", finding.rule_id || "—"));
  tags.appendChild(el("span", `badge badge-${sev.toLowerCase()}`, sev));
  if (finding.confidence) tags.appendChild(el("span", "badge badge-conf", `${finding.confidence} confidence`));
  if (finding.category) tags.appendChild(el("span", "ref", finding.category));
  head.appendChild(tags);
  head.appendChild(el("div", "f-name", finding.rule_name || "Unnamed rule"));
  if (finding.description) head.appendChild(el("div", "f-desc", finding.description));

  const refs = el("div", "f-refs");
  (finding.cwe_ids || []).forEach((c) => refs.appendChild(el("span", "ref", c)));
  (finding.cve_ids || []).forEach((c) => refs.appendChild(el("span", "ref", c)));
  refs.appendChild(el("span", "ref", finding.owasp_category || "OWASP ML: Not directly mapped"));
  if (finding.mitre_technique) refs.appendChild(el("span", "ref", finding.mitre_technique));
  if (refs.childElementCount) head.appendChild(refs);
  card.appendChild(head);

  /* panels */
  const panels = el("div", "f-panels");
  const ev = finding.evidence || {};
  const evGrid = el("div", "ev");
  const evRow = (k, v, mono) => {
    if (v === undefined || v === null || v === "") return;
    evGrid.appendChild(el("span", "ev-k", k));
    evGrid.appendChild(el("span", mono === false ? "" : "ev-v", v));
  };

  evRow("Coordinate", pretty(ev.coordinate_space || "artifact"), false);
  if (ev.archive_member) evRow("Archive member", ev.archive_member);
  if (ev.byte_offset !== undefined && ev.byte_offset !== null) {
    evRow("Byte offset", `0x${Number(ev.byte_offset).toString(16).toUpperCase().padStart(4, "0")} (${ev.byte_offset})`);
  }
  if (ev.byte_length !== undefined && ev.byte_length !== null) evRow("Byte length", `${ev.byte_length} bytes`);
  if (ev.opcode) evRow("Opcode", ev.opcode);
  if (ev.matched_value) evRow("Matched value", ev.matched_value);
  evRow("Artifact SHA-256", finding.artifact_sha256);

  const blob = ev.raw_hex || ev.context_snippet;
  if (blob) evGrid.appendChild(el("pre", "hexdump", blob));

  if (evGrid.childElementCount) panels.appendChild(panel("Static evidence", evGrid, true));
  if (finding.remediation) panels.appendChild(panel("Remediation", el("div", "remediation", finding.remediation), false));

  const corr = finding.correlated_finding_ids || [];
  if (corr.length) {
    const wrap = el("div", "corr-ids");
    corr.forEach((id) => wrap.appendChild(el("span", "ref", id)));
    panels.appendChild(panel(`Correlated findings (${corr.length})`, wrap, false));
  }

  if (panels.childElementCount) card.appendChild(panels);
  return card;
}

function panel(title, body, open) {
  const details = el("details", "f-panel");
  if (open) details.open = true;
  const summary = el("summary", null, title);
  details.appendChild(summary);
  const wrap = el("div", "f-panel-body");
  wrap.appendChild(body);
  details.appendChild(wrap);
  return details;
}

function renderCorrelation(artifact) {
  const section = $("correlation-section");
  const list = $("correlation-list");
  if (!section || !list) return;

  const correlated = ((artifact && artifact.findings) || [])
    .filter((f) => (f.correlated_finding_ids || []).length > 0);

  if (!correlated.length) { hide(section); return; }
  show(section);
  list.textContent = "";

  correlated.forEach((f) => {
    const row = el("div", "corr-row");
    row.appendChild(el("span", "badge badge-rule", f.rule_id || "—"));
    row.appendChild(el("span", null, f.rule_name || ""));
    row.appendChild(el("span", "corr-arrow", "→"));
    const ids = el("span", "corr-ids");
    (f.correlated_finding_ids || []).forEach((id) => ids.appendChild(el("span", "ref", id)));
    row.appendChild(ids);
    list.appendChild(row);
  });
}

/* ---------------- rule catalog ---------------- */

async function preloadRuleCount() {
  try {
    const res = await fetch("/api/rules", { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();
    state.rules = data.rules || [];
    state.rulesLoaded = true;
    const total = data.total ?? state.rules.length;
    setText("rule-count-chip", total);
    setText("rules-total", total);
  } catch (_) {
    setText("rule-count-chip", "—");
  }
}

function initRulesModal() {
  const modal = $("rules-modal");
  const open = $("btn-open-rules");
  const close = $("btn-close-rules");
  const search = $("rules-search-input");
  if (!modal || !open) return;

  let lastFocus = null;

  const openModal = async () => {
    lastFocus = document.activeElement;
    show(modal);
    document.body.style.overflow = "hidden";
    if (!state.rulesLoaded) {
      await preloadRuleCount();
      if (!state.rulesLoaded) {
        const list = $("rules-list");
        if (list) list.textContent = "";
        const empty = $("rules-empty");
        if (empty) { empty.textContent = "Could not load the rule catalog from the server."; show(empty); }
        return;
      }
    }
    renderRuleFilters();
    renderRules();
    if (search) search.focus();
  };

  const closeModal = () => {
    hide(modal);
    document.body.style.overflow = "";
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  };

  open.addEventListener("click", openModal);
  if (close) close.addEventListener("click", closeModal);
  modal.querySelectorAll("[data-close-modal]").forEach((n) => n.addEventListener("click", closeModal));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("is-hidden")) closeModal();
  });
  if (search) search.addEventListener("input", renderRules);
}

function renderRuleFilters() {
  const host = $("rules-sevfilter");
  if (!host) return;
  host.textContent = "";
  const counts = {};
  state.rules.forEach((r) => { const s = safeSev(r.severity); counts[s] = (counts[s] || 0) + 1; });

  const make = (sev, label, n) => {
    const btn = el("button", `sevbtn${state.ruleSev === sev ? " is-active" : ""}`, `${label} ${n}`);
    btn.type = "button";
    btn.dataset.sev = sev;
    btn.addEventListener("click", () => {
      state.ruleSev = sev;
      host.querySelectorAll(".sevbtn").forEach((b) => b.classList.toggle("is-active", b === btn));
      renderRules();
    });
    return btn;
  };

  host.appendChild(make("ALL", "All", state.rules.length));
  SEVERITIES.forEach((sev) => { if (counts[sev]) host.appendChild(make(sev, SEV_LABEL[sev], counts[sev])); });
}

function renderRules() {
  const list = $("rules-list");
  const empty = $("rules-empty");
  if (!list) return;

  const q = ($("rules-search-input")?.value || "").trim().toLowerCase();
  const filtered = state.rules
    .filter((r) => {
      if (state.ruleSev !== "ALL" && safeSev(r.severity) !== state.ruleSev) return false;
      if (!q) return true;
      return [r.rule_id, r.name, r.category, r.affected_format, r.description]
        .some((v) => String(v || "").toLowerCase().includes(q));
    })
    .sort((a, b) => {
      const d = SEV_RANK[safeSev(a.severity)] - SEV_RANK[safeSev(b.severity)];
      return d !== 0 ? d : String(a.rule_id).localeCompare(String(b.rule_id));
    });

  setText("rules-shown", filtered.length);
  setText("rules-total", state.rules.length);

  list.textContent = "";
  if (!filtered.length) {
    if (empty) { empty.textContent = "No rules match the current filter."; show(empty); }
    return;
  }
  hide(empty);

  filtered.forEach((r) => {
    const sev = safeSev(r.severity);
    const item = el("details", "rule");
    item.dataset.sev = sev;

    const summary = el("summary");
    summary.append(
      el("span", "badge badge-rule", r.rule_id),
      el("span", "rule-name", r.name || ""),
      el("span", `badge badge-${sev.toLowerCase()}`, SEV_LABEL[sev]),
      el("span", "rule-fmt", r.affected_format || "")
    );
    item.appendChild(summary);

    const body = el("div", "rule-body");
    if (r.description) {
      const d = el("div");
      d.append(el("span", "ev-k", "Description"), document.createTextNode(r.description));
      body.appendChild(d);
    }
    if (r.remediation) {
      const d = el("div");
      d.append(el("span", "ev-k", "Remediation"), document.createTextNode(r.remediation));
      body.appendChild(d);
    }
    const meta = [];
    if (r.category) meta.push(`Category: ${r.category}`);
    if ((r.cwe_ids || []).length) meta.push(`CWE: ${r.cwe_ids.join(", ")}`);
    if ((r.cve_ids || []).length) meta.push(`CVE: ${r.cve_ids.join(", ")}`);
    meta.push(`OWASP ML: ${r.owasp_category || "Not directly mapped"}`);
    if (meta.length) body.appendChild(el("div", "rule-fmt", meta.join("  ·  ")));

    item.appendChild(body);
    list.appendChild(item);
  });
}
