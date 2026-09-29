"""
model_scanner.presentation.html_report
=======================================

Professional self-contained HTML security assessment report renderer.

Generated from a ReportModel — the same data source used by JSON and PDF.

REQUIREMENTS MET
----------------
- Single HTML file with inline CSS (no external resources)
- No CDN, no external fonts, no external JavaScript
- Vanilla JS only for minor interactivity (fully readable without JS)
- Responsive, print-friendly layout with @page and @media print
- ALL interpolated values are HTML-escaped (artifact paths, evidence,
  metadata, etc. are untrusted)
- Secrets are already redacted in ReportModel before reaching this renderer

SECURITY GUARANTEE
------------------
This module performs PURE IN-MEMORY TEXT GENERATION.
It NEVER executes model code, deserializes binary, or makes network requests.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Final

from model_scanner.presentation.report_model import (
    ReportFinding,
    ReportModel,
    ScanReportModel,
)

# ---------------------------------------------------------------------------
# Severity styling helpers
# ---------------------------------------------------------------------------

_SEVERITY_COLOR: dict[str, str] = {
    "CRITICAL": "#b91c1c",
    "HIGH": "#c2410c",
    "MEDIUM": "#b45309",
    "LOW": "#4d7c0f",
    "INFO": "#0369a1",
}

_SEVERITY_BG: dict[str, str] = {
    "CRITICAL": "#fef2f2",
    "HIGH": "#fff7ed",
    "MEDIUM": "#fffbeb",
    "LOW": "#f7fee7",
    "INFO": "#e0f2fe",
}

_VERDICT_COLOR: dict[str, str] = {
    "FAIL": "#b91c1c",
    "WARN": "#b45309",
    "PASS": "#15803d",
    "UNSUPPORTED": "#475569",
    "UNANALYZED": "#92400e",
}

_RISK_BAND_COLOR: dict[str, str] = {
    "CRITICAL_RISK": "#7f1d1d",
    "HIGH_RISK": "#7c2d12",
    "MEDIUM_RISK": "#78350f",
    "LOW_RISK": "#365314",
    "CLEAN": "#14532d",
}


def _e(value: object) -> str:
    """HTML-escape any value."""
    return html.escape(str(value))


def _sev_badge(severity: str) -> str:
    color = _SEVERITY_COLOR.get(severity, "#374151")
    bg = _SEVERITY_BG.get(severity, "#f3f4f6")
    return (
        f'<span class="badge" style="background:{bg};color:{color};'
        f'border:1px solid {color};">{_e(severity)}</span>'
    )


# ---------------------------------------------------------------------------
# CSS (inline, no external refs)
# ---------------------------------------------------------------------------

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --font-mono: 'Courier New', Courier, monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --color-bg: #f8fafc;
  --color-card: #ffffff;
  --color-border: #e2e8f0;
  --color-text: #1e293b;
  --color-muted: #64748b;
  --color-accent: #1d4ed8;
}
body{font-family:var(--font-sans);background:var(--color-bg);color:var(--color-text);font-size:14px;line-height:1.6}
a{color:var(--color-accent)}
h1{font-size:1.6rem;font-weight:700}
h2{font-size:1.25rem;font-weight:700;margin-bottom:.5rem}
h3{font-size:1.05rem;font-weight:600;margin-bottom:.4rem}
h4{font-size:.95rem;font-weight:600;margin-bottom:.3rem}
h5{font-size:.85rem;font-weight:600;margin-bottom:.2rem;color:var(--color-muted)}

/* ---- layout ---- */
.report-wrap{max-width:1000px;margin:0 auto;padding:1.5rem 1rem}
.card{background:var(--color-card);border:1px solid var(--color-border);border-radius:.5rem;padding:1.25rem;margin-bottom:1.25rem}
.card-header{border-bottom:1px solid var(--color-border);padding-bottom:.75rem;margin-bottom:.75rem}
section{margin-bottom:1.5rem}

/* ---- report header ---- */
.report-header{background:linear-gradient(135deg,#1e3a5f 0%,#1d4ed8 100%);color:#fff;padding:1.75rem;border-radius:.5rem;margin-bottom:1.5rem}
.report-title{font-size:1.5rem;font-weight:700;letter-spacing:.02em}
.report-subtitle{font-size:.85rem;opacity:.85;margin-top:.25rem}
.report-meta-row{display:flex;gap:2rem;margin-top:1rem;flex-wrap:wrap}
.report-meta-item{font-size:.8rem}
.report-meta-item span{opacity:.75;display:block}
.report-meta-item strong{font-size:.95rem}

/* ---- disclaimer ---- */
.disclaimer{background:#fefce8;border:1px solid #fde047;border-radius:.4rem;padding:.9rem 1rem;font-size:.82rem;color:#713f12;margin-bottom:1rem}
.disclaimer-icon{font-weight:700;margin-right:.4rem}

/* ---- verdict banner ---- */
.verdict-banner{display:flex;align-items:center;gap:1rem;padding:1rem 1.25rem;border-radius:.4rem;margin-bottom:1rem;border:2px solid}
.verdict-FAIL{background:#fef2f2;border-color:#fca5a5}
.verdict-WARN{background:#fffbeb;border-color:#fde68a}
.verdict-PASS{background:#f0fdf4;border-color:#86efac}
.verdict-UNSUPPORTED{background:#f1f5f9;border-color:#94a3b8}
.verdict-UNANALYZED{background:#fefce8;border-color:#fde68a}
.verdict-label{font-size:1.2rem;font-weight:800}
.verdict-desc{font-size:.85rem;color:var(--color-muted)}

/* ---- risk score ---- */
.risk-score-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.75rem;margin-bottom:1rem}
.risk-tile{border:1px solid var(--color-border);border-radius:.4rem;padding:.75rem;text-align:center}
.risk-tile .tile-val{font-size:1.5rem;font-weight:700;display:block}
.risk-tile .tile-lbl{font-size:.75rem;color:var(--color-muted);display:block;margin-top:.1rem}

/* ---- severity cards ---- */
.severity-grid{display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1rem}
.sev-card{flex:1;min-width:90px;border-radius:.4rem;padding:.6rem .8rem;text-align:center;border:1px solid}
.sev-card .sev-count{font-size:1.5rem;font-weight:800;display:block}
.sev-card .sev-name{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.sev-CRITICAL{background:#fef2f2;border-color:#fca5a5;color:#b91c1c}
.sev-HIGH{background:#fff7ed;border-color:#fdba74;color:#c2410c}
.sev-MEDIUM{background:#fffbeb;border-color:#fde68a;color:#b45309}
.sev-LOW{background:#f7fee7;border-color:#bbf7d0;color:#4d7c0f}
.sev-INFO{background:#e0f2fe;border-color:#7dd3fc;color:#0369a1}

/* ---- findings table ---- */
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.83rem}
th{background:#f1f5f9;text-align:left;padding:.55rem .75rem;border:1px solid var(--color-border);font-weight:600;white-space:nowrap}
td{padding:.5rem .75rem;border:1px solid var(--color-border);vertical-align:top}
tr:nth-child(even){background:#f8fafc}

/* ---- badge ---- */
.badge{display:inline-block;padding:.15rem .5rem;border-radius:.25rem;font-size:.75rem;font-weight:600;letter-spacing:.02em}
.badge-pass{background:#dcfce7;color:#15803d;border:1px solid #86efac}
.badge-warn{background:#fefce8;color:#854d0e;border:1px solid #fde68a}
.badge-fail{background:#fef2f2;color:#b91c1c;border:1px solid #fca5a5}

/* ---- finding card ---- */
.finding-card{border:1px solid var(--color-border);border-radius:.5rem;margin-bottom:1rem;overflow:hidden}
.finding-card-header{padding:.65rem 1rem;display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;cursor:pointer;user-select:none;background:#f8fafc}
.finding-card-header:hover{background:#f1f5f9}
.finding-rule-id{font-family:var(--font-mono);font-weight:700;font-size:.9rem}
.finding-rule-name{font-weight:600;font-size:.9rem;flex:1}
.finding-card-body{padding:1rem;border-top:1px solid var(--color-border)}
.finding-section{margin-bottom:.9rem}
.finding-section h5{margin-bottom:.25rem}
.finding-section p,.finding-section div{font-size:.84rem}
.evidence-block{background:#1e293b;color:#e2e8f0;font-family:var(--font-mono);font-size:.78rem;padding:.75rem;border-radius:.35rem;overflow-x:auto;white-space:pre-wrap;word-break:break-all}
.evidence-row{margin-bottom:.25rem}
.evidence-key{color:#94a3b8}
.threat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.5rem}
.threat-item{font-size:.82rem}
.threat-item span{color:var(--color-muted);display:block;font-size:.75rem}
.remediation-box{background:#f0fdf4;border:1px solid #86efac;border-radius:.35rem;padding:.7rem .9rem;font-size:.83rem;color:#14532d}
.corr-list{font-family:var(--font-mono);font-size:.8rem;color:var(--color-muted)}
.meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.5rem}
.meta-item{font-size:.83rem}
.meta-item .meta-key{color:var(--color-muted);font-size:.75rem;display:block}
.meta-item code{font-family:var(--font-mono);background:#f1f5f9;padding:.1rem .3rem;border-radius:.2rem;font-size:.78rem;word-break:break-all}
.note-list li{font-size:.82rem;margin-bottom:.25rem;color:var(--color-muted)}
.more-occurrences{font-size:.78rem;color:var(--color-muted);margin-top:.3rem}

/* ---- footer ---- */
.report-footer{text-align:center;font-size:.75rem;color:var(--color-muted);padding:1rem;border-top:1px solid var(--color-border);margin-top:1.5rem}

/* ---- print ---- */
@media print{
  @page{margin:2cm;size:A4}
  body{background:#fff;font-size:11pt}
  .report-wrap{max-width:100%;padding:0}
  .report-header{background:#1e3a5f !important;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .finding-card-header{cursor:default;background:#f8fafc !important}
  .finding-card-body{display:block !important}
  .card{break-inside:avoid-page}
  .finding-card{break-inside:avoid-page}
  a[href]::after{content:" (" attr(href) ")";font-size:.7em;color:#64748b}
}

/* ---- collapse (JS optional) ---- */
.finding-card-body.collapsed{display:none}
"""

_JS = """
document.querySelectorAll('.finding-card-header').forEach(function(hdr){
  hdr.addEventListener('click',function(){
    var body=hdr.nextElementSibling;
    if(body){body.classList.toggle('collapsed');}
  });
});
"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _render_header(r: ReportModel) -> str:
    meta_items = [
        ("Scan ID", r.scan_id),
        ("Timestamp", _fmt_timestamp(r.scan_timestamp)),
        ("Scanner Version", r.scanner_version),
        ("Format Detected", r.detected_format),
    ]
    meta_html = "".join(
        f'<div class="report-meta-item"><span>{_e(k)}</span><strong>{_e(v)}</strong></div>'
        for k, v in meta_items
    )
    return (
        f'<div class="report-header">'
        f'<div class="report-title">Static Security Scanner for Pre-Trained AI Models</div>'
        f'<div class="report-subtitle">Static Analysis Security Report</div>'
        f'<div class="report-meta-row">{meta_html}</div>'
        f'</div>'
    )


def _render_disclaimer(r: ReportModel) -> str:
    return (
        f'<div class="disclaimer">'
        f'<span class="disclaimer-icon">&#9888; IMPORTANT:</span>'
        f'{_e(r.static_analysis_disclaimer)}'
        f'</div>'
    )


_UNANALYZED_DESC: Final[str] = (
    "The format was identified but no analyzer inspected this artifact, so no "
    "rules were evaluated. This is NOT a clean result - the artifact has not "
    "been assessed. Treat it as unverified and inspect it by other means "
    "before loading."
)


def _verdict_label_and_desc(verdict: str, info_findings: int) -> tuple[str, str]:
    """Return the banner label and description for a verdict.

    PASS covers two different states: nothing was found at all, and only
    informational findings were found. Printing "No Findings Detected" over a
    report that also states "Total scanner findings: 3" made the report
    contradict itself, so the label distinguishes them.
    """
    label_map = {
        "FAIL": "FAIL — Security Issues Detected",
        "WARN": "WARN — Advisory Findings Present",
        "PASS": "PASS — No Findings Detected",
        "UNSUPPORTED": "UNSUPPORTED — Artifact Not Analysed",
        "UNANALYZED": "UNANALYZED — Artifact Not Inspected",
    }
    desc_map = {
        "FAIL": "One or more CRITICAL or HIGH severity findings were detected. Remediation is required before deployment.",
        "WARN": "Medium or low severity findings are present. Review and remediate as appropriate.",
        "PASS": "The scanner detected no security findings for this artifact.",
        "UNSUPPORTED": (
            "No analyzer is registered for this format, so no rules were evaluated. "
            "This is NOT a clean result - the artifact has not been assessed. Treat it "
            "as unverified and inspect it by other means before loading."
        ),
        "UNANALYZED": _UNANALYZED_DESC,
    }
    label = label_map.get(verdict, verdict)
    desc = desc_map.get(verdict, "")
    if verdict == "PASS" and info_findings > 0:
        label = "PASS — Informational Findings Only"
        plural = "s" if info_findings != 1 else ""
        desc = (
            "0 actionable security findings. "
            f"{info_findings} informational observation{plural} recorded for "
            "context (CRITICAL, HIGH, MEDIUM and LOW are all zero)."
        )
    return label, desc


def _render_verdict(r: ReportModel) -> str:
    label, desc = _verdict_label_and_desc(r.verdict, r.total_scanner_findings)
    # INCONCLUSIVE means the analyzer began but parsing failed — different from
    # UNANALYZED (no analyzer ran at all). Override the generic label/desc.
    if r.scan_outcome.upper() == "INCONCLUSIVE":
        label = "INCONCLUSIVE — Analysis Could Not Be Completed"
        desc = (
            "The analyzer began inspection but encountered parse or structural errors. "
            "Not all security rules could be evaluated. Parse errors are recorded as "
            "findings above. Treat this artifact as unverified and do not load it."
        )
    color = _VERDICT_COLOR.get(r.verdict, "#374151")
    return (
        f'<div class="verdict-banner verdict-{_e(r.verdict)}">'
        f'<span class="verdict-label" style="color:{color};">{_e(label)}</span>'
        f'<span class="verdict-desc">{_e(desc)}</span>'
        f'</div>'
    )


def _is_repository(r: ReportModel) -> bool:
    """True for the virtual REPOSITORY artifact (no bytes of its own)."""
    return r.detected_format == "REPOSITORY"


def _render_artifact_header(r: ReportModel, exit_label: str | None = None) -> str:
    rows = [
        ("Artifact Name", r.artifact_name),
        ("Source / Path", _display_source(r.artifact_source)),
        ("Input Source", r.input_source_type),
        ("Detected Format", r.detected_format),
        ("Scan Outcome", r.scan_outcome),
        (
            "File Size",
            "n/a — virtual repository artifact"
            if _is_repository(r)
            else f"{r.size_bytes:,} bytes",
        ),
        (
            "Scan Exit Code",
            exit_label
            if exit_label is not None
            else _exit_code_label(r.exit_code, r.scan_outcome, r.severity_counts),
        ),
        ("Duration", f"{r.artifact_duration_ms} ms" if r.artifact_duration_ms is not None else f"{r.total_duration_ms} ms"),
    ]
    items_html = "".join(
        f'<div class="meta-item"><span class="meta-key">{_e(k)}</span>'
        f'<code>{_e(v)}</code></div>'
        for k, v in rows
    )
    sha_label = (
        "Repository path digest (not a file hash)"
        if _is_repository(r)
        else "SHA-256"
    )
    sha_row = (
        f'<div class="meta-item" style="grid-column:1/-1">'
        f'<span class="meta-key">{_e(sha_label)}</span>'
        f'<code>{_e(r.sha256)}</code></div>'
    )
    return (
        f'<div class="card">'
        f'<div class="card-header"><h2>&#x1F4CB; Artifact Information</h2></div>'
        f'<div class="meta-grid">{items_html}{sha_row}</div>'
        f'</div>'
    )


def _render_risk_score(r: ReportModel) -> str:
    band_color = _RISK_BAND_COLOR.get(r.risk_band, "#374151")
    tiles = [
        (str(r.risk_score_final), "Risk Score (0-100)"),
        (r.risk_band.replace("_", " "), "Risk Band"),
        (f"{r.format_multiplier:.2f}x", "Format Multiplier"),
        (f"{r.obfuscation_multiplier:.2f}x", "Obfuscation Multiplier"),
    ]
    tiles_html = "".join(
        f'<div class="risk-tile"><span class="tile-val" style="color:{band_color if i<2 else "#374151"};">'
        f'{_e(v)}</span><span class="tile-lbl">{_e(lbl)}</span></div>'
        for i, (v, lbl) in enumerate(tiles)
    )
    return (
        '<div class="card">'
        '<div class="card-header"><h2>&#x26A0; Risk Assessment</h2></div>'
        f'<div class="risk-score-grid">{tiles_html}</div>'
        '<p style="font-size:.8rem;color:#64748b;">Risk scoring methodology: WeightedSum x FormatMultiplier x ObfuscationMultiplier, clamped [0, 100]. '
        'See project documentation for calibration parameters.</p>'
        '</div>'
    )


def _render_executive_summary(r: ReportModel) -> str:
    sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    sev_html = "".join(
        f'<div class="sev-card sev-{s}">'
        f'<span class="sev-count">{r.severity_counts.get(s, 0)}</span>'
        f'<span class="sev-name">{s}</span></div>'
        for s in sev_order
    )
    themes_str = (
        ", ".join(_e(t) for t in r.dominant_risk_themes)
        if r.dominant_risk_themes else "None detected"
    )
    corr_note = _e(r.correlation_note)
    collapsed_part = (
        f' &nbsp;|&nbsp; <strong>Collapsed (presentation):</strong> {r.collapsed_findings}'
        if r.collapsed_findings > 0 else ""
    )
    counts_html = (
        f'<p style="font-size:.84rem;margin-bottom:.5rem;">'
        f'<strong>Total scanner findings:</strong> {r.total_scanner_findings} &nbsp;|&nbsp; '
        f'<strong>Unique report findings:</strong> {r.unique_report_findings}'
        f'{collapsed_part}</p>'
        f'<p style="font-size:.84rem;color:#64748b;">The scanner finding count is authoritative. '
        f'Collapsed findings share the same rule and approximate location.</p>'
    )
    return (
        f'<div class="card">'
        f'<div class="card-header"><h2>&#x1F4CA; Executive Summary</h2></div>'
        f'<div class="severity-grid">{sev_html}</div>'
        f'{counts_html}'
        f'<p style="font-size:.84rem;margin-top:.75rem;"><strong>Dominant Risk Themes:</strong> {themes_str}</p>'
        f'<p style="font-size:.84rem;margin-top:.35rem;color:#374151;">{corr_note}</p>'
        f'</div>'
    )


def _subject_of(rf: ReportFinding) -> str:
    """Return what a finding is about, for the summary table.

    Two SC106 rows naming different weight files previously rendered as
    identical lines and read as accidental duplicates.
    """
    for sample in rf.evidence_samples:
        if sample.matched_value:
            return sample.matched_value[:60]
    return "—"


_FORMAT_CHECKS: dict[str, str] = {
    "SAFETENSORS": "ST1xx/ST2xx — header validation, tensor name screening, credential exposure",
    "PYTORCH_ZIP": "P1xx/P2xx — Pickle opcode disassembly, dangerous globals, REDUCE chains",
    "RAW_PICKLE": "P1xx/P2xx — Pickle opcode disassembly, dangerous globals, REDUCE chains",
    "GGUF": "GU1xx/GU2xx — GGUF metadata and KV-pair analysis, credential scanning",
    "ONNX": "ON1xx — ONNX graph structure, external data path traversal",
    "HDF5": "KR1xx — Keras/HDF5 config analysis, Lambda bytecode, external link detection",
    "KERAS": "KR1xx — Keras ZIP config analysis, Lambda bytecode, duplicate member detection",
    "TF_SAVEDMODEL": "TF1xx — TensorFlow SavedModel asset and operation validation",
    "REPOSITORY": "SC1xx/SC2xx — supply-chain and repository-level security checks",
}


def _checks_for_format(detected_format: str) -> str:
    return _FORMAT_CHECKS.get(
        detected_format,
        f"security rules applicable to {detected_format} format",
    )


def _render_findings_table(r: ReportModel) -> str:
    if not r.report_findings:
        checks = _checks_for_format(r.detected_format)
        return (
            '<div class="card">'
            '<div class="card-header"><h2>&#x2705; Findings Summary</h2></div>'
            '<p style="color:#15803d;font-weight:600;">No security findings detected.</p>'
            f'<p style="font-size:.83rem;color:#374151;margin-top:.6rem;">'
            f'<strong>Checks performed:</strong> {_e(checks)}.</p>'
            '<p style="font-size:.82rem;color:#64748b;margin-top:.3rem;">'
            'All applicable rules were evaluated — none triggered on this artifact.</p>'
            '</div>'
        )

    rows = "".join(
        f'<tr>'
        f'<td><code>{_e(rf.rule_id)}</code></td>'
        f'<td>{_e(rf.rule_name)}</td>'
        f'<td><code style="font-size:.75rem;">{_e(_subject_of(rf))}</code></td>'
        f'<td>{_sev_badge(rf.severity)}</td>'
        f'<td>{_e(rf.confidence)}</td>'
        f'<td style="font-size:.75rem;">{_e(rf.occurrence_count)}</td>'
        f'</tr>'
        for rf in r.report_findings
    )
    return (
        f'<div class="card">'
        f'<div class="card-header"><h2>&#x1F50D; Findings Summary ({_e(r.unique_report_findings)} unique)</h2></div>'
        f'<div class="table-wrap">'
        f'<table><thead><tr>'
        f'<th>Rule ID</th><th>Rule Name</th><th>Subject</th><th>Severity</th>'
        f'<th>Confidence</th><th>Occurrences</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        f'</div></div>'
    )


def _render_evidence_samples(rf: ReportFinding) -> str:
    if not rf.evidence_samples:
        return '<p style="font-size:.82rem;color:#64748b;">No evidence recorded.</p>'

    blocks: list[str] = []
    for i, ev in enumerate(rf.evidence_samples, 1):
        rows_html = ""
        if ev.archive_member:
            rows_html += f'<div class="evidence-row"><span class="evidence-key">member:</span> {_e(ev.archive_member)}</div>'
        if ev.byte_offset is not None:
            if ev.byte_offset == 0:
                # 0 is the default when no precise location was recorded;
                # displaying it as a real offset would be misleading.
                rows_html += (
                    '<div class="evidence-row"><span class="evidence-key">byte_offset:</span> '
                    '<span style="color:#64748b;font-style:italic;">not recorded</span></div>'
                )
            else:
                rows_html += f'<div class="evidence-row"><span class="evidence-key">byte_offset:</span> {_e(ev.byte_offset)}'
                if ev.byte_length is not None:
                    rows_html += f'  <span class="evidence-key">length:</span> {_e(ev.byte_length)}'
                rows_html += '</div>'
        if ev.opcode:
            rows_html += f'<div class="evidence-row"><span class="evidence-key">opcode:</span> {_e(ev.opcode)}</div>'
        rows_html += f'<div class="evidence-row"><span class="evidence-key">matched_value:</span> {_e(ev.matched_value)}'
        if ev.original_length is not None:
            rows_html += f'  <span class="evidence-key">(original length: {_e(ev.original_length)} chars, redacted)</span>'
        rows_html += '</div>'
        if ev.context_snippet:
            rows_html += f'<div class="evidence-row"><span class="evidence-key">context:</span> {_e(ev.context_snippet)}</div>'
        rows_html += f'<div class="evidence-row"><span class="evidence-key">coordinate_space:</span> {_e(ev.coordinate_space)}</div>'

        sample_label = f"Evidence Sample {i}" if len(rf.evidence_samples) > 1 else ""
        label_html = (
            f'<p style="font-size:.78rem;color:#94a3b8;margin-bottom:.25rem;">{_e(sample_label)}</p>'
            if sample_label else ""
        )
        blocks.append(
            label_html
            + f'<div class="evidence-block">{rows_html}</div>'
        )

    more = ""
    if rf.additional_occurrences > 0:
        more = (
            f'<p class="more-occurrences">…and {_e(rf.additional_occurrences)} '
            f'more occurrence(s) not shown.</p>'
        )

    return "".join(blocks) + more


def _render_finding_card(rf: ReportFinding) -> str:
    cwe_html = (
        ", ".join(f'<code>{_e(c)}</code>' for c in rf.cwe_ids)
        if rf.cwe_ids else '<em>Not mapped</em>'
    )
    owasp_html = _e(rf.owasp_category) if rf.owasp_category else "<em>Not directly mapped</em>"
    mitre_html = _e(rf.mitre_technique) if rf.mitre_technique else "<em>Not mapped</em>"
    refs_html = (
        ", ".join(f'<code>{_e(ref)}</code>' for ref in rf.references)
        if rf.references else '<em>None on record</em>'
    )
    corr_html = (
        '<br>'.join(f'<code>{_e(fid)}</code>' for fid in rf.correlated_finding_ids)
        if rf.correlated_finding_ids else '<em>None</em>'
    )
    occ_plural = "occurrence" if rf.occurrence_count == 1 else "occurrences"

    return (
        f'<div class="finding-card">'
        # Header (clickable toggle)
        f'<div class="finding-card-header">'
        f'<span class="finding-rule-id">{_e(rf.rule_id)}</span>'
        f'{_sev_badge(rf.severity)}'
        f'<span class="finding-rule-name">{_e(rf.rule_name)}</span>'
        f'<span style="font-size:.78rem;color:#64748b;">'
        f'{_e(rf.occurrence_count)} {_e(occ_plural)}</span>'
        f'</div>'
        # Body
        f'<div class="finding-card-body">'
        # 1. Description
        f'<div class="finding-section">'
        f'<h5>Description</h5>'
        f'<p>{_e(rf.description)}</p>'
        f'</div>'
        # 2. Severity + Confidence
        f'<div class="finding-section">'
        f'<h5>Severity &amp; Confidence</h5>'
        f'<p><strong>Severity:</strong> {_sev_badge(rf.severity)} &nbsp; '
        f'<strong>Confidence:</strong> {_e(rf.confidence)}</p>'
        f'<p style="font-size:.8rem;color:#64748b;margin-top:.25rem;">'
        f'<em>Confidence rationale:</em> {_e(rf.confidence_rationale)}</p>'
        f'</div>'
        # 3. Security Issue
        f'<div class="finding-section">'
        f'<h5>Security Issue</h5>'
        f'<p>{_e(rf.security_issue)}</p>'
        f'</div>'
        # 4. Static Evidence
        f'<div class="finding-section">'
        f'<h5>Static Evidence</h5>'
        f'{_render_evidence_samples(rf)}'
        f'</div>'
        # 5. Impact
        f'<div class="finding-section">'
        f'<h5>Impact</h5>'
        f'<p>{_e(rf.impact)}</p>'
        f'</div>'
        # 6. Remediation
        f'<div class="finding-section">'
        f'<h5>Remediation</h5>'
        f'<div class="remediation-box">{_e(rf.remediation)}</div>'
        f'</div>'
        # 7. Threat Mapping
        f'<div class="finding-section">'
        f'<h5>Threat Mapping</h5>'
        f'<div class="threat-grid">'
        f'<div class="threat-item"><span>CWE</span>{cwe_html}</div>'
        f'<div class="threat-item"><span>OWASP ML</span>{owasp_html}</div>'
        f'<div class="threat-item"><span>MITRE ATT&amp;CK</span>{mitre_html}</div>'
        f'</div></div>'
        # 8. References
        f'<div class="finding-section">'
        f'<h5>References</h5>'
        f'<p>{refs_html}</p>'
        f'</div>'
        # 9. Correlated Findings
        f'<div class="finding-section">'
        f'<h5>Correlated Finding IDs</h5>'
        f'<div class="corr-list">{corr_html}</div>'
        f'</div>'
        f'</div>'  # .finding-card-body
        f'</div>'  # .finding-card
    )


def _render_findings_detail(r: ReportModel) -> str:
    if not r.report_findings:
        return ""
    cards = "".join(_render_finding_card(rf) for rf in r.report_findings)
    return (
        f'<div class="card">'
        f'<div class="card-header"><h2>&#x1F512; Detailed Security Findings</h2>'
        f'<p style="font-size:.8rem;color:#64748b;margin-top:.25rem;">'
        f'Click a finding header to expand/collapse details.</p>'
        f'</div>'
        f'{cards}'
        f'</div>'
    )


def _render_correlation(r: ReportModel) -> str:
    note_html = _e(r.correlation_note)
    corr_findings = [rf for rf in r.report_findings if rf.correlated_finding_ids]
    if not corr_findings:
        body = f'<p style="font-size:.84rem;">{note_html}</p>'
    else:
        rows = "".join(
            f'<tr><td><code>{_e(rf.rule_id)}</code></td>'
            f'<td>{_e(rf.rule_name)}</td>'
            f'<td><div class="corr-list">'
            + "  ".join(f'<code>{_e(fid)}</code>' for fid in rf.correlated_finding_ids)
            + '</div></td></tr>'
            for rf in corr_findings
        )
        body = (
            f'<p style="font-size:.84rem;margin-bottom:.75rem;">{note_html}</p>'
            f'<div class="table-wrap"><table><thead><tr>'
            f'<th>Rule ID</th><th>Rule Name</th><th>Correlated Finding IDs</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )
    return (
        f'<div class="card">'
        f'<div class="card-header"><h2>&#x1F517; Correlation Analysis</h2></div>'
        f'{body}'
        f'</div>'
    )


def _render_model_structure(r: ReportModel) -> str:
    """Render the static tensor inventory table when available."""
    ms = r.model_structure
    if not ms:
        return ""
    tensor_count = ms.get("tensor_count", 0)
    tensors = ms.get("tensors") or []
    format_notes = ms.get("format_notes")

    rows = []
    for t in tensors[:100]:  # display at most 100 rows in HTML
        name = _e(str(t.get("name", "")))
        dtype = _e(str(t.get("dtype", "")))
        shape_raw = t.get("shape")
        shape = _e(str(shape_raw) if shape_raw is not None else "—")
        rows.append(
            f'<tr><td style="font-family:monospace;font-size:.82rem;">{name}</td>'
            f'<td style="text-align:center;">{dtype}</td>'
            f'<td style="text-align:center;font-family:monospace;">{shape}</td></tr>'
        )

    truncation = ""
    if len(tensors) > 100:
        truncation = f'<p style="font-size:.8rem;color:#64748b;">Showing first 100 of {len(tensors)} tensors.</p>'

    notes_html = (
        f'<p style="font-size:.8rem;color:#64748b;margin-top:.5rem;">{_e(format_notes)}</p>'
        if format_notes else ""
    )

    table_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:.84rem;">'
        '<thead><tr>'
        '<th style="text-align:left;padding:.25rem .5rem;border-bottom:1px solid #e2e8f0;">Tensor Name</th>'
        '<th style="text-align:center;padding:.25rem .5rem;border-bottom:1px solid #e2e8f0;">DType</th>'
        '<th style="text-align:center;padding:.25rem .5rem;border-bottom:1px solid #e2e8f0;">Shape</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    ) if rows else "<p style='color:#64748b;font-size:.84rem;'>No tensor descriptors extracted.</p>"

    return (
        '<div class="card">'
        '<div class="card-header"><h2>&#x1F4CA; Static Model Structure</h2></div>'
        f'<p style="font-size:.84rem;margin-bottom:.75rem;">'
        f'Tensor count (from parsed descriptors): <strong>{_e(str(tensor_count))}</strong>. '
        f'This inventory is informational — it reflects the static structure declared in the file, '
        f'not actual runtime weight tensors.</p>'
        + table_html
        + truncation
        + notes_html
        + "</div>"
    )


def _render_methodology(
    r: ReportModel,
    formats_analyzed: list[str] | None = None,
) -> str:
    # ReportModel.formats_analyzed holds one artifact's format. A multi-artifact
    # scan must pass the scan-level list or the section claims the whole scan
    # analysed only the first artifact's format.
    formats = formats_analyzed if formats_analyzed is not None else r.formats_analyzed
    formats_html = _e(", ".join(formats)) if formats else "Not available"
    notes_html = "".join(f'<li>{_e(n)}</li>' for n in r.methodology_notes)
    unparsed_html = (
        "".join(f'<li><code>{_e(u)}</code></li>' for u in r.unparsed_members)
        if r.unparsed_members else "<li>None recorded</li>"
    )
    return (
        f'<div class="card">'
        f'<div class="card-header"><h2>&#x1F4DD; Methodology &amp; Limitations</h2></div>'
        f'<h4>Analyzed Formats</h4><p style="font-size:.84rem;margin-bottom:.75rem;">{formats_html}</p>'
        f'<h4>What Was Inspected</h4>'
        f'<ul class="note-list">{notes_html}</ul>'
        f'<h4 style="margin-top:.75rem;">Unparsed / Skipped Members</h4>'
        f'<ul class="note-list">{unparsed_html}</ul>'
        f'<h4 style="margin-top:.75rem;">Limitations</h4>'
        f'<ul class="note-list">'
        f'<li>Static analysis cannot guarantee absence of vulnerabilities.</li>'
        f'<li>Formats without a registered analyzer are not inspected '
        f'(a G002 finding is emitted as a warning).</li>'
        f'<li>Obfuscated or encrypted payloads may evade static detection.</li>'
        f'<li>Formats: TFLite, NumPy, Flax — identified but not analyzed. '
        f'ONNX, HDF5/Keras, TF SavedModel are analyzed with optional dependencies.</li>'
        f'</ul>'
        f'</div>'
    )


_EPHEMERAL_MARKERS: tuple[str, ...] = (
    "model_scanner_uploads",
    "/tmp/",  # noqa: S108
    "\\Temp\\",
    "/Temp/",
)


def _display_source(target: str) -> str:
    """Hide scratch upload paths, which leak the OS username and never persist."""
    if _is_ephemeral_path(target):
        name = target.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        return f"{name} (uploaded via web UI, temporary copy removed)"
    return target


def _fmt_timestamp(raw: str) -> str:
    """Render an ISO 8601 timestamp as dd/mm/yyyy HH:MM for human readers.

    The canonical JSON report keeps ISO 8601 — that is the machine-readable
    contract. This is presentation only.

    Parameters
    ----------
    raw:
        ISO 8601 timestamp, e.g. "2026-08-22T07:21:25.854134+00:00".

    Returns
    -------
    str
        "22/08/2026 07:21", or the input unchanged if it cannot be parsed.
    """
    if not raw:
        return ""
    try:
        text = str(raw).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(text).strftime("%d/%m/%Y %H:%M") + " UTC"
    except (ValueError, TypeError):
        return str(raw)


def _is_ephemeral_path(target: str) -> bool:
    """True when a target path is a scratch upload directory that no longer exists."""
    if not target:
        return False
    return any(marker in target for marker in _EPHEMERAL_MARKERS)


def _is_info_only(severity_counts: dict[str, int] | None) -> bool:
    """True when findings exist and every one of them is INFO.

    Used only for wording. The numeric exit code is ClamAV-compatible and
    severity-agnostic by contract (core/engine.py: any findings -> 1), and is
    NOT changed here.
    """
    if not severity_counts:
        return False
    actionable = sum(
        severity_counts.get(sev, 0)
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    )
    return actionable == 0 and severity_counts.get("INFO", 0) > 0


def _exit_code_label(
    code: int,
    outcome: str = "",
    severity_counts: dict[str, int] | None = None,
) -> str:
    """Render the process exit code with its meaning.

    The field was previously labelled "ClamAV" though no ClamAV is involved;
    these are this scanner's own exit codes.

    Exit code 1 means "findings present" and is severity-agnostic by design.
    Describing it as "security findings present" on an INFO-only scan
    contradicted the same report's own PASS / CLEAN / score 0 verdict, so the
    wording distinguishes the two cases. The numeric code is unchanged.
    """
    if code == 0 and str(outcome).upper() == "UNSUPPORTED":
        # "clean" would be a false assurance: no rules ran at all.
        return "0 - no findings (nothing was analysed)"
    if str(outcome).upper() == "INCONCLUSIVE":
        return "2 - analysis incomplete (parse errors prevented full inspection)"
    if code == 1 and _is_info_only(severity_counts):
        return "1 - informational findings present (no actionable findings)"
    meanings = {
        0: "0 - clean, no findings",
        1: "1 - security findings present",
        2: "2 - operational error during scan",
    }
    return meanings.get(code, str(code))


def _render_reproducibility(r: ReportModel) -> str:
    rows = [
        ("Scan ID", r.scan_id),
        ("Scanner Version", r.scanner_version),
        ("Scan Timestamp", _fmt_timestamp(r.scan_timestamp)),
        ("Target / Source", _display_source(r.target_request)),
        ("Artifact SHA-256", r.sha256),
        (
            "Scan Exit Code",
            _exit_code_label(r.exit_code, r.scan_outcome, r.severity_counts),
        ),
        ("Strict Mode", str(r.strict_mode)),
        ("Total Duration", f"{r.total_duration_ms} ms"),
    ]
    items_html = "".join(
        f'<div class="meta-item"><span class="meta-key">{_e(k)}</span>'
        f'<code>{_e(v)}</code></div>'
        for k, v in rows
    )
    # An uploaded artifact lives in a temp directory that is deleted the moment
    # the scan finishes, so printing that path gave a command that could never
    # run (and leaked the OS username). Name the artifact instead.
    strict_flag = " --strict" if r.strict_mode else ""
    if _is_ephemeral_path(r.target_request):
        repro_html = (
            f'This artifact was uploaded through the web UI and analysed in a '
            f'temporary directory that has since been removed. To reproduce, run '
            f'the scanner against your own copy of '
            f'<code>{_e(r.artifact_name)}</code> '
            f'(SHA-256 <code>{_e(r.sha256)}</code>):<br>'
            f'<code>python -m model_scanner scan '
            f'&lt;path-to&gt;/{_e(r.artifact_name)}{strict_flag}</code>'
        )
    else:
        repro_html = (
            f'To reproduce: <code>python -m model_scanner scan '
            f'{_e(r.target_request)}{strict_flag}</code>'
        )

    return (
        f'<div class="card">'
        f'<div class="card-header"><h2>&#x1F501; Reproducibility</h2></div>'
        f'<div class="meta-grid">{items_html}</div>'
        f'<p style="font-size:.8rem;color:#64748b;margin-top:.75rem;">'
        f'{repro_html}</p>'
        f'</div>'
    )


def _render_footer(r: ReportModel) -> str:
    return (
        f'<div class="report-footer">'
        f'Static Security Scanner for Pre-Trained AI Models &bull; '
        f'Scanner {_e(r.scanner_version)} &bull; '
        f'{_e(_fmt_timestamp(r.scan_timestamp))}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scan-level rendering (multi-artifact)
# ---------------------------------------------------------------------------


def _render_scan_header(sr: ScanReportModel) -> str:
    meta = [
        ("Scan ID", sr.scan_id),
        ("Timestamp", _fmt_timestamp(sr.scan_timestamp)),
        ("Scanner Version", sr.scanner_version),
        ("Artifacts", str(sr.total_artifacts)),
    ]
    items = "".join(
        f'<div class="report-meta-item"><span>{_e(k)}</span>'
        f'<strong>{_e(v)}</strong></div>'
        for k, v in meta
    )
    return (
        '<div class="report-header">'
        '<div class="report-title">AI Model Security Assessment Report</div>'
        '<div class="report-subtitle">Static Analysis — Pre-Trained AI Model '
        'Security Scanner</div>'
        f'<div class="report-meta-row">{items}</div>'
        '</div>'
    )


def _render_scan_verdict(sr: ScanReportModel) -> str:
    info_only = sr.severity_counts.get("INFO", 0) == sr.total_scanner_findings
    label, desc = _verdict_label_and_desc(
        sr.overall_verdict,
        sr.total_scanner_findings if info_only else 0,
    )
    color = _VERDICT_COLOR.get(sr.overall_verdict, "#374151")
    banner = (
        f'<div class="verdict-banner verdict-{_e(sr.overall_verdict)}">'
        f'<span class="verdict-label" style="color:{color};">{_e(label)}</span>'
        f'<span class="verdict-desc">{_e(desc)}</span>'
        f'</div>'
    )
    if sr.has_unanalyzed:
        inconclusive_count = sum(
            1 for r in sr.artifact_reports
            if r.scan_outcome.upper() == "INCONCLUSIVE"
        )
        uninspected_count = sr.unanalyzed_artifacts - inconclusive_count
        if inconclusive_count > 0 and uninspected_count <= 0:
            gap_msg = (
                f'{sr.unanalyzed_artifacts} of {sr.total_artifacts} artifact(s) could not '
                'be fully analyzed — the analyzer began inspection but encountered '
                'structural or parse errors. '
                'Their results carry no assurance and must not be read as clean. '
                'See the artifact inventory below.'
            )
        elif inconclusive_count == 0:
            gap_msg = (
                f'{sr.unanalyzed_artifacts} of {sr.total_artifacts} artifact(s) in '
                'this scan were NOT inspected — no analyzer ran for their format. '
                'Their results carry no assurance and must not be read as clean. '
                'See the artifact inventory below.'
            )
        else:
            gap_msg = (
                f'{sr.unanalyzed_artifacts} of {sr.total_artifacts} artifact(s) could not '
                f'be assessed: {inconclusive_count} encountered parse or structural errors '
                f'during inspection and {uninspected_count} had no registered analyzer '
                'for their format. '
                'Their results carry no assurance and must not be read as clean. '
                'See the artifact inventory below.'
            )
        banner += (
            '<div class="disclaimer">'
            '<span class="disclaimer-icon">&#9888; COVERAGE GAP:</span>'
            + gap_msg
            + '</div>'
        )
    return banner


def _render_artifact_inventory(sr: ScanReportModel) -> str:
    rows = "".join(
        f'<tr>'
        f'<td><code>{_e(r.artifact_name)}</code></td>'
        f'<td>{_e(r.detected_format)}</td>'
        f'<td>{_e(r.scan_outcome)}</td>'
        f'<td>{_e(r.verdict)}</td>'
        f'<td>{_e(r.risk_band)}</td>'
        f'<td style="font-size:.75rem;">{_e(r.risk_score_final)}</td>'
        f'<td style="font-size:.75rem;">{_e(r.total_scanner_findings)}</td>'
        f'</tr>'
        for r in sr.artifact_reports
    )
    return (
        '<div class="card">'
        '<div class="card-header"><h2>&#x1F4E6; Artifact Inventory'
        f' ({_e(sr.total_artifacts)})</h2></div>'
        '<div class="table-wrap"><table><thead><tr>'
        '<th>Artifact</th><th>Format</th><th>Outcome</th><th>Verdict</th>'
        '<th>Risk Band</th><th>Score</th><th>Findings</th>'
        '</tr></thead><tbody>'
        f'{rows}</tbody></table></div>'
        '<p style="font-size:.8rem;color:#64748b;margin-top:.5rem;">'
        'Every artifact in the scan is listed. Detailed sections follow below.'
        '</p></div>'
    )


def _render_scan_summary(sr: ScanReportModel) -> str:
    # Split the total so an INFO-only scan cannot read as "4 findings" next to
    # a CLEAN verdict. Severity itself is unchanged; this is presentation only.
    actionable = sum(
        sr.severity_counts.get(sev, 0)
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    )
    informational = sr.severity_counts.get("INFO", 0)
    cards = "".join(
        f'<div class="sev-card sev-{sev}">'
        f'<span class="sev-count">{_e(sr.severity_counts.get(sev, 0))}</span>'
        f'<span class="sev-name">{sev}</span></div>'
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
    )
    return (
        '<div class="card">'
        '<div class="card-header"><h2>&#x1F4CA; Scan Summary</h2></div>'
        f'<div class="severity-grid">{cards}</div>'
        '<p style="font-size:.84rem;">'
        f'<strong>Artifacts:</strong> {_e(sr.total_artifacts)} &nbsp;|&nbsp; '
        f'<strong>Analysed:</strong> {_e(sr.scanned_successfully)} &nbsp;|&nbsp; '
        f'<strong>Unanalyzed:</strong> {_e(sr.unanalyzed_artifacts)} &nbsp;|&nbsp; '
        f'<strong>Unsupported:</strong> {_e(sr.unsupported_artifacts)} &nbsp;|&nbsp; '
        f'<strong>Failed:</strong> {_e(sr.failed_artifacts)}</p>'
        '<p style="font-size:.84rem;margin-top:.35rem;">'
        f'<strong>Worst risk band:</strong> {_e(sr.worst_risk_band)} &nbsp;|&nbsp; '
        f'<strong>Max risk score:</strong> {_e(sr.max_risk_score)}/100 &nbsp;|&nbsp; '
        f'<strong>Actionable findings:</strong> {_e(actionable)} &nbsp;|&nbsp; '
        f'<strong>Informational observations:</strong> {_e(informational)}</p>'
        '</div>'
    )


def _render_artifact_section(
    r: ReportModel,
    index: int,
    total: int,
    exit_label: str | None = None,
) -> str:
    # In multi-artifact scans the scan-level exit code (exit_label) may say
    # "1 - security findings present" even for a PASS artifact whose own
    # findings count is zero. Show a per-artifact code for clean artifacts so
    # the exit code does not contradict the PASS verdict on the same page.
    if exit_label is not None and r.verdict == "PASS" and r.total_scanner_findings == 0:
        per_artifact_label = _exit_code_label(0, r.scan_outcome, r.severity_counts)
    else:
        per_artifact_label = exit_label
    return (
        '<div class="card">'
        f'<div class="card-header"><h2>&#x1F9E9; Artifact {index} of {total} — '
        f'{_e(r.artifact_name)}</h2></div>'
        f'{_render_verdict(r)}'
        f'{_render_artifact_header(r, per_artifact_label)}'
        f'{_render_risk_score(r)}'
        f'{_render_executive_summary(r)}'
        f'{_render_findings_table(r)}'
        f'{_render_findings_detail(r)}'
        f'{_render_correlation(r)}'
        '</div>'
    )


def _render_scan_reproducibility(sr: ScanReportModel) -> str:
    """Scan-level reproducibility block.

    Per-artifact SHA-256 digests are listed against the artifacts they belong
    to. The REPOSITORY artifact is excluded: its digest is derived from the
    repository's temporary directory path, not from file content, so quoting it
    as an "Artifact SHA-256" to reproduce against was not actionable.
    """
    rows = [
        ("Scan ID", sr.scan_id),
        ("Scanner Version", sr.scanner_version),
        ("Scan Timestamp", _fmt_timestamp(sr.scan_timestamp)),
        ("Target / Source", _display_source(sr.target_request)),
        ("Artifacts Scanned", str(sr.total_artifacts)),
        ("Scan Exit Code", _exit_code_label(sr.exit_code, "", sr.severity_counts)),
        ("Strict Mode", str(sr.strict_mode)),
        ("Total Duration", f"{sr.total_duration_ms} ms"),
    ]
    items_html = "".join(
        f'<div class="meta-item"><span class="meta-key">{_e(k)}</span>'
        f'<code>{_e(v)}</code></div>'
        for k, v in rows
    )

    digest_rows = "".join(
        f'<tr><td><code>{_e(r.artifact_name)}</code></td>'
        f'<td><code style="font-size:.72rem;">{_e(r.sha256)}</code></td></tr>'
        for r in sr.artifact_reports
        if r.detected_format != "REPOSITORY"
    )
    if digest_rows:
        digests_html = (
            '<h4 style="margin-top:.75rem;">Artifact Digests</h4>'
            '<div class="table-wrap"><table><thead><tr>'
            '<th>Artifact</th><th>SHA-256</th></tr></thead>'
            f'<tbody>{digest_rows}</tbody></table></div>'
        )
    else:
        digests_html = (
            '<p style="font-size:.8rem;color:#64748b;margin-top:.5rem;">'
            'No file-backed artifacts in this scan.</p>'
        )

    strict_flag = " --strict" if sr.strict_mode else ""
    if _is_ephemeral_path(sr.target_request):
        repro_html = (
            'These artifacts were uploaded through the web UI and analysed in a '
            'temporary directory that has since been removed. To reproduce, run '
            'the scanner against your own copy of the model folder and match the '
            'per-artifact digests above:<br>'
            f'<code>python -m model_scanner scan '
            f'&lt;path-to-model-folder&gt;{strict_flag}</code>'
        )
    else:
        repro_html = (
            f'To reproduce: <code>python -m model_scanner scan '
            f'{_e(sr.target_request)}{strict_flag}</code>'
        )

    return (
        '<div class="card">'
        '<div class="card-header"><h2>&#x1F501; Reproducibility</h2></div>'
        f'<div class="meta-grid">{items_html}</div>'
        f'{digests_html}'
        '<p style="font-size:.8rem;color:#64748b;margin-top:.75rem;">'
        f'{repro_html}</p>'
        '</div>'
    )


class HTMLReportRenderer:
    """Generate a professional self-contained HTML security report."""

    @staticmethod
    def render(report: ReportModel) -> str:
        """Return the complete HTML report as a UTF-8 string.

        All values are HTML-escaped.  The output contains no external
        resource references and is fully readable without JavaScript.
        """
        body = "".join([
            _render_header(report),
            _render_disclaimer(report),
            _render_verdict(report),
            _render_artifact_header(report),
            _render_risk_score(report),
            _render_executive_summary(report),
            _render_findings_table(report),
            _render_findings_detail(report),
            _render_correlation(report),
            _render_model_structure(report),
            _render_methodology(report),
            _render_reproducibility(report),
            _render_footer(report),
        ])

        return (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>Security Report — {_e(report.artifact_name)}</title>\n'
            f'<style>{_CSS}</style>\n'
            '</head>\n'
            '<body>\n'
            f'<div class="report-wrap">{body}</div>\n'
            f'<script>{_JS}</script>\n'
            '</body>\n'
            '</html>\n'
        )

    @staticmethod
    def render_scan(scan_report: ScanReportModel) -> str:
        """Return a complete HTML report covering EVERY artifact in the scan.

        ReportModel is per-artifact, so rendering a multi-artifact scan through
        `render` silently dropped every artifact after the first.
        """
        primary = scan_report.artifact_reports[0]
        # exit_code on ReportModel is the whole scan's code, so every artifact
        # section must describe the scan's severity mix, not its own.
        scan_exit_label = _exit_code_label(
            scan_report.exit_code, "", scan_report.severity_counts
        )
        sections = "".join(
            _render_artifact_section(
                r, i + 1, scan_report.total_artifacts, scan_exit_label
            )
            for i, r in enumerate(scan_report.artifact_reports)
        )
        body = "".join([
            _render_scan_header(scan_report),
            _render_disclaimer(primary),
            _render_scan_verdict(scan_report),
            _render_artifact_inventory(scan_report),
            _render_scan_summary(scan_report),
            sections,
            _render_methodology(primary, scan_report.formats_analyzed),
            _render_scan_reproducibility(scan_report),
            _render_footer(primary),
        ])
        title = _e(_display_source(scan_report.target_request))
        return (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>Security Report — {title}</title>\n'
            f'<style>{_CSS}</style>\n'
            '</head>\n'
            '<body>\n'
            f'<div class="report-wrap">{body}</div>\n'
            f'<script>{_JS}</script>\n'
            '</body>\n'
            '</html>\n'
        )

    @staticmethod
    def export(
        report: ReportModel,
        destination: str | Path,
    ) -> str:
        """Write the HTML report to a file and return the HTML string."""
        html_content = HTMLReportRenderer.render(report)
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(html_content, encoding="utf-8")
        return html_content


__all__ = [
    "HTMLReportRenderer",
]
