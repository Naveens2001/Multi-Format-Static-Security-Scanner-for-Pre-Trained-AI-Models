"""
model_scanner.presentation
==========================

Presentation and export layer for static model security scan results.

Components
----------
- JSONReportExporter / JsonReportExporter: Canonical JSON forensic report generator.
- ReportBuilder / ReportModel: Derived read-only presentation model.
- ProfessionalJSONExporter: Professional-grade JSON assessment report.
- HTMLReportRenderer: Self-contained HTML security report.
- PDFReportRenderer: Optional PDF renderer (requires model-scanner[report]).
"""

from __future__ import annotations

from model_scanner.presentation.html_report import HTMLReportRenderer
from model_scanner.presentation.json_export import (
    JSONReportExporter,
    JsonReportExporter,
)
from model_scanner.presentation.pdf_report import PDFReportRenderer
from model_scanner.presentation.professional_report import ProfessionalJSONExporter
from model_scanner.presentation.report_model import (
    REPORT_SCHEMA_VERSION,
    STATIC_ANALYSIS_DISCLAIMER,
    RedactedEvidence,
    ReportBuilder,
    ReportFinding,
    ReportModel,
)

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "STATIC_ANALYSIS_DISCLAIMER",
    "HTMLReportRenderer",
    "JSONReportExporter",
    "JsonReportExporter",
    "PDFReportRenderer",
    "ProfessionalJSONExporter",
    "RedactedEvidence",
    "ReportBuilder",
    "ReportFinding",
    "ReportModel",
]
