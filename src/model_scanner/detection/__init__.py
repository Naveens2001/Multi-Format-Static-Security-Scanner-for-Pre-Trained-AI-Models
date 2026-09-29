"""
model_scanner.detection
=======================

Two-layer format identification and routing for acquired model artifacts.

Implements:
  - magic.py   — 16-byte magic sniffing and header prefix detection
  - detector.py — FormatDispatcher with extension fallback and mismatch logic

Design Reference
----------------
Section 13 of docs/implementation_design.md
"""
