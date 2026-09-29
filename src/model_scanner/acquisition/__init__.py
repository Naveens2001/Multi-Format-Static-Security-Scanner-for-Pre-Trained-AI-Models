"""
model_scanner.acquisition
=========================

Input ingestion, bounded download, and SHA-256 fingerprinting layer.

This package implements the InputAcquisitionProvider abstract interface
and all concrete providers:

  - LocalFileAcquisition      — single local file
  - LocalDirectoryAcquisition — recursive directory walk
  - PublicUrlAcquisition      — bounded HTTP/HTTPS download with SSRF guards
  - HuggingFaceAcquisition    — HF direct file and repository adapter

Design Reference
----------------
Sections 9-12 of docs/implementation_design.md
"""
