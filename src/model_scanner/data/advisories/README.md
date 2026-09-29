# Advisory Database Snapshot

This directory stores a **pinned, dated snapshot** of the
[PyPA Advisory Database](https://github.com/pypa/advisory-database) in OSV
(Open Source Vulnerability) schema format.

## Snapshot Policy

- The snapshot is a **build-time artifact**, never fetched at scan time.
- The snapshot date is recorded in `snapshot_meta.json`.
- Every dependency finding states the snapshot date and warns that it may be stale.
- **If this directory contains no advisory files** (i.e. `snapshot_meta.json` is absent
  or the `advisories/` subdirectory is empty), the scanner reports dependency scanning as
  **not performed** rather than as clean.  This is the fail-closed policy (NFR-01/B3).

## Adding Advisories

1. Download OSV JSON files from `https://osv.dev/list` or the PyPA advisory DB.
2. Place each file under `advisories/<ecosystem>/<advisory-id>.json`.
3. Update `snapshot_meta.json` with the current date.

## Schema

Each file follows the OSV 1.3 schema:
```json
{
  "id": "PYSEC-YYYY-NNNN",
  "affected": [
    {
      "package": { "ecosystem": "PyPI", "name": "package-name" },
      "ranges": [
        {
          "type": "ECOSYSTEM",
          "events": [
            { "introduced": "0" },
            { "fixed": "1.2.3" }
          ]
        }
      ]
    }
  ]
}
```
