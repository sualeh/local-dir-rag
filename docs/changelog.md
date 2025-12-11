# Changelog

## 1.0.0 - 2025-12-11

[v0.4.1...main](https://github.com/sualeh/local-dir-rag/compare/v0.4.1...main)

- Added multi-directory ingestion: recurse subdirectories and allow multiple doc roots to embed or update.
- Added incremental tracking: store file checksums, sizes, separate path/name, and a primary key to detect updates safely.
- Enabled updating existing vector data without a full rebuild when source files change.
- Updated dependency pins and GitHub Actions; refreshed project docs (AGENTS instructions).
