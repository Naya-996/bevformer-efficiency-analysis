# Provenance archive

The pre-repair snapshot is stored locally at:

`/home/abc/smq/BEVFormer_archive_20260911`

It contains:

- a verified Git bundle containing every local ref at commit
  `282c79a8a9d5fa6b8cf1764f72a0be8482798f1d`;
- a working-tree source snapshot, including the pending Full-24 additions;
- the original experiment manifests before hash repair;
- all existing evaluation products under `test/`;
- all static checkpoints under `ckpts/`;
- the completed 2-epoch and 6-epoch training directories;
- a timestamped copy of logs, including a point-in-time copy of the active
  Full-24 log.

Large static artifacts are archived through same-filesystem hard links. This
preserves the bytes if the original path is later unlinked without duplicating
physical storage. The active Full-24 run must be archived again after it exits.

Six retrospective evaluation manifests contained config SHA256 values that did
not match their current source paths and did not correspond to any available
Git revision. The original declarations remain in the pre-repair snapshot. In
the live manifests, `sha256` now identifies the currently archived source file,
while `legacy_declared_sha256` and `provenance_note` preserve the discrepancy.
Checkpoint hashes were independently recomputed and matched.
