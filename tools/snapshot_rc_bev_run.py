#!/usr/bin/env python3
"""Archive code/config bytes and their hashes for an experiment run."""
import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

from run_rc_bev_followup import ROOT, now, save


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / 'source_snapshot.zip'
    # Never silently replace the source evidence of an earlier attempt.
    if destination.exists():
        raise FileExistsError(destination)
    records = []
    files = sorted(set(path for subdir in ('projects', 'tools', 'tests')
                       for path in (ROOT / subdir).rglob('*.py')))
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            raw = path.read_bytes()
            relative = str(path.relative_to(ROOT))
            archive.writestr(relative, raw)
            records.append(dict(path=relative, sha256=hashlib.sha256(raw).hexdigest()))
        archive.writestr('tracked_changes.patch', subprocess.check_output(
            ['git', 'diff', 'HEAD', '--', 'projects', 'tools', 'tests'], cwd=ROOT))
    save(directory / 'source_manifest.json', dict(
        created_at=now(), git_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_archive=str(destination), files=records,
        note='Snapshot includes local source edits; git commit alone is not the full code version.'))


if __name__ == '__main__':
    main()
