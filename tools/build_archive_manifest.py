#!/usr/bin/env python3
"""Create a byte-level inventory for a local provenance archive."""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix='.manifest-')
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mirror-output', type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    excluded = {args.output.resolve()}
    if args.mirror_output:
        excluded.add(args.mirror_output.resolve())

    inode_cache = {}
    records = []
    for path in sorted(item for item in archive.rglob('*') if item.is_file()):
        if path.resolve() in excluded:
            continue
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        checksum = inode_cache.get(inode)
        if checksum is None:
            checksum = digest(path)
            inode_cache[inode] = checksum
        records.append({
            'path': str(path.relative_to(archive)),
            'bytes': stat.st_size,
            'sha256': checksum,
            'hardlink_count': stat.st_nlink,
        })

    payload = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'archive_root': str(archive),
        'source_git_commit': '282c79a8a9d5fa6b8cf1764f72a0be8482798f1d',
        'file_count': len(records),
        'logical_bytes': sum(record['bytes'] for record in records),
        'unique_inode_count': len(inode_cache),
        'files': records,
    }
    atomic_json(args.output.resolve(), payload)
    if args.mirror_output:
        atomic_json(args.mirror_output.resolve(), payload)
    print(json.dumps({key: payload[key] for key in (
        'archive_root', 'file_count', 'logical_bytes', 'unique_inode_count')},
        indent=2))


if __name__ == '__main__':
    main()
