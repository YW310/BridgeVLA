"""Loaded checkpoint guards avoid repeated reads and reject changed files."""
import hashlib
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune'))
from bridgevla.utils import checkpoint_identity as module


def test_loaded_identity_hashes_once_and_checks_metadata_without_opening(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'loaded checkpoint identity')
    identity = module.LoadedCheckpointIdentity(checkpoint)
    expected = hashlib.sha256(b'loaded checkpoint identity').hexdigest()
    assert identity.sha256 == expected

    def forbidden(*args, **kwargs):
        raise AssertionError('a per-step loaded identity check must not reread the checkpoint')

    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(module, 'sha256_file', forbidden)
    for _ in range(100):
        assert module.loaded_checkpoint_sha256(checkpoint, identity) == expected


@pytest.mark.parametrize('change', ['rewrite', 'replace', 'touch', 'delete', 'different_path'])
def test_changed_or_missing_checkpoint_is_rejected(tmp_path, change):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'original')
    identity = module.LoadedCheckpointIdentity(checkpoint)
    before = checkpoint.stat()
    if change == 'rewrite':
        checkpoint.write_bytes(b'modified')  # same size as the original
        os.utime(checkpoint, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    elif change == 'replace':
        replacement = tmp_path / 'replacement.pth'
        replacement.write_bytes(b'original')
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, checkpoint)  # same bytes/size/mtime; different inode
    elif change == 'touch':
        os.utime(checkpoint, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    elif change == 'delete':
        checkpoint.unlink()
    else:
        checkpoint = tmp_path / 'other.pth'
        checkpoint.write_bytes(b'original')
    with pytest.raises(ValueError, match='checkpoint.*(changed|accessible)'):
        identity.verify_file(checkpoint)


def test_change_during_initial_hash_is_rejected(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'original')
    original = module.sha256_file

    def changing(path):
        digest = original(path)
        checkpoint.write_bytes(b'updated during initial hashing')
        return digest

    monkeypatch.setattr(module, 'sha256_file', changing)
    with pytest.raises(ValueError, match='changed while computing'):
        module.LoadedCheckpointIdentity(checkpoint)


def test_no_loaded_identity_keeps_full_content_hashing(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'original')
    original = module.sha256_file
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(module, 'sha256_file', counted)
    first = module.loaded_checkpoint_sha256(checkpoint)
    checkpoint.write_bytes(b'updated')
    assert module.loaded_checkpoint_sha256(checkpoint) != first
    assert len(calls) == 2
    with pytest.raises(ValueError, match='model loader'):
        module.loaded_checkpoint_sha256(checkpoint, {'sha256': first})


def test_task_digest_binding_is_local_and_rejects_unknown_paths(tmp_path):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'loaded model')
    identity = module.LoadedCheckpointIdentity(checkpoint)
    namespace = {'_file_digest': lambda path: 'original strict dependency'}
    exec('def fuse_geometry(path):\n    return _file_digest(path)\n', namespace)
    original = namespace['fuse_geometry']
    bound = module.bind_task_fusion_checkpoint(original, identity)
    assert bound(checkpoint) == identity.sha256
    assert original(checkpoint) == 'original strict dependency'
    assert original.__globals__['_file_digest'] is namespace['_file_digest']
    other = tmp_path / 'other.pth'
    other.write_bytes(b'loaded model')
    with pytest.raises(ValueError, match='changed since'):
        bound(other)


def test_unsupported_task_digest_dependency_fails_explicitly(tmp_path):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'loaded model')
    identity = module.LoadedCheckpointIdentity(checkpoint)
    with pytest.raises(RuntimeError, match='no supported checkpoint digest dependency'):
        module.bind_task_fusion_checkpoint(lambda *args: None, identity)
