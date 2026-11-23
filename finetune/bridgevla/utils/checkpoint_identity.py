"""Loaded checkpoint identity with a per-step file metadata guard."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import stat


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'checkpoint must be a regular file: {path}')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True, init=False)
class LoadedCheckpointIdentity:
    """Hash a checkpoint once, then reject changes for this loaded model's lifetime.

    This is scoped to a frozen online agent, not a global path/digest cache.
    Metadata changes require a fresh model load, even if the bytes are identical.
    """
    path: Path
    sha256: str
    signature: tuple

    def __init__(self, path):
        path = Path(path).expanduser().resolve(strict=True)
        before = _file_signature(path)
        digest = sha256_file(path)
        if _file_signature(path) != before:
            raise ValueError('checkpoint changed while computing its loaded identity')
        object.__setattr__(self, 'path', path)
        object.__setattr__(self, 'sha256', digest)
        object.__setattr__(self, 'signature', before)

    def verify_file(self, path):
        try:
            current = Path(path).expanduser().resolve(strict=True)
            matches = current == self.path and _file_signature(current) == self.signature
        except OSError as error:
            raise ValueError('loaded checkpoint is no longer accessible; reload the online model') from error
        if not matches:
            raise ValueError('checkpoint changed since the online model was loaded; reload the online model')
        return self.sha256


def loaded_checkpoint_sha256(path, identity=None):
    if identity is None:
        return sha256_file(path)
    if not isinstance(identity, LoadedCheckpointIdentity):
        raise ValueError('online checkpoint identity must come from the model loader')
    return identity.verify_file(path)
