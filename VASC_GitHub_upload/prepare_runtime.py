"""Unpack the bundled CUDA sources after verifying the source manifest."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parent


def prepare(root=ROOT):
    manifest = json.loads((root / 'source_manifest.json').read_text())
    frozen = {item['path'].removeprefix('third_party/'): item['sha256']
                for item in manifest['files'] if item['path'].startswith('third_party/')}
    destination = (root / 'third_party').resolve()
    expected = json.loads((destination / 'runtime_manifest.json').read_text())
    if any(expected.get(name) != digest for name, digest in frozen.items()):
        raise ValueError('Runtime manifest differs from frozen source hashes')
    with zipfile.ZipFile(destination / 'spargeattn.zip') as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Runtime archive contents do not match runtime_manifest.json')
        checked = []
        for name in names:
            relative = PurePosixPath(name)
            path = (destination / relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(destination):
                raise ValueError('Invalid archive path: ' + name)
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected[name]:
                raise ValueError('Runtime source checksum mismatch: ' + name)
            if path.exists() and (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]):
                raise FileExistsError('Refusing to overwrite modified runtime source: ' + str(path))
            checked.append((path, content))
        for path, content in checked:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(content)
    print(f'Runtime ready: {len(checked)} verified source files in {destination / "spargeattn"}')


if __name__ == '__main__':
    prepare()
