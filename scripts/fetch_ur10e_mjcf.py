"""Fetch the pinned UR10e Menagerie assets, retaining source/license hashes."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "e4049d0a3bfd58d2a3081614e6777d4007e3f86a"
ROOT = Path("models/mujoco/ur10e")
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/universal_robots_ur10e/"
API = f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/universal_robots_ur10e/assets?ref={REVISION}"


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ViKi-MuJoCo-preview"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "source.json"
    if manifest.exists():
        source = json.loads(manifest.read_text())
        if source.get("commit") != REVISION:
            raise RuntimeError("model cache is from another revision; preserve it in another directory first")
        for name, expected in source["sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"model cache hash mismatch: {name}")
        print("UR10e model cache verified")
        return
    assets = json.loads(download(API))
    names = ["ur10e.xml", "scene.xml", "LICENSE", "README.md"]
    for asset in assets:
        if asset["type"] == "file":
            if Path(asset["name"]).name != asset["name"]:
                raise ValueError("unexpected asset filename")
            names.append("assets/" + asset["name"])

    def fetch(name):
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = download(BASE + name)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(content)
        temporary.replace(path)
        return name, hashlib.sha256(content).hexdigest()

    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(fetch, names))
    manifest.write_text(json.dumps({
        "repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "commit": REVISION, "sha256": hashes,
    }, indent=2))
    print(f"UR10e model cache ready: {len(hashes)} files")


if __name__ == "__main__":
    main()
