from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .utils import write_json


def probe_environment(output: Path | None = None) -> dict:
    modules = {}
    for name, distribution in [
        ("numpy","numpy"),("scipy","scipy"),("soundfile","soundfile"),
        ("yaml","PyYAML"),("matplotlib","matplotlib"),("pytest","pytest"),
        ("pesq","pesq"),("webrtcvad","webrtcvad-wheels"),
        ("torch","torch"),("gradio","gradio"),
        ("imageio_ffmpeg","imageio-ffmpeg"),
    ]:
        try:
            module = importlib.import_module(name)
            entry = {"available": True, "version": importlib.metadata.version(distribution)}
            if name == "soundfile":
                entry["libsndfile_version"] = module.__libsndfile_version__
            if name == "torch":
                entry["mps_available"] = bool(module.backends.mps.is_available())
                entry["cuda_available"] = bool(module.cuda.is_available())
            modules[name] = entry
        except Exception as exc:
            modules[name] = {"available": False, "error": repr(exc)}
    binaries = {name: {"path": shutil.which(name), "executed": False}
                for name in ["ffmpeg","ffprobe","opusenc","opusdec"]}
    disk = shutil.disk_usage(Path.cwd())
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "machine": platform.machine(),
              "cpu_count_reported": os.cpu_count(), "disk_free_bytes": disk.free,
              "modules": modules, "binaries": binaries,
              "deferred": {"gradio": "M7 interface",
                           "opus_execution": "See per-run codec_runtime.json and channel_validation.json"}}
    if output:
        write_json(output,result)
    return result

