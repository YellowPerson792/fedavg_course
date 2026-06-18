from __future__ import annotations

import platform
import subprocess


def read_pi_status() -> dict[str, str]:
    if platform.system().lower() != "linux":
        return {}
    status: dict[str, str] = {}
    for key, command in {
        "pi_temp": ["vcgencmd", "measure_temp"],
        "pi_throttled": ["vcgencmd", "get_throttled"],
    }.items():
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            continue
        if completed.returncode == 0:
            status[key] = completed.stdout.strip()
    return status
