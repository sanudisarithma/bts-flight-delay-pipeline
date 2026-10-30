import hashlib
import os
import platform
import sys


def sha256_of(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: " + path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            piece = f.read(1024 * 1024)
            if piece == b"":
                break
            h.update(piece)
    return h.hexdigest()


def system_info():
    import numpy
    import pandas
    import pyarrow

    info = {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["ram_gb"] = "unknown (pip install psutil)"
    return info
