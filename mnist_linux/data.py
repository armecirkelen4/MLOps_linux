"""Compatibility shim to allow tests to import ``data``.

The real implementation lives in the package ``mnist_linux_proj.data``
inside ``src/``. Tests in this repository import ``data`` directly, so
this module re-exports the required symbols.
"""
from mnist_linux_proj.data import corrupt_mnist, preprocess_data

__all__ = ["corrupt_mnist", "preprocess_data"]
