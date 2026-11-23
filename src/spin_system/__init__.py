"""Shared package for spin-system generation and neural-network workflows."""

from .core.file_manager import FileManager
from .core.paths import ProjectPaths

__all__ = ["FileManager", "ProjectPaths"]