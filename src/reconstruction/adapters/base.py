from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from ..core.ir import Scene

class DataAdapter(ABC):
    """External dataset → Scene IR."""
    name: str = "base"

    @abstractmethod
    def probe(self, path: str | Path) -> bool:
        """Quick check if this adapter can handle the given path."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> Scene:
        """Parse input into a Scene (frames, intrinsics, poses, metadata)."""
        ...