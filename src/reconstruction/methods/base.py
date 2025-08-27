from abc import ABC, abstractmethod
from pathlib import Path

from ..core.capabilities import Needs, DataNeeds, ComputeNeeds
from ..core.ir import Scene, ReconArtifacts


class Reconstructor(ABC):
    """A reconstruction Strategy (photogrammetry, nerf, 3dgs, ...)."""
    name: str = "base"
    # Declare requirements so the Facade can enforce / precompute as needed:
    requires_poses: bool = False
    supports_depth: bool = False

    def need(self) -> Needs:
        """
        Return the data+compute requirements for this method.
        Provide conservative defaults that most methods can live with.
        """
        return Needs(
            data=DataNeeds(
                rgb="required",
                intrinsics="required",
                poses="optional",
                depth="optional",
                masks="optional",
                min_images=2,
            ),
            compute=ComputeNeeds(
                gpu=False,  # default safe; methods override if they require GPU
                min_vram_gb=None,
            ),
            fixups={}
        )

    @abstractmethod
    def prepare(self, scene: Scene, workdir: str | Path) -> None:
        """Export datasets / do preflight checks."""
        ...

    @abstractmethod
    def run(self, scene: Scene, workdir: str | Path) -> ReconArtifacts:
        """Do the heavy lifting and return artifact paths."""
        ...

    def export(self, artifacts: ReconArtifacts, outdir: str | Path) -> None:
        """Optional: copy/convert artifacts into final outdir structure."""
        ...
