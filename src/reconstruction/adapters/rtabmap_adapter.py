from pathlib import Path
from .base import DataAdapter
from ..core.ir import Scene, Frame, Intrinsics, Pose
from ..utils.parallel import parallel_map,imap_progress

# from ..core.io_utils import load_opencv_yaml_intrinsics, quat_xyzw_to_matrix
from ..registry.registry import register_adapter

@register_adapter('rtabmap')
class RTABMapAdapter(DataAdapter):
    name = "rtabmap"

    def probe(self, path: str | Path) -> bool:
        # checks if files/folders are present or not!
        p = Path(path)
        return (p / "__rgb").exists() and (p / "__depth").exists() and (p / "__calib").exists()

    def load(self, path: str | Path) -> Scene:
        # parse calib.yml -> Intrinsics
        # read poses.txt -> Pose per frame
        # list rgb/ and depth/ -> Frames
        return Scene(...)