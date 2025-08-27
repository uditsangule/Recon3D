from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Dict, Any, Optional

# What a method needs from the SCENE data
@dataclass
class DataNeeds:
    # Camera models & imagery
    rgb: Literal["required", "optional", "unused"] = "required"
    intrinsics: Literal["required", "optional", "unused"] = "required"
    poses: Literal["required", "preferred", "optional", "unused"] = "optional"
    # Depth/masks/etc. as priors or supervision
    depth: Literal["required", "optional", "unused"] = "optional"
    masks: Literal["required", "optional", "unused"] = "optional"
    normals: Literal["required", "optional", "unused"] = "unused"
    # Minimal dataset characteristics (soft checks)
    min_images: Optional[int] = None
    min_resolution: Optional[tuple[int, int]] = None  # (H, W)
    # Extra keys a method wants to find in Frame.meta
    expected_frame_meta: List[str] = field(default_factory=list)

# What a method needs from the COMPUTE environment
@dataclass
class ComputeNeeds:
    gpu: bool = True
    min_vram_gb: Optional[int] = None
    cuda: Optional[bool] = None           # if specifically needs CUDA
    rocm: Optional[bool] = None           # if specifically needs ROCm
    cpu_threads: Optional[int] = None

# Everything together
@dataclass
class Needs:
    data: DataNeeds = field(default_factory=DataNeeds)
    compute: ComputeNeeds = field(default_factory=ComputeNeeds)
    # Optional hints for the Facade about how to satisfy missing pieces
    # e.g., {"poses": "run_sfm", "masks": "generate_sam"}
    fixups: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data.__dict__,
            "compute": self.compute.__dict__,
            "fixups": dict(self.fixups),
        }
