from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float
    dist: Optional[np.ndarray] = None  # k1,k2,p1,p2,k3...


@dataclass
class Extrinsics:
    """Sensor-to-world (s2w) 4x4 homogeneous transform. Defaults to identity."""
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))


@dataclass
class Pose:
    # world_T_cam: 4x4 homogeneous
    matrix: np.ndarray  # shape (4,4)


@dataclass
class Frame:
    id: str
    rgb_path: str
    depth_path: Optional[str]
    surface_normal_path: Optional[str]
    intrinsics: Intrinsics
    pose: Optional[Pose]  # optional if SfM will solve it
    sfm_pose: Optional[Pose]  # optional if we get some corrected pose
    timestamp: Optional[float] = None
    meta: Dict = None  # detections/segm/keypoints/etc
    description: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Sensor:
    id: str
    type: str  # "pinhole", "tof", "lidar", ...
    intrinsics: Intrinsics
    extrinsics: Extrinsics
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    id: str
    frames: List[Frame]
    sensors: Dict[str, Sensor]
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None  # AABB
    extra: Dict = None  # source-specific bits


@dataclass
class ReconArtifacts:
    sparse_pc: Optional[str] = None  # .ply
    dense_pc: Optional[str] = None
    mesh: Optional[str] = None  # .obj/.ply/.glb
    textures: List[str] = None
    nerf_ckpt: Optional[str] = None
    gaussian_ply: Optional[str] = None
    renders_dir: Optional[str] = None
