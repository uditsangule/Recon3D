from __future__ import annotations
from typing import Dict, Any, List, Tuple
from ..core.ir import Scene
from ..core.capabilities import Needs

class NeedViolation(Exception): pass

def check_scene_against_needs(scene: Scene, needs: Needs) -> Tuple[List[str], List[str]]:
    """Returns (missing, warnings) where missing are hard failures."""
    missing: List[str] = []
    warnings: List[str] = []

    n = needs.data

    # rgb
    if n.rgb == "required":
        has_rgb = all(bool(fr.rgb_path) for fr in scene.frames)
        if not has_rgb:
            missing.append("rgb images")

    # intrinsics
    if n.intrinsics == "required":
        if not all(fr.intrinsics is not None for fr in scene.frames):
            missing.append("camera intrinsics")

    # poses
    if n.poses == "required":
        if not all(fr.pose is not None for fr in scene.frames):
            missing.append("camera poses")
    elif n.poses == "preferred":
        if not all(fr.pose is not None for fr in scene.frames):
            warnings.append("poses missing; quality may degrade (SfM fallback needed)")

    # depth
    if n.depth == "required":
        if not all(fr.depth_path for fr in scene.frames):
            missing.append("depth maps")

    # masks
    if n.masks == "required":
        # expect something like frame.meta["mask_path"]
        if not all("mask_path" in (fr.meta or {}) for fr in scene.frames):
            missing.append("masks (frame.meta['mask_path'])")

    # counts / resolution
    if n.min_images is not None and len(scene.frames) < n.min_images:
        warnings.append(f"only {len(scene.frames)} frames; recommended >= {n.min_images}")
    if n.min_resolution is not None:
        Hmin, Wmin = n.min_resolution
        for fr in scene.frames[:10]:  # sample check; full check may be expensive
            intr = fr.intrinsics
            if intr and intr.height and intr.width:
                if intr.height < Hmin or intr.width < Wmin:
                    warnings.append(f"low resolution frame detected (<{Hmin}x{Wmin})")
                    break

    return missing, warnings
