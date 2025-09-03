from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.registry.registry import register_adapter
from .base import DataAdapter
from ..core.ir import Scene, Frame, Sensor,Pose, Intrinsics, Extrinsics
from ..core import pose_utils as pu
from src.utils.parallel import imap_progress

_NUM_RE = re.compile(r"(\d+)")


def _num_from_name(p: Path) -> Optional[int]:
    m = _NUM_RE.search(p.stem)
    return int(m.group(1)) if m else None


def _natural_indexed(dirpath: Path, exts: Tuple[str, ...]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if not dirpath.exists():
        return out
    lowers = tuple(e.lower() for e in exts)
    for p in dirpath.iterdir():
        if p.is_file() and p.suffix.lower() in lowers:
            n = _num_from_name(p)
            if n is not None and n not in out:
                out[n] = p
    return out


def _slice_indices(indices: List[int], start: Optional[int], stop: Optional[int], skip: int) -> List[int]:
    s = max(0, int(start)) if start else 0
    e = min(len(indices), int(stop)) if stop else len(indices)
    k = max(1, int(skip))
    return indices[s:e:k]

def _decompose_calib(calib_file):
    with open(calib_file, 'r') as f:
        for _ in range(2): f.readline()
        calib = yaml.safe_load(f)
    intr = Intrinsics(matrix=np.array(calib['camera_matrix']['data']).reshape(3,3))
    E = np.eye(4)
    E[:3,:] = np.array(calib['local_transform']['data']).reshape(3,4)
    extr = Extrinsics(matrix=E)
    width = calib.get('image_width', None)
    height = calib.get('image_height', None)
    id = calib.get('camera_name')
    sensor = Sensor(id=id,intrinsics=intr,extrinsics=extr , width=width, height=height , type='calib-rtabmap')
    return sensor

def _load_poses(pose_path, format=11, delimiter=","):
    """

    :param pose_path: file path of pose file.
    :param format: Format used for exported poses (default is 11):
                              0=Raw 3x4 transformation matrix (r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz)
                              1=RGBD-SLAM (in motion capture coordinate frame)
                              2=KITTI (same as raw but in optical frame)
                              3=TORO
                              4=g2o
                              10=RGBD-SLAM in ROS coordinate frame (stamp x y z qx qy qz qw)
                              11=RGBD-SLAM in ROS coordinate frame + ID (stamp x y z qx qy qz qw id)

    :param delimiter:
    :return:
    """
    if not pose_path.exists():
        raise FileNotFoundError(f"{pose_path} does not exist")
    poses = pd.read_csv(pose_path, delimiter=delimiter)
    new_h = ['id', 'timestamp']
    if not len(poses):
        raise RuntimeError(f"pose file has :{len(poses)} entries")
    # converting every pose to raw type
    Tmat = np.zeros((len(poses), 4, 4))


    if format in[0,2]:
        Tmat[:,:3,:] = poses.to_numpy().reshape(len(poses),3,4)
        Tmat[:,3,3] = 1
        poses_new = pd.DataFrame({"id":list(range(1,len(poses) + 1)) , "timestamp":[None] * len(poses) , 'Tmatrix':list(Tmat)})

    if format in [10,11]:
        # convert into 4 x 4 matrix
        poses_new = poses[['id', '#timestamp']].copy()
        poses_new.columns = new_h
        Tmat[:,:3,:3] = [pu.q_to_Rotation(q=p , order='xyzw') for p in poses[['qx','qy','qz','qw']].to_numpy()]
        Tmat[:,:3,3] = poses[['x','y','z']].to_numpy()
        Tmat[:,3,3] = 1
    else:
        raise RuntimeError(f"No proper format found for pose file {pose_path}")
    poses_new['Tmatrix'] = list(Tmat)
    return poses_new



@register_adapter("rtabmap")
class RTABMapAdapter(DataAdapter):
    name = "rtabmap"

    def __init__(self, start=None, step=1, stop=None, parallel=False, format = 11):
        self.start = start
        self.stop = stop
        self.step = step
        self.in_parallel = parallel
        self.pose_format = format
        return

    def probe(self, path: str | Path) -> bool:
        # checks if files/folders are present or not!
        p = Path(path)
        return (p / "__rgb").exists() and (p / "__depth").exists() and (p / "__calib").exists()

    def load(self, path: str | Path, scan_name: str = None) -> Scene:
        dir_path = Path(path)
        rgb_map = _natural_indexed(dir_path / "__rgb", (".jpg", ".jpeg", ".png"))
        depth_map = _natural_indexed(dir_path / "__depth", (".png", ".exr", ".tiff", ".tif"))
        calib_map = _natural_indexed(dir_path / "__calib", (".yaml", ".yml", ".json"))
        poses = _load_poses(pose_path=dir_path / "__poses.txt", delimiter=" " , format=self.pose_format)

        if not rgb_map:   raise FileNotFoundError("No RGB files under '__rgb'")
        if not depth_map: raise FileNotFoundError("No depth files under '__depth'")
        if not calib_map: raise FileNotFoundError("No calib files under '__calib'")


        ids = sorted(set(rgb_map).intersection(depth_map)) if not 'id' in poses.keys() else poses['id'].tolist()
        if not ids:
            raise RuntimeError("RGB/Depth indices do not overlap | some images are missing")

        ids = _slice_indices(ids, self.start, self.stop, self.step)

        frames = []

        for idx in imap_progress(ids):
            rgb_path = rgb_map[idx].as_posix()
            depth_path = depth_map[idx].as_posix()

            sensor = _decompose_calib(calib_map[idx].as_posix())
            pose_item = poses[poses.id == idx]
            pose = Pose(id=pose_item.id.values[0], timestamp=pose_item.timestamp.values[0], matrix=pose_item.Tmatrix.values[0])
            frames.append(Frame(rgb_path=rgb_path, depth_path=depth_path, id=idx, sensor=sensor, pose=pose,
                                surface_normal_path=None, sfm_pose=None))
        return Scene(frames=frames, sensors=None, id=dir_path.name)
