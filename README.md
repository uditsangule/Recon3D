# 3D-Recon

A modular 3D reconstruction toolkit that normalizes any input (iPhone LiDAR/ToF, RTAB-Map, RealityCapture, etc.) and runs multiple reconstruction families: RGB-D fusion (TSDF), Photogrammetry/MVS, NeRF, 3D Gaussian Splatting, with SLAM/VIO pose correction and AI hooks (detections/segmentation) built in.

---
## Approach

We keep a stable Scene IR (intermediate representation) and plug everything else via clean boundaries:

- Adapter → convert external data ➜ Scene
- Strategy (Method) → run a reconstruction algorithm on Scene
- Facade (Pipeline) → a friendly one-liner that orchestrates steps
- Registry → discover adapters/methods/hooks by name (and via plugins)

---
## DataAdapters
- **RTAB-Map** (`rtabmap`) 
    - expects `__rgb/`, `__depth/`, `__calib/`, `poses.txt` 
    - supports `start`/`stop`/`skip` frame selection.
- **ARKit / iPhone** (`arkit_iphone`) — *planned*
- **RealityCapture** (`realitycapture`) — *planned*

---
## Methods (Strategies)
- **RGB-D** 
  - TSDF Fusion — *working*
- **Photogrammetry / MVS** 
  - colmap — *planned*
- **NeRF** 
  - nerfstudio — *planned*
- **3D Gaussian Splatting** 
  - gaussian_splatting — *planned*

---
## Features
- Pipelines (run_pipeline) and registry (adapters/methods)
- SLAM hook interface to fix/estimate poses (e.g., COLMAP SfM, ORB-SLAM3, DROID-SLAM)
- Perception hooks on frame/pointcloud/mesh (YOLO/SAM/etc.) — optional
- Utilities: parallel (threads/processes), system_info, metrics (PSNR/SSIM/Chamfer), IO & camera helpers
- Config-driven: easy to swap adapters/methods, or run auto-selection

---
## Config:

- sample_tsdf_rtabmap.yaml
```
# configs/tsdf_rtabmap.yaml
name: room_tsdf_baseline
pipeline: tsdf
adapter: rtabmap
method: tsdf_fusion

inputs:
  root: /abs/path/to/ScanName
  rtabmap:
    start: 0        # subset windows for big scans (1k–10k frames)
    stop: null
    skip: 1
    format: 11      # pose format of exported data.

tsdf:
  voxel_length: 0.01
  sdf_trunc: 0.05
  depth_trunc_m: 5.0
  depth_scale: 1000    # "mm" | "meters" | <number>

execution:
  workers: 8
  use_processes: false

outputs:
  save_mesh: true
  save_points: true
```
---
## Architecture
```
flowchart LR
  A[Adapter\n(external data → Scene)] --> S[Scene IR]
  S --> P[Pipeline (Facade)]
  P -->|need()/check| M[Method (Strategy)]
  P -->|optional| SL[SLAM/VIO\n(PoseEstimator)]
  SL --> S2[Scene with fixed poses]
  S2 --> M
  M --> AR[Artifacts\n(mesh/pc/ckpt/renders)]
  P --> H[Perception Hooks\n(frame/pc/mesh)]
```

## Integrating Custom code

- DataAdapters
```
# src/reconstruction/adapters/my_adapter.py
from reconstruction.adapters.base import DataAdapter
from reconstruction.registry.registry import register_adapter
from reconstruction.core.ir import Scene, Frame, Sensor, Intrinsics, Pose

@register_adapter("my_adapter")
class MyAdapter(DataAdapter):
    def __init__(self, my_option: int = 42):
        self.my_option = my_option

    def probe(self, path: str) -> bool:
        # return True if you can parse the given folder
        ...

    def load(self, path: str) -> Scene:
        # parse files → build Frames (with intrinsics & optional poses)
        frames = [...]
        sensor = Sensor(id="cam0", type="pinhole",
                        intrinsics=Intrinsics(...))  # Extrinsics default to Identity
        return Scene(id="my_scene", frames=frames, sensors={"cam0": sensor})
```
Enable registration by importing it once (e.g., in adapters/__init__.py) or by shipping as a plugin.
- use in config
```
adapter: my_adapter

inputs:
  my_adapter:
    my_option: 123
```
- adding new method (Strategy)
    Interface: src.reconstruction.methods.base.Reconstructor (+ optional need())
```
# src/reconstruction/methods/awesome/awesome_method.py
from reconstruction.methods.base import Reconstructor
from reconstruction.registry.registry import register_method
from reconstruction.core.ir import Scene, ReconArtifacts

@register_method("awesome_method")
class AwesomeReconstructor(Reconstructor):
    def need(self):
        from three_d_recon.core.capabilities import Needs, DataNeeds, ComputeNeeds
        return Needs(
            data=DataNeeds(rgb="required", intrinsics="required", poses="preferred"),
            compute=ComputeNeeds(gpu=True, min_vram_gb=8),
            fixups={"poses": "run_sfm"}  # pipeline may call SLAM if poses missing
        )

    def prepare(self, scene, workdir): ...
    def run(self, scene, workdir) -> ReconArtifacts:
        # do heavy work; write mesh/pc/ckpt, return paths
        return ReconArtifacts(mesh=f"{workdir}/awesome_mesh.ply")
    def export(self, arts, outdir): ...
```