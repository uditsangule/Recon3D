# in src/three_d_recon/methods/rgbd_fusion/tsdf_fusion.py
import numpy as np
import open3d as o3d
from ..base import Reconstructor
from ...core.ir import Scene, ReconArtifacts
from ...core.capabilities import Needs,DataNeeds,ComputeNeeds
from ...core.io_utils import to_open3d_intrinsics, depth_to_meters
from ...utils.parallel import parallel_map, imap_progress

def _load_rgbd(fr):
    import cv2
    color = cv2.imread(fr.rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # to RGB
    depth = cv2.imread(fr.depth_path, cv2.IMREAD_UNCHANGED)
    depth_m = depth_to_meters(depth, scale_mm=True)  # or your scale
    return (fr.id, color, depth_m)

class TSDFReconstructor(Reconstructor):
    name = "tsdf_fusion"

    def need(self) -> Needs:
        return Needs(
            data=DataNeeds(
                rgb="optional",
                intrinsics="required",
                poses="required",
                depth="required",
                masks="optional",
                min_images=5,
            ),
            compute=ComputeNeeds(gpu=False),
            fixups={}
        )

    def prepare(self, scene, workdir): pass

    def run(self, scene: Scene, workdir) -> ReconArtifacts:
        # 1) parallel preload (IO-bound → threads)
        items = parallel_map(_load_rgbd, scene.frames, workers=8, use_processes=False, progress=True)
        preload = {fid: (color, depth) for fid, color, depth in items}

        # 2) sequential integration (shared state)
        intr = to_open3d_intrinsics(scene.frames[0].intrinsics)
        tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01, sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        for fr in imap_progress(scene.frames, total=len(scene.frames)):
            color, depth_m = preload[fr.id]
            rgb = o3d.geometry.Image(color.astype(np.uint8))
            dep = o3d.geometry.Image(depth_m.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, dep, convert_rgb_to_intensity=False, depth_trunc=5.0
            )
            extr = np.linalg.inv(fr.pose.matrix)  # world_T_cam → cam_T_world
            tsdf.integrate(rgbd, intr, extr)

        mesh = tsdf.extract_triangle_mesh(); mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(f"{workdir}/tsdf_mesh.ply", mesh)
        return ReconArtifacts(mesh=f"{workdir}/tsdf_mesh.ply")
