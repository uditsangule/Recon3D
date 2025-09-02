# in src/three_d_recon/methods/rgbd_fusion/tsdf_fusion.py
import cv2
import numpy as np
import open3d as o3d

from ..base import Reconstructor
from ...core.capabilities import Needs, DataNeeds, ComputeNeeds
from ...core.ir import Scene, ReconArtifacts
# from ...core.io_utils import to_open3d_intrinsics, depth_to_meters
from src.registry.registry import register_method
from src.utils.parallel import parallel_map, imap_progress


def show_image(image, windowname='output', waitkey=0, dest=False):
    """
    image vizualizer
    """
    cv2.namedWindow(winname=windowname, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(windowname, image)
    cv2.waitKey(waitkey)
    if dest or waitkey == 0: cv2.destroyWindow(winname=windowname)
    return


def _load_rgbd(fr):
    import cv2
    color = cv2.imread(fr.rgb_path, cv2.IMREAD_ANYCOLOR)  # to RGB
    depth = cv2.imread(fr.depth_path, cv2.IMREAD_UNCHANGED)
    if depth.shape != color.shape[:2]:
        c2d_scale = np.array([depth.shape[1] / color.shape[1], depth.shape[0] / color.shape[0], 1])
        depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        #fr.sensor.intrinsics.matrix = fr.sensor.intrinsics.matrix * c2d_scale
        #fr.sensor.width = color.shape[1]
        #fr.sensor.height = color.shape[0]

    return (fr, fr.id, color[:,:,::-1], depth)


def _to_open3d_intrinsics(fr):
    sensor = fr.sensor
    intr = o3d.camera.PinholeCameraIntrinsic(width=sensor.width or 0, height=sensor.height or 0,
                                             intrinsic_matrix=sensor.intrinsics.matrix)
    return intr


@register_method("tsdf_fusion")
class TSDFReconstructor(Reconstructor):
    name = "tsdf_fusion"

    def __init__(self, voxel_length=0.01, sdf_trunc=0.05, depth_trunc_m=5.0, depth_scale=1000, workers=8,
                 use_processes=False, save_points=True):
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        self.depth_trunc = depth_trunc_m
        self.depth_scale = depth_scale
        self.workers = workers
        self.use_processes = use_processes
        self.save_points = save_points
        return

    def need(self) -> Needs:
        return Needs(
            data=DataNeeds(rgb="optional", intrinsics="required", poses="required", depth="required", masks="optional",
                min_images=5, ), compute=ComputeNeeds(gpu=False), fixups={})

    def prepare(self, scene, workdir):
        pass

    def run(self, scene: Scene, workdir) -> ReconArtifacts:
        # 1) parallel preload (IO-bound → threads)
        allow_preload = True if (len(scene.frames) < 200) else False
        require_fragments = True if (len(scene.frames) > 1000) else False

        if allow_preload:
            items = parallel_map(_load_rgbd, scene.frames, workers=self.workers, use_processes=self.use_processes,progress=True)
            # for fr in imap_progress(scene.frames):  fr, items = _load_rgbd(fr)
            scene.frames = [i[0] for i in items]
            preload = {fid: (color, depth) for _, fid, color, depth in items}

        tsdf = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=self.voxel_length, sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        last = None
        for fr in imap_progress(scene.frames, total=len(scene.frames)):
            if allow_preload:
                color, depth_m = preload[fr.id]
            else:
                _, color, depth_m = _load_rgbd(fr)
            rgb = o3d.geometry.Image(color.astype(np.uint8))
            dep = o3d.geometry.Image(depth_m.astype(np.float32))
            intr = _to_open3d_intrinsics(fr)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, dep, convert_rgb_to_intensity=False,
                depth_trunc=self.depth_trunc, depth_scale=self.depth_scale)

            local_transform = np.linalg.inv(fr.sensor.extrinsics.matrix) # cam_T_imu
            extr = local_transform @ np.linalg.inv(fr.pose.matrix)  # world_T_cam → cam_T_world

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd, intrinsic=intr, extrinsic= extr)
            # o3d.visualization.draw_geometries([pcd] +[last] + [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=[0,0,0])])
            last = pcd

            tsdf.integrate(rgbd, intr, extr)

        mesh = tsdf.extract_triangle_mesh();
        mesh.compute_vertex_normals()
        
        o3d.visualization.draw_geometries([mesh])
        mesh_path = f"{workdir}/{self.name}_mesh.ply"
        o3d.io.write_triangle_mesh(mesh_path, mesh, print_progress=True)
        if self.save_points:
            pcd_path = f"{workdir}/{self.name}_pointcloud.ply"
            point_cloud = tsdf.extract_point_cloud();
            point_cloud.estimate_normals()
            point_cloud.orient_normals_towards_camera_location()
            o3d.io.write_point_cloud(filename=pcd_path, pointcloud=point_cloud, print_progress=True)
            o3d.visualization.draw_geometries([point_cloud])
        return ReconArtifacts(mesh=mesh_path, sparse_pc=pcd_path)
