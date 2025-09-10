from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import torch
import numpy as np
import cv2
import os
from read_write_model import *


def get_scale_for_single_image(image_key, cameras, images, points3d, mono_depth_map):
    """
    Calculates the scale factor for a single image by comparing COLMAP points
    to a provided monocular depth map.
    """
    image_meta = images[image_key]
    camera_meta = cameras[image_meta.camera_id]

    # Get the 3D points visible in this camera view
    pts_3d_idx = image_meta.point3D_ids
    valid_mask = (pts_3d_idx != -1)
    
    if valid_mask.sum() == 0:
        return None # No COLMAP points visible in this image

    visible_pts_3d_idx = pts_3d_idx[valid_mask]
    visible_xys = image_meta.xys[valid_mask]
    
    # Get the corresponding 3D points from the point cloud
    colmap_pts = np.array([points3d[pt_id].xyz for pt_id in visible_pts_3d_idx])

    # Transform points to the camera's coordinate system
    R = qvec2rotmat(image_meta.qvec)
    T = image_meta.tvec
    pts_in_camera_frame = colmap_pts @ R.T + T
    
    # Get the depth from the Z-coordinate (depth is positive)
    colmap_depths = pts_in_camera_frame[:, 2]

    # Ensure monocular depth map is float32
    mono_depth_map = mono_depth_map.astype(np.float32)

    # Remap/sample the monocular depth at the exact locations of the COLMAP keypoints
    # We need to scale the keypoint coordinates if the depth map resolution differs
    h, w = mono_depth_map.shape[:2]
    scale_x = w / camera_meta.width
    scale_y = h / camera_meta.height
    
    remap_coords = visible_xys.copy()
    remap_coords[:, 0] *= scale_x
    remap_coords[:, 1] *= scale_y

    map1 = remap_coords[:, 0].astype(np.float32)
    map2 = remap_coords[:, 1].astype(np.float32)

    mono_depths_at_keypoints = cv2.remap(
        mono_depth_map,
        map1, # Now guaranteed to be float32
        map2, # Now guaranteed to be float32
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    ).flatten()

    # Filter for valid depth pairs (both must be positive)
    valid_depth_mask = (colmap_depths > 0) & (mono_depths_at_keypoints > 0)
    
    if valid_depth_mask.sum() < 10:
        return None # Not enough corresponding points to be reliable

    colmap_depths = colmap_depths[valid_depth_mask]
    mono_depths_at_keypoints = mono_depths_at_keypoints[valid_depth_mask]
    
    # Use median to find a robust ratio, avoiding outliers
    scale = np.median(colmap_depths) / np.median(mono_depths_at_keypoints)
    
    return scale


def calculate_global_depth_scale(base_dir):
    """
    Calculates a single global depth scale for the entire dataset.
    
    Args:
        colmap_data: The object containing cameras, images, and points3D from COLMAP.
        image_dir: The directory where your input images are stored.
    
    Returns:
        A single float representing the global depth scale.
    """

    cam_intrinsics, images_metas, points3d = read_model(os.path.join(base_dir, "sparse", "0"), ext=f".bin")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    depth_model = DepthAnythingV2(**model_configs[encoder])
    depth_model.load_state_dict(torch.load(f'utils/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_model = depth_model.to('cuda').eval()

    print("Calculating global depth scale...")
    
    all_scales = []
    depth_maps = {}
    
    # 2. Loop through all images registered in COLMAP
    for image_key in tqdm(images_metas.keys(), desc="Processing images for depth scale"):
        image_name = images_metas[image_key].name
        image_path = f"{base_dir}/images/{image_name}"
        
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Call DepthAnythingV2 to get the depth map
        mono_depth_map = depth_model.infer_image(image)
        depth_maps[image_name] = mono_depth_map.astype(np.float32)

        # 4. Get the scale for this single image using our helper
        scale = get_scale_for_single_image(
            image_key,
            cam_intrinsics,
            images_metas,
            points3d,
            mono_depth_map
        )
        
        if scale is not None and scale > 0:
            all_scales.append(scale)

    if not all_scales:
        raise RuntimeError("Could not compute depth scale. Check COLMAP and depth data.")

    # 5. Calculate the median of all scales for a robust global value
    global_scale = np.median(all_scales)
    
    print(f"Computed global depth scale: {global_scale}")
    return global_scale, depth_maps


# global_scale, _ = calculate_global_depth_scale('E:\\Learning\\hcmus\\computer_vision\\3DGS\DashGaussian\\Dataset\\train')
# print(global_scale)