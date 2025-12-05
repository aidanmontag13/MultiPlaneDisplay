from math import sqrt
import onnxruntime as ort
import cv2
import numpy as np
import glob
import os
import time
import copy
import open3d as o3d
from skimage.filters import threshold_multiotsu

import headtracker
import queue
import threading
from ultralytics import YOLO

H = 426
W = 800
VIEWER_DISTANCE = 0.7
viewer_distance = 0.700
SCREEN_0_DISTANCE = 0.0
SCREEN_1_DISTANCE = 0.07239
SCREEN_2_DISTANCE = 0.1452
DISPLAY_WIDTH = 0.13596

# Load model
session = ort.InferenceSession(
    "models/depth_anything/depth_anything_v2_vits.onnx",
    providers=['CPUExecutionProvider']
)

# Get correct input name
input_name = session.get_inputs()[0].name

def find_usb_images():
    user = "multiplane"
    media_root = f"/media/{user}"

    image_paths = []

    # Loop over all USB drives
    for drive in os.listdir(media_root):
        drive_path = os.path.join(media_root, drive)
        images_folder = os.path.join(drive_path, "images")
        output_folder = os.path.join(drive_path, "display")

        if os.path.isdir(images_folder):
            print(f"Found images folder on USB: {images_folder}")

            # Add all image files (jpg, png, tiff)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
                image_paths.extend(glob.glob(os.path.join(images_folder, ext)))
        else:
            print(f"No images folder found on USB: {drive_path}")
            continue

    return image_paths, output_folder

def resize_and_crop(img, target_size):
    target_w, target_h = target_size
    h, w = img.shape[:2]
    
    # Compute scale factor to cover the target
    scale = target_w / w
    
    # Resize image
    new_w = int(w * scale)
    new_h = int(h * scale)
    #print(f"Resizing from ({w}, {h}) to ({new_w}, {new_h})")
    if w / h > target_w / target_h:
        resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        start_x = (new_w - target_w) // 2
        cropped = resized[:, start_x:start_x + target_w]
    else:
        resized = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_LINEAR)
        start_y = (new_h - target_h) // 2
        cropped = resized[start_y:start_y + target_h, :]
    
    return cropped

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (518, 518))
    img = img.astype(np.float32) / 255.0

    # model normalization
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = img[None, ...]          # add batch dim
    return img

def infer(inp):
    out = session.run(None, {input_name: inp})
    depth = out[0][0]

    # rescale depth for saving
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    return depth

def create_pointcloud(color, depth):
    depth_range = 0.21718
    projection_distance = VIEWER_DISTANCE
    display_width = DISPLAY_WIDTH

    #depth = depth - np.min(depth)
    #depth = depth / np.max(depth)
    depth = depth * depth_range + (projection_distance - depth_range / 6)

    h, w = color.shape[:2]
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    fx = (w / display_width) * projection_distance
    fy = fx
    cx = W / 2
    cy = H / 2

    color_o3d = o3d.geometry.Image((color).astype(np.uint8))
    depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, convert_rgb_to_intensity=False
    )

    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pcd_og = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd

def despeckle(mask, kernel_size):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)   # or (5,5) depending on outline thickness
    mask = mask * -1 + 1
    mask = (mask * 255).astype(np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = (closed.astype(np.float32)) / 255.0
    mask = mask * -1 + 1
    return mask

def blur_and_dilate(mask, blur_size):
    mask = mask * -1 + 1
    kernel = np.ones((int(blur_size * 0.3), int(blur_size * 0.3)), np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    expanded = cv2.dilate(mask_uint8, kernel)
    mask = expanded.astype(np.float32) / 255.0
    mask = mask * -1 + 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return mask

def create_mask(depth, low_thresh, high_thresh, rolloff_a, rolloff_b, blur):
    mask = np.zeros_like(depth, dtype=np.float32)

    if rolloff_a == 0 and rolloff_b == 0:
        mask = ((depth >= low_thresh) & (depth <= high_thresh)).astype(np.float32)
        mask = blur_and_dilate(mask, blur)
        mask = despeckle(mask, kernel_size=3)
        mask = cv2.resize(mask, (800, 426), interpolation=cv2.INTER_LINEAR)
        return mask
    
    lower_ramp = (depth - (low_thresh - 0.5 * rolloff_a)) / rolloff_a
    lower_ramp = np.clip(lower_ramp, 0, 1)
    
    upper_ramp = ((high_thresh + 0.5 * rolloff_b) - depth) / rolloff_b
    upper_ramp = np.clip(upper_ramp, 0, 1)
    
    if low_thresh == 0:
        mask = upper_ramp
    elif high_thresh == 1:
        mask = lower_ramp
    else:
        mask = np.minimum(lower_ramp, upper_ramp)

    mask = despeckle(mask, kernel_size=3)
    mask = blur_and_dilate(mask, blur)

    mask = cv2.resize(mask, (800, 426), interpolation=cv2.INTER_LINEAR)
    
    return mask

def apply_mask(image, mask):
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = image * mask
    return masked_image

def inpaint_mask(image, mask):
    mask = 1.0 - mask
    mask = cv2.resize(mask, (800, 426), interpolation=cv2.INTER_LINEAR)
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = np.ones((int(3), int(3)), np.uint8)
    mask = cv2.dilate(mask_uint8, kernel)
    inpainted = cv2.inpaint((image * 255).astype(np.uint8), mask_uint8, 5, cv2.INPAINT_NS)
    inpainted = inpainted.astype(np.float32) / 255.0
    return inpainted

def render_perspective(point_cloud, viewer_position, inpaint=True):
    pcd = copy.deepcopy(point_cloud)
    fx = (W / DISPLAY_WIDTH) * VIEWER_DISTANCE
    fy = fx
    cx = W / 2
    cy = H / 2

    # move point cloud to center
    translation = np.array([0, 0, -VIEWER_DISTANCE])
    pcd.translate(translation, relative=True)

    # calculate pitch and yaw angle to keep camera centered
    yaw_angle = np.arctan((viewer_position[0]) / viewer_position[1])
    pitch_angle = np.arctan((viewer_position[2]) / viewer_position[1])

    # rotate object
    rotation_matrix = pcd.get_rotation_matrix_from_xyz((pitch_angle, yaw_angle, 0))
    pcd.rotate(rotation_matrix, center=(0, 0, 0))

    # find offset
    viewer_distance = np.linalg.norm(viewer_position)
    offset_translation = np.array([0, 0, viewer_distance])
    pcd.translate(offset_translation, relative=True)

    #origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    #o3d.visualization.draw_geometries([pcd, pcd_og, origin])

    # Project point cloud back to image plane
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Project 3D points to 2D
    x_proj = (points[:, 0] * fx / points[:, 2]) + cx
    y_proj = (points[:, 1] * fy / points[:, 2]) + cy
    
    # Create output image and depth buffer
    warped = np.zeros((H, W, 3), dtype=np.float32)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
    
    # Render points to image, keeping closest point
    valid_mask = (points[:, 2] > 0) & \
                 (x_proj >= 0) & (x_proj < W) & \
                 (y_proj >= 0) & (y_proj < H)
    
    x_proj = x_proj[valid_mask].astype(np.int32)
    y_proj = y_proj[valid_mask].astype(np.int32)
    z_depth = points[valid_mask, 2]
    colors_valid = colors[valid_mask]
    
    for i in range(len(x_proj)):
        # Only update if this point is closer than what's already there
        if z_depth[i] < depth_buffer[y_proj[i], x_proj[i]]:
            warped[y_proj[i], x_proj[i]] = colors_valid[i]
            depth_buffer[y_proj[i], x_proj[i]] = z_depth[i]

    mask = (depth_buffer == np.inf).astype(np.uint8)
    mask = 1 - mask
    
    if inpaint == True:
        warped = inpaint_mask(warped, mask)
        depth_buffer = inpaint_mask(depth_buffer, mask)
    
    return warped, depth_buffer

def reproject_2D(color, viewer_position, plane_depth):
    display_width = 0.13596
    projection_distance = 0.7
    
    h, w = color.shape[:2]
    fx = (w / display_width) * projection_distance
    fy = fx
    cx = w / 2
    cy = h / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    src_points = np.float32([
        [0, 0],           # Top-left
        [w, 0],           # Top-right
        [w, h],           # Bottom-right
        [0, h]            # Bottom-left
    ])

    pts_h = np.hstack([src_points, np.ones((src_points.shape[0], 1))])

    K_inv = np.linalg.inv(K)

    pts_3d = (K_inv @ pts_h.T).T

    #print("3d points", pts_3d)

    # create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    pcd_og = o3d.geometry.PointCloud()
    pcd_og.points = o3d.utility.Vector3dVector(pts_3d)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # Project corner points
    offset_depth = 1 - (plane_depth / projection_distance)
    #print("offset depth", offset_depth)
    t1 = np.array([0, 0, -offset_depth])
    pcd.translate(t1, relative=True)
    
    yaw_angle = np.arctan2(viewer_position[0], viewer_position[1])
    pitch_angle = np.arctan2(viewer_position[2], viewer_position[1])

    rotation_matrix = pcd.get_rotation_matrix_from_xyz((pitch_angle, yaw_angle, 0))
    pcd.rotate(rotation_matrix, center=(0, 0, 0))

    # find offset
    viewer_distance = np.linalg.norm(viewer_position) / projection_distance
    #print("viewer distance", viewer_distance)
    #print("viwer distance", viewer_distance)
    offset_translation = np.array([0, 0, viewer_distance])
    pcd.translate(offset_translation, relative=True)

    #o3d.visualization.draw_geometries([pcd, pcd_og])
    
    transformed = np.asarray(pcd.points)  # (4×3)

    # Reproject to 2D
    proj = (K @ transformed.T).T
    dst_pts = proj[:, :2] / proj[:, 2:3]     # normalize by Z
    
    # Apply 2D warp
    H = cv2.getPerspectiveTransform(src_points, dst_pts.astype(np.float32))
    H_inv = np.linalg.inv(H)
    warped = cv2.warpPerspective(color, H_inv, (w, h))
    
    return warped
    

def stack_images(foreground, middleground, background):
    foreground_flipped = cv2.flip(foreground, 0) #* (0.33, 0.33, 0.33)
    middleground_flipped = cv2.flip(middleground, 0) #* (0.8, 0.9, 1.0)
    background_flipped = cv2.flip(background, 0) #* (2.8, 3.15, 3.5)
    black_row = np.zeros((2, 800, 3), dtype=foreground_flipped.dtype)
    combined = np.vstack((background_flipped, middleground_flipped, foreground_flipped, black_row))

    combined = cv2.rotate(combined, cv2.ROTATE_90_CLOCKWISE)
    combined = np.clip(combined, 0, 1)
    combined_srgb = (combined ** (1/2.2) * 255.0).astype(np.uint8)
    return combined_srgb

def prepare_plane(image, depth, viewer_position, plane_depth, t1, t2, rolloff_a, rolloff_b, blur):
    #linear_perspective_image = (image.astype(np.float32)) ** 2.2
    #mask = create_mask(depth, t1, t2, rolloff_a, rolloff_b, blur)
    #masked_image = apply_mask(linear_perspective_image, mask)
    warped_image = reproject_2D(image, viewer_position, plane_depth)

    return warped_image

def initialize_render(image_path):
    image = cv2.imread(image_path)
    image = resize_and_crop(image, (800, 426))
    inp = preprocess(image)
    depth = infer(inp)
    depth = depth * -1 + 1
    depth = depth ** (2)

    thresholds = threshold_multiotsu(depth, classes=3)

    t1, t2 = thresholds
    t1 = t1 * 0.21759 + (VIEWER_DISTANCE - SCREEN_1_DISTANCE / 2)
    t2 = t2 * 0.21759 + (VIEWER_DISTANCE - SCREEN_1_DISTANCE / 2)
    #t1 = screen_1_distance + viewer_distance
    #t2 = screen_2_distance + viewer_distance
    t3 = 100
    #print("Thresholds:", t1, t2)
    rolloff_a = SCREEN_1_DISTANCE / 4
    rolloff_b = SCREEN_1_DISTANCE / 4
    #rolloff_a = 0
    #rolloff_b = 0
    blur = 3

    pcd = create_pointcloud(image, depth)

    return pcd

def renderer_worker(pcd, position_queue, stop_event, inpaint=False):
    while not stop_event.is_set():
        print("position_queue", position_queue)
        position = position_queue.get()

        x, y, z = position
        print("got position", x, y, z)
        

        render, depth = render_perspective(pcd, [x, y, z], inpaint)

        #foreground = prepare_plane(render, depth, [x, y, z], SCREEN_0_DISTANCE, 0, t1, 0, rolloff_a, 3)
        #middleground = prepare_plane(render, depth, [x, y, z], screen_1_distance, t1, t2, rolloff_a, rolloff_b, 11)
        #background = prepare_plane(render, depth, [x, y, z], screen_2_distance, t2, t3, rolloff_b, 0, 21)

        #combined_image = stack_images(foreground, middleground, background) 

        #cv2.namedWindow("combined_image", cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty("combined_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #cv2.imshow("combined_image", combined_image)
        cv2.imshow("rendered_perspective", (render * 255).astype(np.uint8))
        position_queue.task_done()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def process_all_images():
    images, output_folder = find_usb_images()
    
    if not images:
        print("No USB drive with images found!")
        return
    
    print(f"Found {len(images)} images in folder.")

    for img_path in images:
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        if not os.path.exists(f"{output_folder}/{image_name}.png"):
            print(f"Processing {img_path}...")
            output_image = prepare_image(img_path)
            cv2.imwrite(f"{output_folder}/{image_name}.png", output_image)
            print(f"Saved display/{image_name}.png")

def main():
    image_path = "images/godzilla.jpg"

    cap, model, camera_matrix, dist_coeffs = headtracker.initialize_headtracker()
    position_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    
    headtracker_thread = threading.Thread(
        target=headtracker.headtracker_worker,
        args=(cap, model, camera_matrix, dist_coeffs, position_queue, stop_event),
        daemon=False,
    )

    pcd = initialize_render(image_path)

    inpaint=False

    renderer_thread = threading.Thread(
        target=renderer_worker,
        args=(pcd, position_queue, stop_event),
        daemon=False,
    )

    headtracker_thread.start()
    renderer_thread.start()

    time.sleep(30)
    stop_event.set()
    headtracker_thread.join()
    renderer_thread.join()

if __name__ == "__main__":
    main()
