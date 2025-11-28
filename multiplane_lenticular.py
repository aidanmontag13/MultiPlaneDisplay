from math import sqrt
import onnxruntime as ort
import cv2
import numpy as np
import glob
import os
import time
import open3d as o3d
from skimage.filters import threshold_multiotsu

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
    print(f"Resizing from ({w}, {h}) to ({new_w}, {new_h})")
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

def render_perspective(color, depth, offset, plane_depth):
    depth_range = 0.21718
    viewer_distance = 0.7
    display_width = 0.13596

    depth = depth - np.min(depth)
    depth = depth / np.max(depth)
    depth = depth * depth_range + viewer_distance + plane_depth

    h, w = color.shape[:2]
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    fx = (w / display_width) * viewer_distance
    fy = fx
    cx = w / 2
    cy = h / 2

    color_o3d = o3d.geometry.Image((color * 255).astype(np.uint8))
    depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, convert_rgb_to_intensity=False
    )

    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    # Shift + rotate
    translation = np.array([-offset, 0, 0])
    pcd.translate(translation, relative=True)
    rotation_angle = np.arctan(offset / viewer_distance)
    rotation_matrix = pcd.get_rotation_matrix_from_xyz((0, rotation_angle, 0))
    pcd.rotate(rotation_matrix, center=(0, 0, 0))

    # Project point cloud back to image plane
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Project 3D points to 2D
    x_proj = (points[:, 0] * fx / points[:, 2]) + cx
    y_proj = (points[:, 1] * fy / points[:, 2]) + cy
    
    # Create output image and depth buffer
    warped = np.zeros((h, w, 3), dtype=np.float32)
    depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
    
    # Render points to image, keeping closest point
    valid_mask = (points[:, 2] > 0) & \
                 (x_proj >= 0) & (x_proj < w) & \
                 (y_proj >= 0) & (y_proj < h)
    
    x_proj = x_proj[valid_mask].astype(np.int32)
    y_proj = y_proj[valid_mask].astype(np.int32)
    z_depth = points[valid_mask, 2]
    colors_valid = colors[valid_mask]
    
    for i in range(len(x_proj)):
        # Only update if this point is closer than what's already there
        if z_depth[i] < depth_buffer[y_proj[i], x_proj[i]]:
            warped[y_proj[i], x_proj[i]] = colors_valid[i]
            depth_buffer[y_proj[i], x_proj[i]] = z_depth[i]
    
    return warped

def lenticular_interleave(image1, image2, image3):
    h, w = image1.shape[:2]
    interleaved = np.zeros_like(image1)

    # Assign rows in repeating pattern: row % 3
    interleaved[:, 0::3, :] = image1[:, 0::3, :]
    interleaved[:, 1::3, :] = image2[:, 1::3, :]
    interleaved[:, 2::3, :] = image3[:, 2::3, :]

    return interleaved

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

def soft_threshold_range(depth, low_thresh, high_thresh, rolloff_a, rolloff_b, blur):
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

def create_parrallax_mask(depth, low_thresh, high_thresh, rolloff_a, rolloff_b, blur, screen_distance, eye_separation, viewer_distance):
    close_mask = soft_threshold_range(depth, low_thresh, 1, rolloff_a, 0, blur)
    far_mask = soft_threshold_range(depth, 0, high_thresh, 0, rolloff_b, blur)
    parallax_offset = (eye_separation * screen_distance) / viewer_distance
    print("Parallax offset (meters):", parallax_offset)
    scale_factor = 800 / 0.13596  # screen width in meters
    pixel_offset = int(parallax_offset * scale_factor)
    print("Parallax offset (pixels):", pixel_offset)
    # shift image by pixel offset
    h, w = close_mask.shape
    M = np.float32([[1, 0, pixel_offset], [0, 1, 0]])
    shifted_close_mask = cv2.warpAffine(close_mask, M, (w, h))
    shifted_mask = np.minimum(shifted_close_mask, far_mask)

    return shifted_mask

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
    #cv2.imshow("inpaint_mask", mask)
    mask_vis = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = ((image * 255).astype(np.uint8) // 2) + (mask_vis // 2)
    #cv2.imshow("overlay", overlay)
    inpainted = cv2.inpaint((image * 255).astype(np.uint8), mask_uint8, 3, cv2.INPAINT_TELEA)
    inpainted = inpainted.astype(np.float32) / 255.0
    return inpainted

def magnify_image(image, viewer_distance, screen_distance):
    magnification = (viewer_distance + screen_distance) / viewer_distance
    h = 426
    w = 800
    new_h = int(h * magnification)
    new_w = int(w * magnification)
    magnified = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_y = (new_h - h) // 2
    start_x = (new_w - w) // 2
    cropped = magnified[start_y:start_y + h, start_x:start_x + w]
    return cropped
    

def stack_images(foreground, middleground, background):
    foreground_flipped = cv2.flip(foreground, 0) * (0.33, 0.33, 0.33)
    middleground_flipped = cv2.flip(middleground, 0) * (0.8, 0.9, 1.0)
    background_flipped = cv2.flip(background, 0) * (2.8, 3.15, 3.5)
    combined = np.vstack((background_flipped, middleground_flipped, foreground_flipped))
    combined = cv2.rotate(combined, cv2.ROTATE_90_CLOCKWISE)
    combined = np.clip(combined, 0, 1)
    combined_srgb = (combined ** (1/2.2) * 255.0).astype(np.uint8)
    return combined_srgb

def prepare_image(image_path):
    image = cv2.imread(image_path)
    image = resize_and_crop(image, (800, 426))
    linear_float_image = (image.astype(np.float32) / 255.0) ** 2.2
    inp = preprocess(image)
    depth = infer(inp)
    depth = depth * -1 + 1
    depth = depth ** (3)

    thresholds = threshold_multiotsu(depth, classes=3)

    t1, t2 = thresholds
    #t1 = 0.33
    #t2 = 0.66
    print("Thresholds:", t1, t2)
    rolloff_a = 0.0
    rolloff_b = 0.0
    blur = 3

    binary_middleground_mask = soft_threshold_range(depth, t1 - rolloff_a / 2, 1, 0.0, 0.0, 1)
    binary_background_mask = soft_threshold_range(depth, t2 - rolloff_b / 2, 1, 0.0, 0.0, 1)

    inpainted_middleground = inpaint_mask(linear_float_image, binary_middleground_mask)
    inpainted_background = inpaint_mask(linear_float_image, binary_background_mask)

    viewer_distance = 1000.0
    screen_1_distance = 72.6
    screen_2_distance = 145.2

    foreground_mask = soft_threshold_range(depth, 0, t1, 0.0, rolloff_b, 35)

    middleground_mask = soft_threshold_range(depth, t1, t2, rolloff_b, rolloff_a, 25)
    left_middleground_mask = create_parrallax_mask(depth, t1, t2, rolloff_b, rolloff_a, 25, screen_1_distance, 0.06, viewer_distance)
    right_middleground_mask = create_parrallax_mask(depth, t1, t2, rolloff_b, rolloff_a, 25, screen_1_distance, -0.06, viewer_distance)

    background_mask = soft_threshold_range(depth, t2, 1, rolloff_a, 0, 5)
    left_background_mask = create_parrallax_mask(depth, t2, 1, rolloff_a, 0, 5, screen_2_distance, 0.06, viewer_distance + screen_1_distance)
    right_background_mask = create_parrallax_mask(depth, t2, 1, rolloff_a, 0, 5, screen_2_distance, -0.06, viewer_distance + screen_1_distance)

    combined_mask = foreground_mask + middleground_mask + background_mask
    print("combine mask max", combined_mask.max(), "min", combined_mask.min())

    foreground_image = apply_mask(linear_float_image, foreground_mask)
    left_foreground_image = render_perspective(linear_float_image, depth, -0.06, 0)

    middleground_image = apply_mask(linear_float_image, middleground_mask)
    left_middleground_image = apply_mask(inpainted_middleground, left_middleground_mask)
    right_middleground_image = apply_mask(inpainted_middleground, right_middleground_mask)

    background_image = apply_mask(linear_float_image, background_mask)
    left_background_image = apply_mask(inpainted_background, left_background_mask)
    right_background_image = apply_mask(inpainted_background, right_background_mask)

    middleground_image = magnify_image(middleground_image, viewer_distance, screen_1_distance)
    left_middleground_image = magnify_image(left_middleground_image, viewer_distance, screen_1_distance)
    right_middleground_image = magnify_image(right_middleground_image, viewer_distance, screen_1_distance)
    background_image = magnify_image(background_image, viewer_distance, screen_2_distance)
    left_background_image = magnify_image(left_background_image, viewer_distance, screen_2_distance)
    right_background_image = magnify_image(right_background_image, viewer_distance, screen_2_distance)

    interleaved_middleground_image = lenticular_interleave(left_middleground_image, middleground_image, right_middleground_image)

    interleaved_background_image = lenticular_interleave(left_background_image, background_image, right_background_image)

    cv2.imshow("interleaved_middleground_image", (interleaved_middleground_image ** (1/2.2) * 255).astype(np.uint8))

    combined_image = stack_images(foreground_image, interleaved_middleground_image, interleaved_background_image)

    #cv2.imshow("original_image", (overlay ** (1/2.2) * 255).astype(np.uint8))
    cv2.imshow("depth_map", (depth * 255).astype(np.uint8))
    cv2.imshow("foreground image", (foreground_image ** (1/2.2) * 255).astype(np.uint8))
    cv2.imshow("left foreground image", (left_foreground_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("middleground_image", (middleground_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("left_middleground_image", (left_middleground_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("right_middleground_image", (right_middleground_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("background_image", (background_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("left_background_image", (left_background_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("right_background_image", (right_background_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("inpainted_foreground", (inpainted_foreground ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("right_color", (right_foreground_image ** (1/2.2) * 255).astype(np.uint8))  
    #cv2.imshow("left_color", (left_foreground_image ** (1/2.2) * 255).astype(np.uint8))  
    
    #cv2.imshow("combined_mask", (combined_mask * 255).astype(np.uint8))
    #cv2.imshow("background_image", (background_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("middleground_image", (middleground_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("foreground_image", (foreground_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("background_mask", (background_mask * 255).astype(np.uint8))
    #cv2.imshow("middleground_mask", (middleground_mask * 255).astype(np.uint8))
    #cv2.imshow("foreground_mask", (foreground_mask * 255).astype(np.uint8))
    #cv2.namedWindow("combined_image", cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("combined_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("combined_image", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return combined_image

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
    image_path = "images/raiders.jpg"
    output_image = prepare_image(image_path)
    #time.sleep(10)
    #process_all_images()

if __name__ == "__main__":
    main()
