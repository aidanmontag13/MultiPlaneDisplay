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

H = 213
W = 400

PROJECTION_DISTANCE = 0.3
SCREEN_0_DISTANCE = 0.0
SCREEN_1_DISTANCE = 0.07239
SCREEN_2_DISTANCE = 0.1452
DEPTH_RANGE = 0.21718
DISPLAY_WIDTH = 0.13596

ROLLOFF_A = 0.10
ROLLOFF_B = 0.15

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
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
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

    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    
    return mask

def apply_mask(image, mask):
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = image * mask
    return masked_image

def inpaint_mask(image, mask):
    mask = 1.0 - mask
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = np.ones((int(3), int(3)), np.uint8)
    mask = cv2.dilate(mask_uint8, kernel)
    inpainted = cv2.inpaint((image * 255).astype(np.uint8), mask_uint8, 5, cv2.INPAINT_NS)
    inpainted = inpainted.astype(np.float32) / 255.0
    return inpainted
    
def stack_images(foreground, middleground, background):
    foreground_flipped = cv2.flip(foreground, 0) * (1.0, 1.0, 1.0)
    middleground_flipped = cv2.flip(middleground, 0) * (0.33, 0.33, 0.33)
    background_flipped = cv2.flip(background, 0) * (1.0, 1.0, 1.0)
    #combined = np.vstack((background_flipped, middleground_flipped, foreground_flipped))
    combined = np.vstack((foreground, middleground, background))

    combined = np.clip(combined, 0, 1)
    combined_srgb = (combined ** (1/2.2) * 255.0).astype(np.uint8)
    combined_srgb = cv2.resize(combined_srgb, (400, 640), interpolation=cv2.INTER_LINEAR)
    return combined_srgb

def prepare_planes(image_path):
    image = cv2.imread(image_path)
    image = resize_and_crop(image, (W, H))
    linear_float_image = (image.astype(np.float32) / 255.0) ** 2.2
    inp = preprocess(image)
    depth = infer(inp)
    depth = 1 - depth
    depth = depth ** (3.0)
    ot1, ot2 = threshold_multiotsu(depth, classes=3)

    binary_middleground_mask = create_mask(depth, ot1 - ROLLOFF_A / 2, 1, 0.0, 0.0, 1)
    binary_background_mask = create_mask(depth, ot2 - ROLLOFF_B / 2, 1, 0.0, 0.0, 1)

    middleground = inpaint_mask(linear_float_image, binary_middleground_mask)
    background = inpaint_mask(linear_float_image, binary_background_mask)

    foreground_mask = create_mask(depth, 0, ot1, 0.0, ROLLOFF_A, 3)
    middleground_front_mask = create_mask(depth, ot1, 1.0, ROLLOFF_A, ROLLOFF_B, 11)
    middleground_back_mask = create_mask(depth, 0.0, ot2, ROLLOFF_A, ROLLOFF_B, 11)
    background_mask = create_mask(depth, ot2, 1.0, ROLLOFF_B, 0.0, 21)

    foreground = apply_mask(linear_float_image, foreground_mask)
    middleground = apply_mask(middleground, middleground_back_mask)

    return foreground, middleground, background, middleground_front_mask, background_mask

def shift_mask(mask, screen_distance, viewer_position):
    x, y, z = viewer_position

    shift_x = -(x/y) * screen_distance * (W / DISPLAY_WIDTH)
    shift_y = ((z/y) * screen_distance) * (W / DISPLAY_WIDTH)

    M = np.float32([[1, 0, shift_x],
                [0, 1, shift_y]])

    shifted_mask = cv2.warpAffine(mask, M, (W, H),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=1)

    return shifted_mask

def magnify_image(image, screen_distance, viewer_position):
    x, y, z = viewer_position

    viewer_distance = sqrt(x**2 + y**2 + z**2)

    magnification = (viewer_distance + screen_distance) / viewer_distance

    new_h = int(H * magnification)
    new_w = int(W * magnification)

    magnified = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_y = (new_h - H) // 2
    start_x = (new_w - W) // 2
    cropped = magnified[start_y:start_y + H, start_x:start_x + W]

    return cropped

def renderer_worker(foreground, middleground, background, middleground_mask, background_mask, position_queue, stop_event):
    while not stop_event.is_set():
        print("position_queue", position_queue)
        try:
            position = position_queue.get(timeout=0.1)
            if position is None:
                break
        except queue.Empty:
            continue

        x, y, z = position
        print("got position", x, y, z)

        shifted_middleground_mask = shift_mask(middleground_mask, SCREEN_1_DISTANCE, [x, y, z])
        shifted_background_mask = shift_mask(background_mask, SCREEN_2_DISTANCE, [x, y, z])

        masked_middleground = apply_mask(middleground, shifted_middleground_mask)
        masked_background = apply_mask(background, shifted_background_mask)

        masked_middleground = magnify_image(masked_middleground, SCREEN_1_DISTANCE, [x, y, z])
        masked_background = magnify_image(masked_background, SCREEN_2_DISTANCE, [x, y, z])

        combined_image = stack_images(foreground, masked_middleground, masked_background)

        #cv2.namedWindow("combined_image", cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty("combined_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("combined_image", combined_image)
        position_queue.task_done()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    image_path = "images/moonlanding2.jpg"

    cap, model, camera_matrix, dist_coeffs = headtracker.initialize_headtracker()
    position_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    
    headtracker_thread = threading.Thread(
        target=headtracker.headtracker_worker,
        args=(cap, model, camera_matrix, dist_coeffs, position_queue, stop_event),
        daemon=False,
    )

    foreground, middleground, background, middleground_mask, background_mask = prepare_planes(image_path)

    renderer_thread = threading.Thread(
        target=renderer_worker,
        args=(foreground, middleground, background, middleground_mask, background_mask, position_queue, stop_event),
        daemon=False,
    )

    headtracker_thread.start()
    renderer_thread.start()

    time.sleep(30)
    stop_event.set()
    position_queue.put(None)
    headtracker_thread.join()
    renderer_thread.join()

if __name__ == "__main__":
    main()
