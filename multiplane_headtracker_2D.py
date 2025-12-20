from math import sqrt
import onnxruntime as ort
import cv2
import numpy as np
import glob
import os
import time
import copy

from skimage.filters import threshold_multiotsu

import headtracker
import queue
import threading
from ultralytics import YOLO

H = 426
W = 800

PROJECTION_DISTANCE = 0.3
SCREEN_0_DISTANCE = 0.0
SCREEN_1_DISTANCE = 0.07239
SCREEN_2_DISTANCE = 0.1452
DEPTH_RANGE = 0.21718
DISPLAY_WIDTH = 0.13596

ROLLOFF_A = 0.05
ROLLOFF_B = 0.05

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
        images_folder = os.path.join(drive_path)

        if os.path.isdir(images_folder):
            print(f"Found images folder on USB: {images_folder}")

            # Add all image files (jpg, png, tiff)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
                image_paths.extend(glob.glob(os.path.join(images_folder, ext)))
        else:
            print(f"No images folder found on USB: {drive_path}")
            continue

    return image_paths

def resize_and_crop(img):
    h, w = img.shape[:2]

    if w/h < 1:
        border_size = int((h - w * 1) / 2)
        img = cv2.copyMakeBorder(
            img, 0, 0, border_size, border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    h, w = img.shape[:2]
    
    # Compute scale factor to cover the target
    scale = W / w
    
    # Resize image
    new_w = int(w * scale)
    new_h = int(h * scale)
    #print(f"Resizing from ({w}, {h}) to ({new_w}, {new_h})")
    if w / h > W / H:
        resized = cv2.resize(img, (new_w, H), interpolation=cv2.INTER_LINEAR)
        start_x = (new_w - W) // 2
        cropped = resized[:, start_x:start_x + W]
    else:
        resized = cv2.resize(img, (W, new_h), interpolation=cv2.INTER_LINEAR)
        start_y = (new_h - H) // 2
        cropped = resized[start_y:start_y + H, :]
    
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
    
    lower_ramp = (depth - (low_thresh - 0.5 * rolloff_a)) / (rolloff_a + 1e-6)
    lower_ramp = np.clip(lower_ramp, 0, 1)
    
    upper_ramp = ((high_thresh + 0.5 * rolloff_b) - depth) / (rolloff_b + 1e-6)
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
    foreground_flipped = cv2.flip(foreground, 0) * (0.7, 1.0, 1.3)
    middleground_flipped = cv2.flip(middleground, 0) * (0.33, 0.33, 0.33)
    background_flipped = cv2.flip(background, 0) * (0.7, 1.0, 1.2)
    combined = np.vstack((background_flipped, middleground_flipped, foreground_flipped))
    #combined = np.vstack((foreground, middleground, background))

    combined = np.clip(combined, 0, 1)
    #combined = combined.astype(np.float32)
    combined = cv2.rotate(combined, cv2.ROTATE_90_CLOCKWISE)
    combined = cv2.resize(combined, (1280, 800), interpolation=cv2.INTER_LINEAR)
    return combined

def prepare_planes(image_path):
    image = cv2.imread(image_path)
    image = resize_and_crop(image)
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
    middleground_back_mask = create_mask(depth, 0.0, ot2, ROLLOFF_A, ROLLOFF_B, 3)
    background_mask = create_mask(depth, ot2, 1.0, ROLLOFF_B, 0.0, 11)

    foreground = apply_mask(linear_float_image, foreground_mask)
    middleground = apply_mask(middleground, middleground_back_mask)

    return foreground, middleground, background, linear_float_image, middleground_front_mask, background_mask

def shift_mask(mask, screen_distance, viewer_position):
    max_shift = 100
    x, y, z = viewer_position

    x = x + 0.02
    y = y + 0.08
    z = z - 0.06

    shift_x = -(x/y) * screen_distance * (W / DISPLAY_WIDTH)
    shift_y = ((z/y) * screen_distance) * (W / DISPLAY_WIDTH)

    if shift_x > max_shift:
        shift_x = max_shift
    if shift_x < -max_shift:
        shift_x = -max_shift

    if shift_y > max_shift:
        shift_y = max_shift
    if shift_y < -max_shift:
        shift_y = -max_shift    

    M = np.float32([[1, 0, shift_x],
                [0, 1, shift_y]])

    shifted_mask = cv2.warpAffine(mask, M, (W, H),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE,
                          borderValue=0)

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

def interpolate_position(position, previous_position, alpha):
    smoothed_position = [
            alpha * position[0] + (1 - alpha) * previous_position[0],
            alpha * position[1] + (1 - alpha) * previous_position[1],
            alpha * position[2] + (1 - alpha) * previous_position[2],
        ]

    return smoothed_position

def renderer_worker(foreground, middleground, background, merged, middleground_mask, background_mask, position_queue, render_queue, render_stop_event):
    alpha = 0.2
    smoothed_position = [0, 0.7, 0]
    target_position = [0, 0.7, 0] 

    flat_image = np.zeros((1280, 800, 3), dtype=np.float32)
    merged_srgb = (merged * (0.7, 1.0, 1.3)) ** (1 / 2.2)
    flat_image[0:426, 0:800] = merged_srgb
    flat_image = cv2.flip(flat_image, 0)
    flat_image = cv2.rotate(flat_image, cv2.ROTATE_90_CLOCKWISE)
    render_queue.put(flat_image)
    time.sleep(3)

    while not render_stop_event.is_set():
        # Try to get a new target position
        start_time = time.time()
        try:
            new_position = position_queue.get_nowait()
            if new_position is not None:
                target_position = list(new_position)
                print("got position", new_position)
            position_queue.task_done()
        except queue.Empty:
            pass  # no new position, keep last target

        # Interpolate toward target
        smoothed_position = interpolate_position(target_position, smoothed_position, alpha)

        x, y, z = smoothed_position

        shifted_middleground_mask = shift_mask(middleground_mask, SCREEN_1_DISTANCE, [x, y, z])
        shifted_background_mask = shift_mask(background_mask, SCREEN_2_DISTANCE, [x, y, z])

        masked_middleground = apply_mask(middleground, shifted_middleground_mask)
        masked_background = apply_mask(background, shifted_background_mask)

        masked_middleground = magnify_image(masked_middleground, SCREEN_1_DISTANCE, [0, 0.7, 0])
        masked_background = magnify_image(masked_background, SCREEN_2_DISTANCE, [0, 0.7, 0])

        combined_image = stack_images(foreground, masked_middleground, masked_background)

        combined_image_srgb = combined_image ** (1 / 2.2)

        try:
            render_queue.put_nowait(combined_image_srgb)
        except queue.Full:
            pass
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Render time: {elapsed_time:.4f}s")

    black_image = np.zeros((800, 1280, 3), dtype=np.float32)
    render_queue.put(black_image)

def display_worker(render_queue, stop_event, idle_event):
    alpha = 0.1 

    current_image = np.zeros((800, 1280, 3), dtype=np.float32)
    target_image  = np.zeros((800, 1280, 3), dtype=np.float32)

    cv2.namedWindow("combined_image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "combined_image",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    while not stop_event.is_set():

        if idle_event.is_set():
            print("display idle on")
            time.sleep(3)
            continue
        
        start_time = time.time()
        try:
            while True:
                new_image = render_queue.get_nowait()
                if new_image is None:
                    stop_event.set()
                    break
                target_image = new_image.astype(np.float32)
                render_queue.task_done()
        except queue.Empty:
            pass
        
        current_image += alpha * (target_image - current_image)
        display = np.clip(current_image * 255, 0, 255).astype(np.uint8)
        cv2.imshow("combined_image", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Frame time: {elapsed_time:.4f}s")

def main():
    images = find_usb_images()

    cap, model, camera_matrix, dist_coeffs = headtracker.initialize_headtracker()
    position_queue = queue.Queue(maxsize=1)
    render_queue = queue.Queue(maxsize=2)

    stop_event = threading.Event()
    render_stop_event = threading.Event()
    idle_event = threading.Event()
    
    headtracker_thread = threading.Thread(
        target=headtracker.headtracker_worker,
        args=(cap, model, camera_matrix, dist_coeffs, position_queue, stop_event, idle_event),
        daemon=False,
    )

    display_thread = threading.Thread(
        target=display_worker,
        args=(render_queue, stop_event, idle_event),
        daemon=False,
    )

    headtracker_thread.start()
    display_thread.start()

    for image_path in images:
        if idle_event.is_set():
            print(" renderer idle on")
            time.sleep(3)
            continue

        foreground, middleground, background, merged, middleground_mask, background_mask = prepare_planes(image_path)

        renderer_thread = threading.Thread(
            target=renderer_worker,
            args=(foreground, middleground, background, merged, middleground_mask, background_mask, position_queue, render_queue, render_stop_event),
            daemon=False,
        )

        renderer_thread.start()

        time.sleep(30)
    
        render_stop_event.set()
        renderer_thread.join()
        render_stop_event.clear()

    stop_event.set()
    position_queue.put(None)
    headtracker_thread.join()
    display_thread.join()

if __name__ == "__main__":
    main()