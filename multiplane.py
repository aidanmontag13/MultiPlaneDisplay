import onnxruntime as ort
import cv2
import numpy as np
import glob
import os
import time
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

def despeckle(mask, kernel_size):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)   # or (5,5) depending on outline thickness
    mask = mask * -1 + 1
    mask = (mask * 255).astype(np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = (closed.astype(np.float32)) / 255.0
    mask = mask * -1 + 1
    return mask

def soft_threshold_range(depth, low_thresh, high_thresh, rolloff, blur):
    mask = np.zeros_like(depth, dtype=np.float32)

    if rolloff == 0:
        mask = ((depth >= low_thresh) & (depth <= high_thresh)).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        mask = despeckle(mask, kernel_size=3)
        return mask
    
    lower_ramp = (depth - (low_thresh - 0.5 * rolloff)) / rolloff
    lower_ramp = np.clip(lower_ramp, 0, 1)
    
    upper_ramp = ((high_thresh + 0.5 * rolloff) - depth) / rolloff
    upper_ramp = np.clip(upper_ramp, 0, 1)
    
    if low_thresh == 0:
        mask = upper_ramp
    elif high_thresh == 1:
        mask = lower_ramp
    else:
        mask = np.minimum(lower_ramp, upper_ramp)

    mask = despeckle(mask, kernel_size=3)
    mask = mask * -1 + 1
    kernel = np.ones((int(blur * 0.3), int(blur * 0.3)), np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    expanded = cv2.dilate(mask_uint8, kernel)
    mask = expanded.astype(np.float32) / 255.0
    mask = mask * -1 + 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    
    return mask

def apply_mask(image, mask):
    mask = cv2.resize(mask, (800, 426), interpolation=cv2.INTER_LINEAR)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = image * mask
    return masked_image

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
    depth = depth ** (1/3)
    thresholds = threshold_multiotsu(depth, classes=3)
    t1, t2 = thresholds
    print("Thresholds:", t1, t2)
    rolloff = 0.10
    blur = 3

    background_mask = soft_threshold_range(depth, 0, t1, rolloff, 35)
    middleground_mask = soft_threshold_range(depth, t1, t2, rolloff, 25)
    foreground_mask = soft_threshold_range(depth, t2, 1, rolloff, 5)

    combined_mask = foreground_mask + middleground_mask + background_mask
    print("combine mask max", combined_mask.max(), "min", combined_mask.min())

    background_image = apply_mask(linear_float_image, background_mask)
    middleground_image = apply_mask(linear_float_image, middleground_mask)
    foreground_image = apply_mask(linear_float_image, foreground_mask)

    viewer_distance = 1000.0
    screen_1_distance = 72.6
    screen_2_distance = 145.2

    middleground_image = magnify_image(middleground_image, viewer_distance, screen_1_distance)
    background_image = magnify_image(background_image, viewer_distance, screen_2_distance)

    combined_image = stack_images(foreground_image, middleground_image, background_image)

    #cv2.imshow("depth_map", (depth * 255).astype(np.uint8))
    #cv2.imshow("combined_mask", (combined_mask * 255).astype(np.uint8))
    #cv2.imshow("background_image", (background_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("middleground_image", (middleground_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("foreground_image", (foreground_image ** (1/2.2) * 255).astype(np.uint8))
    #cv2.imshow("background_mask", (background_mask * 255).astype(np.uint8))
    #cv2.imshow("middleground_mask", (middleground_mask * 255).astype(np.uint8))
    #cv2.imshow("foreground_mask", (foreground_mask * 255).astype(np.uint8))
    cv2.namedWindow("combined_image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("combined_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
    image_path = "/media/multiplane/30D6-C9B5/images/moonlanding2.jpg"
    output_image = prepare_image(image_path)
    #time.sleep(10)
    #process_all_images()

if __name__ == "__main__":
    main()
