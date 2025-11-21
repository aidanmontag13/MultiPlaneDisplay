import os
import pygame
import time
import glob

IMAGE_DIR = "/home/pi/images"  # Change to your image folder
DISPLAY_TIME = 30              # Seconds per image
FADE_TIME = 2                  # Seconds for fade in/out

def find_usb_images():
    user = os.getlogin()
    media_root = f"/media/{user}"

    image_paths = []

    # Loop over all USB drives
    for drive in os.listdir(media_root):
        drive_path = os.path.join(media_root, drive)
        images_folder = os.path.join(drive_path, "display")

        if os.path.isdir(images_folder):
            print(f"Found images folder on USB: {images_folder}")

            # Add all image files (jpg, png, tiff)
            for ext in ("*.jpg", "*.png", "*.tif", "*.tiff"):
                image_paths.extend(glob.glob(os.path.join(images_folder, ext)))

    return image_paths

def load_image(path):
    img = pygame.image.load(path)
    img = pygame.transform.scale(img, (1280, 800))
    return img

def fade(screen, image, fade_in=True):
    fade_surface = image.copy()
    for alpha in range(0, 256) if fade_in else range(255, -1, -1):
        fade_surface.set_alpha(alpha)
        screen.blit(fade_surface, (0, 0))
        pygame.display.flip()
        pygame.time.delay(int(FADE_TIME * 1000 / 255))

def main():
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((1280, 800), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    image_files = find_usb_images()

    try:
        for img_file in image_files:
            img = load_image(img_file)
            print(f"Displaying image: {img_file}")
            fade(screen, img, fade_in=True)
                
            start_time = time.time()
                
            while time.time() - start_time < 5:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    if event.type == pygame.KEYDOWN:
                        print("Exiting on key press.")
                        pygame.quit()
                        exit()
                time.sleep(0.01)
        fade(screen, img, fade_in=False)

    except KeyboardInterrupt:
        pygame.quit()

if __name__ == "__main__":
    main()