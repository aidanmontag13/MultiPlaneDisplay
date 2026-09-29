import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"

import pyrender
from OpenGL import GL

import numpy as np
import trimesh
import cv2
import time

#import headtracker
import queue
import threading

def create_mask(depth, viewer_position, center, low_thresh, high_thresh):
    depth[depth == 0] = 100

    viewer_distance = np.linalg.norm((0, 0, center) + viewer_position)

    normalized_depth = depth - viewer_distance
        
    if low_thresh:
        low_mask = (np.clip(normalized_depth, None, 0) + low_thresh) / (low_thresh)

    else:
        low_mask = np.ones_like(depth)

    if high_thresh:
        high_mask = (high_thresh - np.clip(normalized_depth, 0, None)) / (high_thresh)

    else:
        high_mask = np.ones_like(depth)

    mask = np.minimum(low_mask, high_mask)

    mask = np.clip(mask, 0, 1).astype(np.float32)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
    return mask

def stack_images(composite_list):
    foreground_float = ((composite_list[0] / 255) ** 2.2).astype(np.float32) * 2.0
    middleground_float = ((composite_list[1] / 255) ** 2.2).astype(np.float32) * 2.0
    background_float = ((composite_list[2] / 255) ** 2.2).astype(np.float32) * 2.0

    foreground_float = foreground_float * (0.7, 1.0, 1.3)
    middleground_float = middleground_float * (0.33, 0.33, 0.33)
    background_float = background_float * (0.7, 1.0, 1.2)
    combined = np.vstack((foreground_float, middleground_float, background_float))

    combined = np.clip(combined, 0, 1)
    combined = cv2.resize(combined, (800, 1280), interpolation=cv2.INTER_LINEAR)

    combined = (combined ** (1/2.2) * 255).astype(np.uint8)

    return combined

class MultiplaneRenderer:
    def __init__(self):
        self.display_resolution_x = 800
        self.display_resolution_y = 426
        self.display_width = 0.13596
        self.display_height = 0.0724
        self.meters_to_pixels = self.display_width / self.display_resolution_x
        self.display_distances = [0.0, 0.07, 0.14]

    def setup_scene(self):

        model = trimesh.load('./3Dmodels/model.obj')

        translation_vector = -model.bounding_box.center_mass
        model.apply_translation(translation_vector)

        model_dimensions = model.bounding_box_oriented.extents
        model.apply_scale(self.display_width / (model_dimensions[0]))

        model.apply_translation((0.0, 0.0, -self.display_distances[1]))

        mesh = pyrender.Mesh.from_trimesh(model)

        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])
        self.scene.add(mesh)

        camera = pyrender.IntrinsicsCamera(
            fx=1000,
            fy=1000,
            cx=400,
            cy=213,
            znear=0.01,
            zfar=10
        )

        self.camera_node = self.scene.add(camera)

        self.renderer = pyrender.OffscreenRenderer(self.display_resolution_x, self.display_resolution_y)

    def render_scene(self, viewer_position):
        composite_list = []

        camera_pose = np.array([
            [1.0, 0.0, 0.0, viewer_position[0]],
            [0.0, 1.0, 0.0, viewer_position[1]],
            [0.0, 0.0, 1.0, viewer_position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])

        self.camera_node.matrix = camera_pose

        for display_distance in self.display_distances:
            fx= (viewer_position[2] + display_distance) / self.meters_to_pixels
            cx= (self.display_resolution_x + display_distance) / 2 - viewer_position[0] / self.meters_to_pixels
            cy= (self.display_resolution_y + display_distance) / 2 - viewer_position[1] / self.meters_to_pixels
        
            self.camera_node.camera._fx = fx
            self.camera_node.camera._fy = fx
            self.camera_node.camera._cx = cx
            self.camera_node.camera._cy = cy

            color, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.FLAT)

            mask = create_mask(depth, viewer_position, display_distance, 0.07, 0.07)
            composite = ((color.astype(np.float32) * mask)).astype(np.uint8)

            composite_list.append(composite)

        combined = stack_images(composite_list)

        return combined

def main():
    multiplane_renderer = MultiplaneRenderer()
    multiplane_renderer.setup_scene()

    viewer_position = np.array([0, 0, 1])
    combined = multiplane_renderer.render_scene(viewer_position)

    cv2.namedWindow("combined_image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "combined_image",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    cv2.imshow("combined_image", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
