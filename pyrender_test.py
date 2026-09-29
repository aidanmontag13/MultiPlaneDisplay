import numpy as np
import trimesh
import pyrender
import cv2
import time

DISPLAY_RESOLUTION_X = 800 #pixels
DISPLAY_RESOLUTION_Y = 426 #pixels

DISPLAY_WIDTH = 0.13596 #meters
DISPLAY_HEIGHT = 0.0724 #meters

METERS_TO_PIXELS = DISPLAY_WIDTH / DISPLAY_RESOLUTION_X

DISPLAY_1_DISTANCE = 0.0 #meters
DISPLAY_2_DISTANCE = 0.07 #meters
DISPLAY_3_DISTANCE = 0.14 #meters

DISPLAY_1 = np.array([
        [-DISPLAY_WIDTH / 2 + 0.001, -DISPLAY_HEIGHT / 2 + 0.001, DISPLAY_1_DISTANCE],
        [DISPLAY_WIDTH / 2 - 0.001, -DISPLAY_HEIGHT / 2 + 0.001, DISPLAY_1_DISTANCE],
        [-DISPLAY_WIDTH / 2 + 0.001, DISPLAY_HEIGHT / 2 - 0.001, DISPLAY_1_DISTANCE],
        [DISPLAY_WIDTH / 2 - 0.001, DISPLAY_HEIGHT / 2 - 0.001, DISPLAY_1_DISTANCE],
    ])

DISPLAY_2 = np.array([
        [-DISPLAY_WIDTH / 2 + 0.001, -DISPLAY_HEIGHT / 2 + 0.001, -DISPLAY_2_DISTANCE],
        [DISPLAY_WIDTH / 2 - 0.001, -DISPLAY_HEIGHT / 2 + 0.001, -DISPLAY_2_DISTANCE],
        [-DISPLAY_WIDTH / 2 + 0.001, DISPLAY_HEIGHT / 2 - 0.001, -DISPLAY_2_DISTANCE],
        [DISPLAY_WIDTH / 2 - 0.001, DISPLAY_HEIGHT / 2 - 0.001, -DISPLAY_2_DISTANCE],
    ])

DISPLAY_3 = np.array([
        [-DISPLAY_WIDTH / 2 + 0.001, -DISPLAY_HEIGHT / 2 + 0.001, -DISPLAY_3_DISTANCE],
        [DISPLAY_WIDTH / 2 - 0.001, -DISPLAY_HEIGHT / 2 + 0.001, -DISPLAY_3_DISTANCE],
        [-DISPLAY_WIDTH / 2 + 0.001, DISPLAY_HEIGHT / 2 - 0.001, -DISPLAY_3_DISTANCE],
        [DISPLAY_WIDTH / 2 - 0.001, DISPLAY_HEIGHT / 2 - 0.001, -DISPLAY_3_DISTANCE],
    ])

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

faces = np.array([
    [0, 1, 2],
    [3, 2, 1]
])

plane_model = trimesh.Trimesh(
    vertices=DISPLAY_3,
    faces=faces,
    process=False
)

plane_mesh = pyrender.Mesh.from_trimesh(plane_model)

model = trimesh.load('./3Dmodels/model.obj')

translation_vector = -model.bounding_box.center_mass
model.apply_translation(translation_vector)

model_dimensions = model.bounding_box_oriented.extents
model.apply_scale(DISPLAY_WIDTH / (model_dimensions[0]))

model.apply_translation((0.0, 0.0, -DISPLAY_2_DISTANCE))

mesh = pyrender.Mesh.from_trimesh(model)

scene = pyrender.Scene()
scene.add(mesh)
#scene.add(plane_mesh)

viewer_position = np.array([0, 0, 5])

camera_1 = pyrender.IntrinsicsCamera(
    fx= (viewer_position[2] + DISPLAY_1_DISTANCE) / METERS_TO_PIXELS,
    fy= (viewer_position[2] + DISPLAY_1_DISTANCE) / METERS_TO_PIXELS,
    cx= (DISPLAY_RESOLUTION_X + DISPLAY_1_DISTANCE) / 2 - viewer_position[0] / METERS_TO_PIXELS,
    cy= (DISPLAY_RESOLUTION_Y + DISPLAY_1_DISTANCE) / 2 - viewer_position[1] / METERS_TO_PIXELS,
    znear=0.001,
    zfar=10
)

camera_2 = pyrender.IntrinsicsCamera(
    fx= (viewer_position[2] + DISPLAY_2_DISTANCE) / METERS_TO_PIXELS,
    fy= (viewer_position[2] + DISPLAY_2_DISTANCE) / METERS_TO_PIXELS,
    cx= (DISPLAY_RESOLUTION_X + DISPLAY_2_DISTANCE) / 2 - viewer_position[0] / METERS_TO_PIXELS,
    cy= (DISPLAY_RESOLUTION_Y + DISPLAY_2_DISTANCE) / 2 - viewer_position[1] / METERS_TO_PIXELS,
    znear=0.001,
    zfar=10
)

camera_3 = pyrender.IntrinsicsCamera(
    fx= (viewer_position[2] + DISPLAY_3_DISTANCE) / METERS_TO_PIXELS,
    fy= (viewer_position[2] + DISPLAY_3_DISTANCE) / METERS_TO_PIXELS,
    cx= (DISPLAY_RESOLUTION_X + DISPLAY_3_DISTANCE) / 2 - viewer_position[0] / METERS_TO_PIXELS,
    cy= (DISPLAY_RESOLUTION_Y + DISPLAY_3_DISTANCE) / 2 - viewer_position[1] / METERS_TO_PIXELS,
    znear=0.001,
    zfar=100
)

camera_pose = np.array([
    [1.0, 0.0, 0.0, viewer_position[0]],
    [0.0, 1.0, 0.0, viewer_position[1]],
    [0.0, 0.0, 1.0, viewer_position[2]],
    [0.0, 0.0, 0.0, 1.0],
])

camera_node_1 = scene.add(camera_1, pose=camera_pose)
camera_node_2 = scene.add(camera_2, pose=camera_pose)
camera_node_3 = scene.add(camera_3, pose=camera_pose)

renderer = pyrender.OffscreenRenderer(DISPLAY_RESOLUTION_X, DISPLAY_RESOLUTION_Y)

start_time = time.time()
scene.main_camera_node = camera_node_1
color_1, depth_1 = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
end_time = time.time()
print("render took", end_time - start_time, "seconds")

start_time = time.time()
scene.main_camera_node = camera_node_2
color_2, depth_2 = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
end_time = time.time()
print("render took", end_time - start_time, "seconds")

start_time = time.time()
scene.main_camera_node = camera_node_3
color_3, depth_3 = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
end_time = time.time()
print("render took", end_time - start_time, "seconds")

mask_1 = create_mask(depth_1, viewer_position, 0.0, None, 0.07)
composite_1 = ((color_1.astype(np.float32) * mask_1)).astype(np.uint8) 

mask_2 = create_mask(depth_2, viewer_position, 0.07, 0.07, 0.07)
composite_2 = ((color_2.astype(np.float32) * mask_2)).astype(np.uint8) 

mask_3 = create_mask(depth_3, viewer_position, 0.14, 0.07, None)
composite_3 = ((color_3.astype(np.float32) * mask_3)).astype(np.uint8) 

print("min depth = ", np.max(mask_1))

combined_masks = (mask_1 + mask_2 + mask_3)
print("combined_masks max", np.max(combined_masks))

cv2.imshow("mask1", mask_1 * 255)
cv2.imshow("mask2", mask_2 * 255)
cv2.imshow("mask3", mask_3 * 255)
cv2.imshow("color1", composite_1)
cv2.imshow("color2", composite_2)
cv2.imshow("color3", composite_3)
cv2.waitKey(0)
cv2.destroyAllWindows()