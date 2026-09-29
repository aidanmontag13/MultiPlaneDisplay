import glfw
import numpy as np
import trimesh
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import ctypes
import time
import math
import cv2
import multiplane_headtracker_3D_windows as multiplane

H = 213
W = 400
PROJECTION_DISTANCE = 0.3
SCREEN_0_DISTANCE = 0.0
SCREEN_1_DISTANCE = 0.07239
SCREEN_2_DISTANCE = 0.1452
DEPTH_RANGE = 0.21718
DISPLAY_WIDTH = 0.13596

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
out vec3 fragColor;
uniform mat4 MVP;
void main()
{
    fragColor = color;
    gl_Position = MVP * vec4(position, 1.0);
    gl_PointSize = 2.0;  // size of points
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
out vec4 FragColor;
void main()
{
    FragColor = vec4(fragColor, 1.0);
}
"""

def depth_to_pointcloud(color, depth, fx, fy, cx, cy):
    """
    Convert depth + color images to a point cloud (Nx3) and colors (Nx3)
    """
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    
    z = 1-depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    points = np.stack((x, -y, -z), axis=-1).reshape(-1, 3)  # flip y,z to match OpenGL
    colors = color.reshape(-1, 3) / 255.0  # normalize to 0-1
    
    return points.astype(np.float32), colors.astype(np.float32)

def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(fov / 2)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -eye @ np.array([s, u, -f])
    return m

import numpy as np

def rotate_pointcloud(points, angle_x=0, angle_y=0, angle_z=0):
    # Convert degrees to radians
    ax = np.radians(angle_x)
    ay = np.radians(angle_y)
    az = np.radians(angle_z)

    # Rotation matrices
    Rx = np.array([
        [1, 0,      0     ],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax),  np.cos(ax)]
    ], dtype=np.float32)

    Ry = np.array([
        [ np.cos(ay), 0, np.sin(ay)],
        [ 0,          1, 0        ],
        [-np.sin(ay), 0, np.cos(ay)]
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az),  np.cos(az), 0],
        [0,          0,           1]
    ], dtype=np.float32)

    # Rotate around the center of the point cloud
    center = points.mean(axis=0)
    points_centered = points - center

    # Apply rotations: R = Rz * Ry * Rx
    rotated = points_centered @ Rx.T
    rotated = rotated @ Ry.T
    rotated = rotated @ Rz.T

    # Translate back
    rotated += center

    return rotated


def render_pointcloud_to_image(points, colors, width=800, height=600):
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    # Hidden window (offscreen)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, "Offscreen", None, None)
    glfw.make_context_current(window)

    # Compile shader
    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # Vertex positions
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # Colors
    cbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, cbo)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)
    glEnable(GL_DEPTH_TEST)

    # Camera
    proj = perspective(math.radians(60), width/height, 0.1, 10.0)
    view = look_at(np.array([0, 0, 1.0], dtype=np.float32),
                   np.array([0, 0, 0], dtype=np.float32),
                   np.array([0, 1, 0], dtype=np.float32))
    MVP = proj @ view

    # Create framebuffer
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # Texture to render into
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)

    # Depth buffer
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("Framebuffer not complete")

    # Render
    glViewport(0, 0, width, height)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(shader)
    glUniformMatrix4fv(glGetUniformLocation(shader, "MVP"), 1, GL_FALSE, MVP)

    glBindVertexArray(vao)
    glDrawArrays(GL_POINTS, 0, len(points))
    glBindVertexArray(0)

    # Read pixels
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    image = np.flip(image, axis=0)  # Flip vertically

    # Cleanup
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glfw.terminate()

    return image

def main():
    image_folder = r"C:\Users\aidan\Documents\MultiplaneRepo\images"
    image_paths = multiplane.find_folder_images(image_folder)

    image = cv2.imread(image_paths[0])
    image = multiplane.resize_and_crop(image)
    inp = multiplane.preprocess(image)
    depth = multiplane.infer(inp)

    fx = (800 / DISPLAY_WIDTH) * PROJECTION_DISTANCE
    fy = fx
    cx = W / 2
    cy = H / 2

    points, colors = depth_to_pointcloud(image, depth, fx, fy, cx, cy)
    points = rotate_pointcloud(points, 180, 356, 180)

    angle_x = 0
    angle_y = 0
    angle_z = 0

    while True:
        updated_points = rotate_pointcloud(points, angle_x, angle_y, angle_z)

        img = render_pointcloud_to_image(updated_points, colors, width=800, height=600)
        print("Rendered image shape:", img.shape)
        cv2.imshow("Point Cloud", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        angle_y += 0
        print("Angle y:", angle_y)
        time.sleep(0.2)

if __name__ == "__main__":
    main()



