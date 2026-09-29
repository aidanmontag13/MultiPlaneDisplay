import cv2
import glob
import numpy as np
import time
import math
import queue
import threading
import os
import argparse
from ultralytics import YOLO
from picamera2 import Picamera2
from functools import partial

def draw_face_keypoints(frame, keypoints, confidences, conf_thresh=0.5):

    # YOLOv8 pose indices
    labels = {
        0: ("nose", (0, 255, 255)),
        1: ("left_eye", (255, 0, 0)),
        2: ("right_eye", (0, 0, 255)),
        3: ("left_ear", (255, 255, 0)),
        4: ("right_ear", (0, 255, 0)),
    }

    for idx, (name, color) in labels.items():
        if confidences[idx] > conf_thresh:
            x, y = keypoints[idx][:2].cpu().numpy().astype(int)
            print(f"Drawing keypoint: {name} at ({x}, {y}) with confidence {confidences[idx]:.2f}")
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(
                frame,
                name,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,                    1,
                cv2.LINE_AA,
            )

    return frame

def set_backlight(value):
    for path in glob.glob("/sys/class/backlight/*/brightness"):
        with open(path, "w") as f:
            f.write(str(value))

class HeadTracker:
    def __init__(self):
        self.camera_fov = 59
        self.model_points = [
            (0.0, 0.0, 0.0,),             # Nose tip (origin)
            (-0.030, 0.035, -0.030),        # Right eye
            (0.030, 0.035, -0.030),         # Left eye
            (-0.06, 0.020, -0.095),        # Right ear
            (0.060, 0.020, -0.095),         # Left ear
        ]

        self.frame_width = 320
        self.frame_height = 240

        self.focal_length = self.frame_width / (2 * np.tan(np.deg2rad(self.camera_fov / 2)))
        self.center = (self.frame_width / 2, self.frame_height / 2)

        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Apply lens Distorion Correction (assuming none)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        self.confidence_threshold = 0.5
        self.default_position = np.array([0, 0, 1])

    def initialize_headtracker(self):
        # Load the YOLOv8n-pose model

        self.model = YOLO("yolov8n-pose.pt")

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (1640, 1232), "format": "RGB888"}
        )

        self.picam2.configure(config)

        self.picam2.start()
        time.sleep(0.5)

    def acquire_head_positions(self):
        frame = self.picam2.capture_array()
        frame = cv2.resize(frame, (320, 320))
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        results = self.model(frame, imgsz=320, conf=self.confidence_threshold, verbose=False)

        viewer_position = self.default_position

        if len(results[0].keypoints.data) > 0 and results[0].keypoints.conf is not None:

            try:
                kp = results[0].keypoints.data[0]
                confidences = results[0].keypoints.conf[0].cpu().numpy()

                nose = kp[0].cpu().numpy() if confidences[0] > self.confidence_threshold else None
                right_eye = kp[2].cpu().numpy() if confidences[2] > self.confidence_threshold else None
                left_eye = kp[1].cpu().numpy() if confidences[1] > self.confidence_threshold else None
                right_ear = kp[4].cpu().numpy() if confidences[4] > self.confidence_threshold else None
                left_ear = kp[3].cpu().numpy() if confidences[3] > self.confidence_threshold else None

                keypoints = [nose, right_eye, left_eye, right_ear, left_ear]

                print("keypoints", keypoints)

                valid_model_points = []
                valid_image_points = []

                for model_point, keypoint in zip(self.model_points, keypoints):
                    if keypoint is not None:
                        valid_model_points.append(model_point)
                        valid_image_points.append(keypoint[:2])

                valid_image_points = np.asarray(
                    valid_image_points,
                    dtype=np.float32
                )

                valid_model_points = np.asarray(
                    valid_model_points,
                    dtype=np.float32
                )

                print("valid image points", valid_image_points)
                print("valid model points", valid_model_points)
                
                if len(valid_model_points) > 3:
                    print("enought for pnp!")
                    
                    # Solve for pose
                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        valid_model_points, # 3D points of head model
                        valid_image_points, # 2D keypoints from image
                        self.camera_matrix,
                        self.dist_coeffs, 
                        flags=cv2.SOLVEPNP_SQPNP
                    )

                    print("sucess", success)
                        
                    if success:
                        print("PNP Sucess!")
                        # Extract position (in m)
                        x, z, y = translation_vector.flatten()
                        x = -x
                        z = -z

                        viewer_position = np.array([x, z, y])
                        print("Found position!!")

            except Exception as e:
                print("Error:", e)
    
        print("viewer pos", viewer_position)
        return viewer_position
