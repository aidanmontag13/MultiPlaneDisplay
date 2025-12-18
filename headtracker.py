import cv2
import numpy as np
import time
import math
import queue
import threading
import os
from ultralytics import YOLO
from picamera2 import Picamera2

# Set Camera FOV as a constant
CAMERA_FOV = 69  # degrees 

# Define 3D model of the face (m)
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],             # Nose tip (origin)
    [-0.030, 0.035, -0.030],        # Right eye
    [0.030, 0.035, -0.030],         # Left eye
    [-0.06, 0.020, -0.095],        # Right ear
    [0.060, 0.020, -0.095],         # Left ear
], dtype=np.float32)

def initialize_headtracker():
    # Load the YOLOv8n-pose model
    model = YOLO("yolov8n-pose.pt")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)

    picam2.start()
    time.sleep(0.5)

    frame_width, frame_height = 640, 480

    # Calculate the camera matrix
    focal_length = frame_width / (2 * np.tan(np.deg2rad(CAMERA_FOV / 2)))
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Apply lens Distorion Correction (assuming none)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    return picam2, model, camera_matrix, dist_coeffs

def headtracker_worker(picam2, model, camera_matrix, dist_coeffs, position_queue, stop_event):
    while not stop_event.is_set():
        frame = picam2.capture_array()
        #frame = cv2.resize(frame, (640, 480))
        if frame is None:
            print("ERROR: Failed to read from camera!")
            break

        # Get facial keypoints from yolo
        results = model(frame, imgsz=320, conf=0.5, verbose=False)
        
        if len(results[0].keypoints.data) > 0 and results[0].keypoints.conf is not None:
            # Get the first person detected
            kp = results[0].keypoints.data[0]
            confidences = results[0].keypoints.conf[0].cpu().numpy()
            
            # Extract keypoints of interest (nose, eyes, ears)
            try:
                # Only use points with high enough confidence 
                CONFIDENCE_THRESHOLD = 0.5
                
                # Set keypoint position to none if not past confidence threshold
                nose = kp[0].cpu().numpy() if confidences[0] > CONFIDENCE_THRESHOLD else None
                right_eye = kp[2].cpu().numpy() if confidences[2] > CONFIDENCE_THRESHOLD else None
                left_eye = kp[1].cpu().numpy() if confidences[1] > CONFIDENCE_THRESHOLD else None
                right_ear = kp[4].cpu().numpy() if confidences[4] > CONFIDENCE_THRESHOLD else None
                left_ear = kp[3].cpu().numpy() if confidences[3] > CONFIDENCE_THRESHOLD else None
                
                valid_model_points = []
                valid_image_points = []
                
                #create array of 2D keypoints
                if confidences[4] > confidences[3]: 
                    keypoints = [(nose, 0), (left_eye, 1), (right_eye, 2), (right_ear, 4)]

                else:
                    keypoints = [(nose, 0), (left_eye, 1), (right_eye, 2), (left_ear, 3)]
                
                # Only keep neccesary data
                for point, idx in keypoints:
                    if point is not None:
                        valid_model_points.append(MODEL_POINTS[idx])
                        valid_image_points.append(point[:2])  # Only x,y coordinates
                
                # Veify we have enough keypoints for PnP (4)
                if len(valid_image_points) >= 4:
                    valid_model_points = np.array(valid_model_points, dtype=np.float32)
                    valid_image_points = np.array(valid_image_points, dtype=np.float32)
                    
                    # Solve for pose
                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        valid_model_points, # 3D points of head model
                        valid_image_points, # 2D keypoints from image
                        camera_matrix,
                        dist_coeffs, 
                        flags=cv2.SOLVEPNP_P3P 
                    )
                    
                    if success:
                        # Extract position (in m)
                        x, z, y = translation_vector.flatten()
                        x = -x
                        z = -z
                        print("sending position", x,y,z)

                        try:
                            position_queue.put_nowait((x, y, z))
                        except queue.Full:
                            pass  # drop frame

            except (IndexError, cv2.error) as e:
                print(f"Error in pose estimation: {e}")

        else:
            x = 0
            z = 0
            y = 0.7
            print("sending empty", x,y,z)

            try:
                position_queue.put_nowait((x, y, z))
            except queue.Full:
                pass  # drop frame

    picam2.stop()
    picam2.close()
    os._exit(0)

def main():
    picam2, model, camera_matrix, dist_coeffs = initialize_headtracker()

    position_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    headtracker_thread = threading.Thread(
        target=headtracker_worker,
        args=(picam2, model, camera_matrix, dist_coeffs, position_queue, stop_event),
        daemon=False,
    )

    headtracker_thread.start()
    time.sleep(15)
    stop_event.set()
    headtracker_thread.join()


if __name__ == "__main__":
    main()