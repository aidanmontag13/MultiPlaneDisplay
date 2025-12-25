import cv2
import glob
import numpy as np
import time
import math
import queue
import threading
import os
from ultralytics import YOLO
from picamera2 import Picamera2
# Set Camera FOV as a constant
CAMERA_FOV = 59  # degrees 

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
        main={"size": (1640, 1232), "format": "RGB888"}
    )
    picam2.configure(config)

    picam2.start()
    time.sleep(0.5)

    frame_width, frame_height = 320, 240

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
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(
                frame,
                name,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    return frame

def set_backlight(value):
    for path in glob.glob("/sys/class/backlight/*/brightness"):
        with open(path, "w") as f:
            f.write(str(value))

def headtracker_worker(picam2, model, camera_matrix, dist_coeffs, position_queue, stop_event, idle_event):
    idle_time = 0
    idle_start_time = None
    display_on = True
    while not stop_event.is_set():
        print("idle_time:", idle_time)
        if idle_time > 30:
            if display_on:
                idle_event.set()
                print("Headtracker is idle")
                set_backlight(0)
                display_on = False
                print("Turning off display")
            time.sleep(5)
            print("Headtracker is in idle")
        start_time = time.time()
        frame = picam2.capture_array()

        frame = cv2.resize(frame, (320, 240))
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        if frame is None:
            print("ERROR: Failed to read from camera!")
            break


        # Get facial keypoints from yolo
        results = model(frame, imgsz=320, conf=0.5, verbose=False)
        print("Detected persons:", len(results[0].keypoints.data))
        
        if len(results[0].keypoints.data) > 0 and results[0].keypoints.conf is not None:
            if len(results[0].keypoints.data) > 1:
                print("Multiple persons detected, sending default position")
                try:
                    position_queue.put_nowait((0.0, 0.62, -0.18))
                except queue.Full:
                    pass
                
                continue

            idle_time = 0
            idle_start_time = None
            if not display_on:
                idle_event.clear()
                set_backlight(255)
                display_on = True
                #idle_start_time = None
                print("Turning on display")

            # Get the first person detected
            kp = results[0].keypoints.data[0]
            confidences = results[0].keypoints.conf[0].cpu().numpy()

            #drawn_frame = draw_face_keypoints(frame, kp, confidences)

            #cv2.imshow("Head Tracker Debug", drawn_frame)
            #cv2.waitKey(1)

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

                #print("Using keypoints:", keypoints)
                
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

                        #print(f"Head position: x={x:.3f} m, y={y:.3f} m, z={z:.3f} m")

                        try:
                            position_queue.put_nowait((x, y, z))
                        except queue.Full:
                            pass
                    
            except (IndexError, cv2.error) as e:
                print(f"Error in pose estimation: {e}")

        else:
            if idle_start_time is None:
                idle_start_time = time.time()
            
            else:    
                idle_time = time.time() - idle_start_time
                print(f"Idle time: {idle_time:.3f} s")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Head processing time: {elapsed_time:.3f} s")
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