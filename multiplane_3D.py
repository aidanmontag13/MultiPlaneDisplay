from multiplane_renderer import MultiplaneRenderer
from headtracker2 import HeadTracker
import cv2
import numpy as np
import time

def main():
    multiplane_renderer = MultiplaneRenderer()
    multiplane_renderer.setup_scene()

    headtracker = HeadTracker()
    headtracker.initialize_headtracker()

    cv2.namedWindow("combined_image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "combined_image",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    while True:
        viewer_position = headtracker.acquire_head_positions()

        print("viewer position", viewer_position)

        #viewer_position = np.array([0.1, 0, 1])

        combined = multiplane_renderer.render_scene(viewer_position)

        cv2.imshow("combined_image", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #time.sleep(0.01)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()