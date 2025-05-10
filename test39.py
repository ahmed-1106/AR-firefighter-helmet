# All your imports remain unchanged
import cv2
import numpy as np
from ultralytics import YOLO
import time

class FirefighterARSimulator:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.model = YOLO('yolov8n.pt')
        self.detection_history = []
        self.detection_stability = 3
        
        self.home_position = np.array([0.0, 0.0])
        self.current_position = np.array([0.0, 0.0])
        self.body_heading = 0
        self.path_history = [tuple(self.current_position)]
        self.step_size = 1.0

        self.return_path = []

        self.walls = [
            ((2, -2), (2, 3)),
            ((-2, 1), (3, 1)),
            ((-1, -1), (-1, 2)),
            ((1, 2), (1, 4))
        ]

        self.window_name = "Firefighter AR Thermal Navigation (ESC to quit)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        self.panel_height = 220
        self.minimap_size = 320
        self.minimap_scale = 40
        self.arrow_size = 60

        self.arrows = self._create_arrows()
        self.thermal_colormap = self._create_thermal_colormap()

    def _create_arrows(self):
        arrows = {}
        size = self.arrow_size

        arrows['forward'] = np.zeros((size, size, 3), np.uint8)
        cv2.arrowedLine(arrows['forward'], (30, 45), (30, 15), (0, 255, 0), 3)
        arrows['back'] = np.zeros((size, size, 3), np.uint8)
        cv2.arrowedLine(arrows['back'], (30, 15), (30, 45), (0, 0, 255), 3)
        arrows['left'] = np.zeros((size, size, 3), np.uint8)
        cv2.arrowedLine(arrows['left'], (45, 30), (15, 30), (255, 255, 0), 3)
        arrows['right'] = np.zeros((size, size, 3), np.uint8)
        cv2.arrowedLine(arrows['right'], (15, 30), (45, 30), (255, 255, 0), 3)
        arrows['arrival'] = np.zeros((size, size, 3), np.uint8)
        cv2.putText(arrows['arrival'], "EXIT", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return arrows

    def _create_thermal_colormap(self):
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            intensity = 255 - i
            colormap[i, 0, 0] = intensity
            colormap[i, 0, 1] = intensity
            colormap[i, 0, 2] = min(255, intensity * 2)
        return colormap

    def _apply_thermal_effect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        heat = cv2.GaussianBlur(gray, (5, 5), 0)
        normalized = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
        thermal = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        mask = cv2.threshold(normalized, 200, 255, cv2.THRESH_BINARY)[1]
        glow = cv2.GaussianBlur(mask, (21, 21), 0)
        thermal = cv2.addWeighted(thermal, 1.0, cv2.applyColorMap(glow, cv2.COLORMAP_HOT), 0.4, 0)
        return thermal

    def _detect_objects(self, thermal_frame):
        detection_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_RGB2BGR)
        results = self.model(detection_frame, verbose=False)[0]
        self.detection_history.append(results)
        if len(self.detection_history) > self.detection_stability:
            self.detection_history.pop(0)
        return self.detection_history[-1]

    def _check_wall_collision(self, start_pos, end_pos):
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        for wall in self.walls:
            (w_x1, w_y1), (w_x2, w_y2) = wall
            def ccw(A, B, C):
                return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
            A = (start_x, start_y)
            B = (end_x, end_y)
            C = (w_x1, w_y1)
            D = (w_x2, w_y2)
            if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                return True
        return False

    def _get_movement_vector(self, direction, heading=None):
        if heading is None:
            heading = self.body_heading
        angle = np.radians(heading)
        if direction == 'back':
            angle = np.radians((heading + 180) % 360)
        return np.array([
            self.step_size * np.sin(angle),
            self.step_size * np.cos(angle)
        ])

    def _get_heading_name(self):
        directions = {
            0: "North",
            90: "East", 
            180: "South",
            270: "West"
        }
        return directions.get(self.body_heading, f"{self.body_heading}°")

    def toggle_navigation(self):
        self.return_path = list(self.path_history[::-1])
        print("\n=== RETURN PATH ACTIVATED ===")
        print(f"Steps back to start: {len(self.return_path)}")

    def update_position(self, key):
        if key == ord('n'):
            self.toggle_navigation()
            return
        if key == ord('a'):
            self.body_heading = (self.body_heading - 90) % 360
            print(f"Turned left 90°. Now facing {self._get_heading_name()}")
            return
        elif key == ord('d'):
            self.body_heading = (self.body_heading + 90) % 360
            print(f"Turned right 90°. Now facing {self._get_heading_name()}")
            return
        if key == ord('w'):
            direction = 'forward'
        elif key == ord('s'):
            direction = 'back'
        else:
            return
        new_pos = self.current_position + self._get_movement_vector(direction)
        if self._check_wall_collision(self.current_position, new_pos):
            print(f"Cannot move {direction} - wall in the way!")
            return
        self.current_position = new_pos
        self.path_history.append(tuple(self.current_position))
        print(f"Moved {direction} to X={self.current_position[0]:.1f}, Y={self.current_position[1]:.1f}")

    def _draw_minimap(self, frame):
        minimap = np.zeros((self.minimap_size, self.minimap_size, 3), np.uint8)
        center = self.minimap_size // 2

        for i in range(-10, 11):
            x = center + i * self.minimap_scale
            y = center + i * self.minimap_scale
            cv2.line(minimap, (x, 0), (x, self.minimap_size), (50, 50, 50), 1)
            cv2.line(minimap, (0, y), (self.minimap_size, y), (50, 50, 50), 1)

        for wall in self.walls:
            (x1, y1), (x2, y2) = wall
            pt1 = (center + int(x1 * self.minimap_scale), center - int(y1 * self.minimap_scale))
            pt2 = (center + int(x2 * self.minimap_scale), center - int(y2 * self.minimap_scale))
            cv2.line(minimap, pt1, pt2, (0, 0, 255), 3)

        cv2.circle(minimap, (center, center), 10, (0, 255, 0), -1)

        # Return path visualization
        if self.return_path:
            for idx, pt in enumerate(self.return_path):
                pos = (center + int(pt[0] * self.minimap_scale),
                       center - int(pt[1] * self.minimap_scale))
                color = ((idx * 5) % 256, 255 - (idx * 5) % 256, 150)
                cv2.circle(minimap, pos, 4, color, -1)

        curr_x = center + int(self.current_position[0] * self.minimap_scale)
        curr_y = center - int(self.current_position[1] * self.minimap_scale)
        cv2.circle(minimap, (curr_x, curr_y), 8, (255, 0, 0), -1)
        end_x = curr_x + int(30 * np.sin(np.radians(self.body_heading)))
        end_y = curr_y - int(30 * np.cos(np.radians(self.body_heading)))
        cv2.arrowedLine(minimap, (curr_x, curr_y), (end_x, end_y), (0, 255, 255), 2)

        frame_start = frame.shape[1] - self.minimap_size - 20
        frame[20:20+self.minimap_size, frame_start:frame_start+self.minimap_size] = minimap
        return frame

    def _draw_navigation_panel(self, frame):
        panel_width = frame.shape[1] - 40
        panel = np.zeros((self.panel_height, panel_width, 3), np.uint8)
        cv2.putText(panel, f"Position: X={self.current_position[0]:.1f}m Y={self.current_position[1]:.1f}m",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Heading: {self._get_heading_name()}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Return path (N): {len(self.return_path)} step(s)",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame[20:20+self.panel_height, 20:20+panel_width] = panel
        return frame

    def run(self):
        print("Firefighter AR Thermal Navigation System")
        print("W - Move forward | S - Move back | A/D - Turn left/right")
        print("N - Activate return path | R - Reset | ESC - Quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                frame = np.zeros((720, 1280, 3), np.uint8)
                cv2.putText(frame, "CAMERA NOT AVAILABLE", (400, 360), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                thermal_frame = self._apply_thermal_effect(frame)
            else:
                thermal_frame = self._apply_thermal_effect(frame)
                results = self._detect_objects(thermal_frame)
                detection_frame = results.plot()
                detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_RGB2BGR)
                processed_frame = cv2.addWeighted(thermal_frame, 0.7, detection_frame, 0.5, 0)

            nav_frame = self._draw_navigation_panel(processed_frame)
            nav_frame = self._draw_minimap(nav_frame)

            cv2.imshow(self.window_name, nav_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('r'):
                self.current_position = np.array([0.0, 0.0])
                self.body_heading = 0
                self.path_history = [tuple(self.current_position)]
                self.return_path = []
                print("Reset position and path.")
            else:
                self.update_position(key)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    simulator = FirefighterARSimulator()
    simulator.run()
