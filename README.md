# 🚒 Firefighter AR Thermal Navigation Simulator

A real-time Augmented Reality (AR) simulation built with OpenCV and YOLOv8 that helps simulate a firefighter's navigation through smoke-filled environments using thermal imagery, path tracking, and object detection.



## 🔥 Features

- **Thermal Vision Simulation**: Applies an artificial thermal filter over the live camera feed.
- **YOLOv8 Object Detection**: Detects and overlays objects in real-time.
- **Minimap with Path Tracking**: Visualizes the firefighter's movement and walls in the environment.
- **Keyboard Navigation**: Move forward, back, turn left/right, and activate return path.
- **Wall Collision Detection**: Prevents movement through predefined wall boundaries.
- **Return Path Assistance**: Guides user back to the starting point.

## 🎮 Controls

| Key | Action                  |
|-----|-------------------------|
| `W` | Move forward            |
| `S` | Move backward           |
| `A` | Turn left 90°           |
| `D` | Turn right 90°          |
| `N` | Activate return path    |
| `R` | Reset position and path |
| `ESC` | Quit simulator        |

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- NumPy
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## 📦 Installation

1. Clone this repository:

```bash
git clone https://github.com/ahmed-1106/AR-firefighter-helmet.git
cd AR-firefighter-helmet
Install the required packages:
pip install opencv-python numpy ultralytics
Run the simulator:
🧱 Virtual Environment
The simulator uses a predefined wall layout. You can modify the self.walls list to reflect custom obstacle positions.
self.walls = [
    ((2, -2), (2, 3)),
    ((-2, 1), (3, 1)),
    ((-1, -1), (-1, 2)),
    ((1, 2), (1, 4))
]
📷 Output
Real-time video with augmented overlays

Navigation panel and directional arrows

Return path steps visualized on a minimap

📄 License
This project is open source and available under the MIT License.
Created with ❤️ by ahmed sherif lotfy
