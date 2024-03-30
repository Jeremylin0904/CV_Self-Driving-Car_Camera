# Self-driving car: Lane Following ROS Node

This ROS node implements lane following functionality for a robot as part of  self-driving tasks. It subscribes to the camera's raw image topic, processes the images to predict lane lines, and controls the robot's motion based on the predictions.

## Requirements
- Python 3
- ROS (Robot Operating System)
- PyTorch
- OpenCV
- NumPy
- Pillow

## Usage
1. Launch ROS core: `roscore`.
2. Run the lane following node: `lane_following_group5_1.py`[lane_following_group5_1.py].
4. The node will subscribe to the camera's raw image topic and start processing the images to control the robot's motion.

## Video Demonstration
You can watch the lane following functionality in action in this video: [Lane Following Demo](https://youtu.be/46JBglssC7o).



