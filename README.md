# Dynamic Visual SLAM

A real-time visual SLAM (Simultaneous Localization and Mapping) system designed for dynamic environments using ROS 2. This package implements a feature-based monocular visual SLAM pipeline with bundle adjustment optimization for robust camera pose estimation and 3D landmark mapping.

## Features

- **Real-time Visual Odometry**: ORB feature detection and tracking for camera pose estimation
- **Bundle Adjustment**: Sliding window bundle adjustment using Ceres Solver for pose optimization
- **3D Landmark Mapping**: Persistent landmark storage and visualization
- **RealSense Camera Support**: Built-in support for Intel RealSense depth cameras
- **ROS 2 Integration**: Full ROS 2 compatibility with custom message interfaces
- **Visual Feedback**: RViz visualization with camera poses and landmark markers

## System Architecture

The system consists of two main components:

### Frontend (`frontend` node)
- **Feature Detection**: ORB feature extraction with depth masking
- **Feature Matching**: Robust feature matching with geometric consistency checks
- **Pose Estimation**: PnP RANSAC for camera pose estimation
- **Keyframe Selection**: Adaptive keyframe selection based on tracking quality
- **Coordinate Transformation**: Handles optical to ROS coordinate frame conversion

### Backend (`backend` node)
- **Bundle Adjustment**: Sliding window optimization using Levenberg-Marquardt
- **Landmark Management**: Persistent 3D landmark storage and tracking
- **Map Visualization**: Real-time landmark visualization in RViz
- **Pose Refinement**: Optimized camera pose broadcasting

## Dependencies

### Required ROS 2 Packages
- `rclcpp`
- `sensor_msgs`
- `geometry_msgs`
- `visualization_msgs`
- `tf2_ros`
- `cv_bridge`
- `message_filters`

### External Libraries
- **OpenCV 4.x**: Computer vision operations
- **Eigen3**: Linear algebra computations
- **Ceres Solver**: Non-linear optimization for bundle adjustment

### Hardware
- Intel RealSense camera (tested with D435/D455)

## Installation

1. **Install Dependencies**:
   ```bash
   sudo apt update
   sudo apt install ros-jazzy-realsense2-camera
   sudo apt install libceres-dev
   sudo apt install libeigen3-dev
   ```

2. **Clone and Build**:
   ```bash
   cd ~/ros2_ws/src
   git clone <repository-url> dynamic_visual_slam
   cd ~/ros2_ws
   colcon build --packages-select dynamic_visual_slam dynamic_visual_slam_interfaces
   source install/setup.bash
   ```

## Usage

### Quick Start with RealSense Camera

1. **Launch the complete system**:
   ```bash
   ros2 launch dynamic_visual_slam camera_rviz.launch.xml
   ```

   This will start:
   - RealSense camera driver
   - Frontend node (feature detection and pose estimation)
   - Backend node (bundle adjustment and mapping)
   - RViz visualization

2. **View the results** in RViz:
   - Camera feed with detected features
   - Real-time camera trajectory
   - 3D landmark map visualization

### Manual Launch

If you prefer to launch components separately:

1. **Start RealSense camera**:
   ```bash
   ros2 launch realsense2_camera rs_launch.py depth_module.profile:=1280x720x30 pointcloud.enable:=true align_depth.enable:=true
   ```

2. **Launch SLAM nodes**:
   ```bash
   # Terminal 1 - Frontend
   ros2 run dynamic_visual_slam frontend
   
   # Terminal 2 - Backend  
   ros2 run dynamic_visual_slam backend
   
   # Terminal 3 - Visualization
   rviz2 -d src/dynamic_visual_slam/config/realsense.rviz
   ```

## Configuration

### Camera Parameters
The system automatically reads camera intrinsics from the RealSense camera info topics. For custom cameras, modify the camera info publisher or update the camera matrix in the frontend node.

### Bundle Adjustment Parameters
Key parameters in `bundle_adjustment.hpp`:
- `window_size_`: Number of keyframes in sliding window (default: 10)
- `bundle_adjustment_frequency_`: Optimization frequency (default: every 10 keyframes)
- Convergence thresholds and solver options

### Feature Detection
ORB detector parameters in `frontend.cpp`:
- Number of features: 800
- Depth range: 0.3m - 3.0m
- Feature matching threshold: 50.0 pixels

## Topics

### Subscriptions
- `/camera/camera/color/image_raw` - RGB camera feed
- `/camera/camera/aligned_depth_to_color/image_raw` - Aligned depth images
- `/camera/camera/color/camera_info` - Camera calibration parameters

### Publications
- `/feature_detector/features_image` - Annotated image with detected features
- `/frontend/keyframe` - Keyframe data for bundle adjustment
- `/backend/landmark_markers` - 3D landmark visualization markers
- `/tf` - Camera pose transforms

## Message Interfaces

### Custom Messages
- `Keyframe.msg`: Contains pose, landmarks, and observations
- `Landmark.msg`: 3D landmark position and metadata
- `Observation.msg`: 2D pixel observation of landmarks

## Coordinate Frames

The system maintains several coordinate frames:
- `world`: Global reference frame
- `odom`: Odometry frame (child of world)
- `camera_link`: Camera pose frame
- `camera_color_optical_frame`: RealSense optical frame

## Performance Notes

- **Real-time Performance**: Optimized for 30 FPS operation
- **Memory Usage**: Sliding window approach limits memory growth
- **Accuracy**: Bundle adjustment provides sub-pixel reprojection accuracy
- **Robustness**: Outlier rejection and geometric consistency checks

## Troubleshooting

### Common Issues

1. **No features detected**:
   - Check lighting conditions
   - Verify depth mask parameters
   - Ensure sufficient texture in environment

2. **Poor tracking**:
   - Reduce camera motion speed
   - Check for sufficient feature matches
   - Verify camera calibration

3. **Bundle adjustment convergence**:
   - Check for sufficient parallax between keyframes
   - Verify landmark depth values
   - Adjust solver parameters

### Debug Information

Enable debug logging:
```bash
ros2 run dynamic_visual_slam frontend --ros-args --log-level debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Andrew Kwolek (andrewkwolek2025@u.northwestern.edu)

## Acknowledgments

- Built with ROS 2 and modern C++ practices
- Uses industry-standard libraries (OpenCV, Ceres, Eigen)
- Designed for educational and research purposes
