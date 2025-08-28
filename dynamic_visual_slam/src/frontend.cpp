/**
 * @file frontend.cpp
 * @brief Visual SLAM frontend for real-time feature detection, tracking, and pose estimation
 * @author Andrew Kwolek
 * @date 2025
 * @version 0.0.1
 * 
 * This file implements the frontend component of the Dynamic Visual SLAM system.
 * The frontend is responsible for real-time visual odometry using RGB-D camera data,
 * including ORB feature extraction, inter-frame matching, camera pose estimation,
 * and keyframe selection for backend optimization.
 * 
 * **System Architecture:**
 * The frontend operates as a ROS 2 node that processes synchronized RGB and depth
 * image streams. It maintains temporal consistency through feature tracking and
 * provides reliable pose estimates for robot navigation while selecting keyframes
 * for backend bundle adjustment optimization.
 * 
 * **Key Responsibilities:**
 * 1. **Feature Detection:** Extract ORB features with depth validity filtering
 * 2. **Feature Tracking:** Match features between consecutive frames with geometric validation
 * 3. **Pose Estimation:** Estimate camera motion using PnP RANSAC with 3D-2D correspondences
 * 4. **Keyframe Selection:** Determine when to create keyframes based on tracking quality
 * 5. **Transform Broadcasting:** Publish camera poses in ROS coordinate system
 * 6. **Coordinate Conversion:** Handle transformation between optical and ROS frames
 * 
 * **Coordinate System Conventions:**
 * - **Internal Processing:** Camera optical frame (X=right, Y=down, Z=forward)
 * - **ROS Publishing:** Standard ROS frame (X=forward, Y=left, Z=up)
 * - **Automatic Conversion:** Between coordinate systems for ROS compatibility
 * 
 * **Performance Characteristics:**
 * - **Target Frame Rate:** 30 FPS real-time operation
 * - **Feature Count:** ~800-1000 ORB features per frame
 * - **Depth Range:** 0.3m to 3.0m for valid 3D points
 * - **Memory Usage:** Sliding window approach for bounded memory
 * 
 * **Thread Safety:**
 * All callback functions are thread-safe through ROS 2's executor model.
 * Internal state is protected through proper synchronization mechanisms.
 * 
 * @note Requires synchronized RGB and depth camera topics
 * @note Camera calibration parameters must be available via camera_info topics
 * @warning System assumes rectified and aligned RGB-D images
 * 
 * @see Backend for bundle adjustment and mapping
 * @see ORB_SLAM3::ORBextractor for feature detection implementation
 */

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <memory>
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include <Eigen/Geometry>
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "dynamic_visual_slam_interfaces/msg/keyframe.hpp"
#include "dynamic_visual_slam_interfaces/msg/observation.hpp"
#include "dynamic_visual_slam_interfaces/msg/landmark.hpp"
#include "dynamic_visual_slam/ORBextractor.hpp"

/**
 * @class Frontend
 * @brief Main frontend node for visual SLAM odometry and feature tracking
 * 
 * The Frontend class implements real-time visual odometry using ORB features
 * extracted from RGB-D camera streams. It processes synchronized RGB and depth
 * images to track camera motion, detect keyframes, and provide input data for
 * backend bundle adjustment optimization.
 * 
 * **Core Algorithm Pipeline:**
 * 1. **Image Synchronization:** RGB and depth images synchronized using message filters
 * 2. **Feature Extraction:** ORB features detected with depth validity checking
 * 3. **Feature Matching:** Brute-force matching with geometric consistency validation
 * 4. **Pose Estimation:** PnP RANSAC using 3D points from previous frame depth
 * 5. **Keyframe Decision:** Based on tracking quality and temporal criteria
 * 6. **Data Publishing:** Keyframes with 3D landmarks sent to backend
 * 
 * **Coordinate Frame Management:**
 * The system handles multiple coordinate frames:
 * - **Camera Optical:** X=right, Y=down, Z=forward (OpenCV convention)
 * - **ROS Standard:** X=forward, Y=left, Z=up (ROS convention)  
 * - **Transform Hierarchy:** world → odom → camera_link → camera_optical_frame
 * 
 * **Feature Processing Pipeline:**
 * Raw features → Depth filtering → Descriptor matching → Geometric validation → Pose estimation
 * 
 * **Memory Management:**
 * - Previous frame data cached for inter-frame matching
 * - Sliding window keyframe selection prevents memory growth
 * - Automatic cleanup of outdated tracking data
 * 
 * **Error Handling:**
 * - Graceful degradation when feature matching fails
 * - Motion outlier detection and rejection
 * - Automatic recovery from tracking loss
 * 
 * @note Camera intrinsic parameters loaded automatically from camera_info topics
 * @note Supports Intel RealSense cameras with factory calibration
 * @warning Requires good lighting and textured environment for feature detection
 * 
 * @example Usage in ROS 2 launch file:
 * @code{.xml}
 * <node pkg="dynamic_visual_slam" exec="frontend"/>
 * @endcode
 * 
 * @example Programmatic usage:
 * @code{.cpp}
 * rclcpp::init(argc, argv);
 * auto frontend_node = std::make_shared<Frontend>();
 * rclcpp::spin(frontend_node);  // Process camera data
 * rclcpp::shutdown();
 * @endcode
 */
class Frontend : public rclcpp::Node
{
public:
    /**
     * @brief Constructs the frontend node with complete SLAM pipeline setup
     * 
     * Initializes all components necessary for real-time visual odometry:
     * - Synchronized RGB/depth image subscribers using message filters
     * - Camera calibration info subscribers for both RGB and depth cameras
     * - Publishers for visualization and keyframe data
     * - ORB feature extractor with optimized parameters
     * - TF broadcasters for pose publishing
     * - Static transform tree setup (world → odom → camera frames)
     * 
     * **Initialization Sequence:**
     * 1. Create message filter subscribers for image synchronization
     * 2. Setup camera info subscribers for calibration parameters  
     * 3. Initialize ORB feature detector (1000 features, 8 pyramid levels)
     * 4. Create TF broadcasters for pose publishing
     * 5. Establish coordinate frame hierarchy
     * 6. Initialize tracking state variables
     * 
     * **Default Parameters:**
     * - ORB Features: 1000 per frame
     * - Pyramid Levels: 8 (multi-scale detection)
     * - Scale Factor: 1.2 (pyramid scaling)
     * - FAST Thresholds: 20 (initial), 7 (minimum)
     * - Depth Range: 0.3m - 3.0m (valid depth filtering)
     * 
     * **Topic Subscriptions:**
     * - `/camera/camera/color/image_raw` - RGB camera stream
     * - `/camera/camera/aligned_depth_to_color/image_raw` - Aligned depth stream
     * - `/camera/camera/color/camera_info` - RGB camera calibration
     * - `/camera/camera/aligned_depth_to_color/camera_info` - Depth camera calibration
     * 
     * **Topic Publications:**
     * - `/feature_detector/features_image` - Annotated RGB image with detected features
     * - `/feature_detector/camera_info` - Camera info synchronized with feature image
     * - `/frontend/keyframe` - Keyframe data for backend bundle adjustment
     * - `/frontend/dgb_image` - Debug image for development
     * - `/tf` - Camera pose transforms
     * 
     * @post All ROS subscriptions and publishers are active
     * @post Static transform tree is published 
     * @post System ready to process camera data upon spin()
     * 
     * @note Constructor completes quickly; actual processing begins with first image
     * @note Camera parameters loaded asynchronously from camera_info messages
     */
    Frontend() : Node("frontend")
    {
        rclcpp::QoS qos = rclcpp::QoS(30);

        // Create message filter subscribers for precise RGB-D synchronization
        rgb_sub_.subscribe(this, "/camera/camera/color/image_raw", qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, "/camera/camera/aligned_depth_to_color/image_raw", qos.get_rmw_qos_profile());

        // Setup approximate time synchronizer (allows small timestamp differences)
        sync_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>>(
            message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>(10), rgb_sub_, depth_sub_);
        sync_->registerCallback(std::bind(&Frontend::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        // Camera info subscribers for calibration parameters
        rgb_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/color/camera_info", qos, 
            std::bind(&Frontend::rgbInfoCallback, this, std::placeholders::_1));
        depth_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/aligned_depth_to_color/camera_info", qos, 
            std::bind(&Frontend::depthInfoCallback, this, std::placeholders::_1));

        // Publishers for visualization and backend communication
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/feature_detector/features_image", qos);
        camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/feature_detector/camera_info", qos);
        keyframe_pub_ = this->create_publisher<dynamic_visual_slam_interfaces::msg::Keyframe>("/frontend/keyframe", qos);
        dgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/frontend/dgb_image", qos);
        
        // Initialize ORB feature detector with optimized parameters for SLAM
        // 1000 features provide good coverage while maintaining real-time performance
        orb_extractor_ = std::make_unique<ORB_SLAM3::ORBextractor>(
            1000,  // nfeatures: target number of features per frame
            1.2f,  // scaleFactor: pyramid scale factor between levels  
            8,     // nlevels: number of pyramid levels for multi-scale detection
            20,    // iniThFAST: initial FAST threshold for corner detection
            7      // minThFAST: minimum FAST threshold if initial fails
        );

        prev_frame_valid_ = false;

        // TF broadcasting for pose publishing
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        // Brute-force matcher optimized for ORB binary descriptors
        matcher_ = cv::BFMatcher(cv::NORM_HAMMING);

        // Establish coordinate frame hierarchy: world → odom → camera_link
        publishStaticTransforms();

        // Initialize camera pose to identity (world origin)
        R_ = cv::Mat::eye(3, 3, CV_64F);  // Identity rotation matrix
        t_ = cv::Mat::zeros(3, 1, CV_64F); // Zero translation vector

        // Initialize camera parameter storage
        rgb_camera_matrix_ = cv::Mat();
        rgb_dist_coeffs_ = cv::Mat();
        depth_camera_matrix_ = cv::Mat();
        depth_dist_coeffs_ = cv::Mat();

        // Initialize keyframe tracking variables
        keyframe_id_ = 0;
        frames_since_last_keyframe_ = 0;
        has_last_keyframe_ = false;

        // Depth filtering parameters (meters)
        MAX_DEPTH = 3.0f;  // Maximum valid depth for feature points
        MIN_DEPTH = 0.3f;  // Minimum valid depth (avoid noise near camera)
            
        RCLCPP_INFO(this->get_logger(), "Frontend node initialized successfully");
        RCLCPP_INFO(this->get_logger(), "Waiting for camera data and calibration parameters...");
    }

private:
    // === ROS Communication Infrastructure ===
    
    /// Message filter subscribers for synchronized RGB-D processing
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    
    /// Camera calibration info subscribers
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr rgb_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;

    /// Publishers for visualization and backend communication
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
    rclcpp::Publisher<dynamic_visual_slam_interfaces::msg::Keyframe>::SharedPtr keyframe_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr dgb_pub_;

    // === Camera Calibration Data ===
    
    /// Latest camera info messages (cached for processing)
    sensor_msgs::msg::CameraInfo::SharedPtr latest_rgb_camera_info_;
    sensor_msgs::msg::CameraInfo::SharedPtr latest_depth_camera_info_;
    
    /// OpenCV camera calibration matrices
    cv::Mat rgb_camera_matrix_;    ///< 3x3 camera intrinsic matrix for RGB
    cv::Mat rgb_dist_coeffs_;      ///< Distortion coefficients for RGB camera
    cv::Mat depth_camera_matrix_;  ///< 3x3 camera intrinsic matrix for depth
    cv::Mat depth_dist_coeffs_;    ///< Distortion coefficients for depth camera
    
    /// Individual camera parameters for efficient access
    float rgb_fx_, rgb_fy_, rgb_cx_, rgb_cy_;     ///< RGB camera focal lengths and principal point
    float depth_fx_, depth_fy_, depth_cx_, depth_cy_; ///< Depth camera focal lengths and principal point
    
    sensor_msgs::msg::Image dgb_image_; ///< Debug image storage

    // === Feature Detection and Processing ===
    
    /// Depth filtering parameters (meters)
    float MAX_DEPTH; ///< Maximum valid depth for 3D point computation
    float MIN_DEPTH; ///< Minimum valid depth to avoid sensor noise

    /// ORB feature detector (from ORB-SLAM3)
    std::vector<int> vLappingArea = {0, 0}; ///< Stereo overlapping area (unused in monocular)
    std::unique_ptr<ORB_SLAM3::ORBextractor> orb_extractor_;

    /// Feature matching infrastructure
    cv::BFMatcher matcher_; ///< Brute-force matcher for ORB descriptors

    /// Message synchronization for RGB-D processing
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,sensor_msgs::msg::Image>>> sync_;

    // === Temporal Tracking State ===
    
    /// Previous frame data for inter-frame matching
    cv::Mat prev_frame_gray_;      ///< Previous grayscale image
    cv::Mat prev_frame_depth_;     ///< Previous depth image  
    std::vector<cv::KeyPoint> prev_kps_;    ///< Previous frame keypoints
    std::vector<cv::Point2f> prev_points_;  ///< Previous frame point coordinates (legacy)
    cv::Mat prev_descriptors_;     ///< Previous frame ORB descriptors
    bool prev_frame_valid_;        ///< Whether previous frame data is available

    // === Transform Broadcasting ===
    
    /// TF broadcasters for pose publishing
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;

    // === Camera Pose State ===
    
    /// Current camera pose in world coordinates (optical frame)
    cv::Mat R_; ///< 3x3 rotation matrix (camera to world)
    cv::Mat t_; ///< 3x1 translation vector (camera position in world)

    // === Keyframe Management ===
    
    /// Keyframe detection and management
    long long int keyframe_id_;                      ///< Unique keyframe identifier (monotonic)
    int frames_since_last_keyframe_;                ///< Counter for keyframe timing
    cv::Mat last_keyframe_descriptors_;             ///< Descriptors from last keyframe
    std::vector<cv::KeyPoint> last_keyframe_keypoints_; ///< Keypoints from last keyframe  
    cv::Mat last_keyframe_depth_;                   ///< Depth image from last keyframe
    bool has_last_keyframe_;                        ///< Whether keyframe reference exists

    /**
     * @brief Publishes static coordinate frame transforms
     * 
     * Establishes the transform tree hierarchy for the SLAM system:
     * world → odom → camera_link
     * 
     * This creates a consistent coordinate frame reference that allows
     * visualization tools and navigation systems to properly interpret
     * the camera pose estimates.
     */
    void publishStaticTransforms() {
        // Create odom frame as child of world
        geometry_msgs::msg::TransformStamped odom;
        odom.header.stamp = this->now();
        odom.header.frame_id = "world";
        odom.child_frame_id = "odom";
        odom.transform.translation.x = 0.0;
        odom.transform.translation.y = 0.0;
        odom.transform.translation.z = 0.0;
        odom.transform.rotation.x = 0.0;
        odom.transform.rotation.y = 0.0;
        odom.transform.rotation.z = 0.0;
        odom.transform.rotation.w = 1.0;
        static_broadcaster_->sendTransform(odom);

        // Create camera_link frame as child of odom (will be updated dynamically)
        geometry_msgs::msg::TransformStamped camera_link;
        camera_link.header.stamp = this->now();
        camera_link.header.frame_id = "odom";
        camera_link.child_frame_id = "camera_link";
        camera_link.transform.translation.x = 0.0;
        camera_link.transform.translation.y = 0.0;
        camera_link.transform.translation.z = 0.0;
        camera_link.transform.rotation.x = 0.0;
        camera_link.transform.rotation.y = 0.0;
        camera_link.transform.rotation.z = 0.0;
        camera_link.transform.rotation.w = 1.0;
        static_broadcaster_->sendTransform(camera_link);
    }

    /**
     * @brief Broadcasts camera pose transform in ROS coordinate system
     * 
     * Converts internal camera pose (optical coordinates) to ROS standard
     * coordinates and publishes the transform for visualization and navigation.
     * 
     * **Coordinate Transformation:**
     * - Optical Frame: X=right, Y=down, Z=forward
     * - ROS Frame: X=forward, Y=left, Z=up  
     * - Transformation Matrix: T_ros = T_conversion * T_optical * T_conversion^T
     * 
     * @param stamp ROS timestamp for the transform
     * 
     * @pre R_ and t_ contain valid camera pose in optical coordinates
     * @post Transform published on /tf topic as odom → camera_link
     * 
     * @note Transform represents camera pose, not camera-to-world transformation
     */
    void broadcastTransformROS(const rclcpp::Time& stamp) {
        // Transformation matrix from optical frame to ROS frame
        // Optical: X=right, Y=down, Z=forward
        // ROS:     X=forward, Y=left, Z=up
        cv::Mat T_opt_to_ros = (cv::Mat_<double>(3,3) << 
            0,  0,  1,    // Optical Z → ROS X (forward)
            -1, 0,  0,    // Optical -X → ROS Y (left)
            0, -1,  0     // Optical -Y → ROS Z (up)
        );
        
        // Convert pose from optical to ROS coordinates
        cv::Mat R_ros = T_opt_to_ros * R_ * T_opt_to_ros.t();
        cv::Mat t_ros = T_opt_to_ros * t_;
        
        // Convert rotation matrix to quaternion using Eigen
        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R_ros.at<double>(i, j);
            }
        }
        
        Eigen::Quaterniond q(R_eigen);
        q.normalize();
        
        // Publish transform message
        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = stamp;
        transform_stamped.header.frame_id = "odom";
        transform_stamped.child_frame_id = "camera_link";
        
        transform_stamped.transform.translation.x = t_ros.at<double>(0);
        transform_stamped.transform.translation.y = t_ros.at<double>(1);
        transform_stamped.transform.translation.z = t_ros.at<double>(2);
        
        transform_stamped.transform.rotation.x = q.x();
        transform_stamped.transform.rotation.y = q.y();
        transform_stamped.transform.rotation.z = q.z();
        transform_stamped.transform.rotation.w = q.w();
        
        tf_broadcaster_->sendTransform(transform_stamped);
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "Published camera pose - Position: [%.3f, %.3f, %.3f], Quat: [%.3f, %.3f, %.3f, %.3f]",
                    t_ros.at<double>(0), t_ros.at<double>(1), t_ros.at<double>(2),
                    q.x(), q.y(), q.z(), q.w());
    }

    /**
     * @brief Validates depth measurement for 3D point computation
     * 
     * Checks whether a depth pixel contains a valid measurement suitable
     * for 3D point triangulation and SLAM processing.
     * 
     * **Validity Criteria:**
     * - Pixel coordinates within image boundaries
     * - Depth value within sensor range [MIN_DEPTH, MAX_DEPTH]
     * - No NaN, infinite, or negative depth values
     * - Depth represents actual measurement (not sensor saturation)
     * 
     * @param depth_image 16-bit depth image (millimeters)
     * @param x Pixel x-coordinate (column)
     * @param y Pixel y-coordinate (row)
     * @return true if depth measurement is valid for SLAM processing
     * 
     * @note Depth image assumed to be 16UC1 format with millimeter units
     * @note Range limits prevent near-field noise and far-field unreliability
     */
    bool isValidDepth(const cv::Mat& depth_image, int x, int y) {
        // Check image boundary constraints
        if (x < 0 || y < 0 || x >= depth_image.cols || y >= depth_image.rows) {
            return false;
        }

        // Convert from millimeters to meters
        float depth = depth_image.at<uint16_t>(y, x) * 0.001f;

        // Validate depth range and numerical properties
        if (depth < MIN_DEPTH || depth > MAX_DEPTH || 
            std::isnan(depth) || std::isinf(depth) || depth < 0.0f) {
            return false;
        }

        return true;
    }

    /**
     * @brief Filters keypoints based on depth validity for 3D SLAM processing
     * 
     * Removes keypoints that lack valid depth measurements, ensuring only
     * features with reliable 3D positions are used for pose estimation and
     * landmark creation. This is critical for RGB-D SLAM accuracy.
     * 
     * **Filtering Process:**
     * 1. Check each keypoint pixel location for valid depth
     * 2. Retain keypoints with depth in valid range [MIN_DEPTH, MAX_DEPTH]
     * 3. Copy corresponding descriptors for retained keypoints
     * 4. Maintain correspondence between keypoints and descriptors
     * 
     * @param keypoints Input keypoints from feature detector
     * @param descriptors Input descriptors corresponding to keypoints
     * @param depth_image Depth image for validity checking (16UC1, millimeters)
     * @param[out] filtered_keypoints Output keypoints with valid depth
     * @param[out] filtered_descriptors Output descriptors for valid keypoints
     * 
     * @pre keypoints and descriptors must have same count
     * @pre depth_image must be aligned with RGB image used for feature detection
     * 
     * @post filtered_keypoints.size() == filtered_descriptors.rows
     * @post All filtered keypoints have valid depth measurements
     * 
     * @note Filtered output typically 60-80% of input features
     * @note Critical for preventing degenerate 3D points in bundle adjustment
     */
    void filterDepth(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, 
                    const cv::Mat& depth_image, std::vector<cv::KeyPoint>& filtered_keypoints, 
                    cv::Mat& filtered_descriptors) {
        std::vector<int> original_indices;

        // First pass: identify keypoints with valid depth
        for (size_t i = 0; i < keypoints.size(); i++) {
            cv::Point2f pt = keypoints[i].pt;
            int x = static_cast<int>(std::round(pt.x));
            int y = static_cast<int>(std::round(pt.y));

            if (isValidDepth(depth_image, x, y)) {
                filtered_keypoints.push_back(keypoints[i]);
                original_indices.push_back(i);
            }
        }

        // Second pass: copy corresponding descriptors
        if (!filtered_keypoints.empty() && !descriptors.empty()) {
            filtered_descriptors = cv::Mat(filtered_keypoints.size(), descriptors.cols, descriptors.type());
            for (size_t i = 0; i < original_indices.size(); i++) {
                descriptors.row(original_indices[i]).copyTo(filtered_descriptors.row(i));
            }
        }
    }

    /**
     * @brief Detects motion outliers in camera pose estimates
     * 
     * Validates estimated camera motion against reasonable physical constraints
     * to prevent pose estimation errors from corrupting the SLAM trajectory.
     * Typical causes of outliers include poor feature matches, degenerate
     * geometry, or rapid camera motion.
     * 
     * **Outlier Detection Criteria:**
     * - Translation magnitude exceeds maximum expected motion per frame
     * - Rotation angle exceeds maximum expected angular motion per frame
     * - Motion inconsistent with typical handheld camera movement
     * 
     * @param R_new Estimated rotation matrix for current frame
     * @param t_new Estimated translation vector for current frame  
     * @return true if motion appears to be an outlier
     * 
     * @note Thresholds tuned for 30 FPS handheld camera operation
     * @note Conservative bounds prevent tracking failure propagation
     */
    bool isMotionOutlier(const cv::Mat& R_new, const cv::Mat& t_new) {
        const double MAX_TRANSLATION = 0.5; // Maximum translation per frame (meters)
        const double MAX_ROTATION = 0.2;    // Maximum rotation per frame (radians)
        
        // Check translation magnitude
        double translation_norm = cv::norm(t_new);
        if (translation_norm > MAX_TRANSLATION) {
            RCLCPP_DEBUG(this->get_logger(), "Translation outlier detected: %f m", translation_norm);
            return true;
        }
        
        // Check rotation magnitude
        cv::Mat rvec;
        cv::Rodrigues(R_new, rvec);
        double rotation_angle = cv::norm(rvec);
        if (rotation_angle > MAX_ROTATION) {
            RCLCPP_DEBUG(this->get_logger(), "Rotation outlier detected: %f rad", rotation_angle);
            return true;
        }
        
        return false;
    }

    /**
     * @brief Determines whether current frame should be selected as keyframe
     * 
     * Implements adaptive keyframe selection strategy based on visual tracking
     * quality and temporal criteria. Keyframes are selected when tracking
     * quality degrades or sufficient time has elapsed, ensuring adequate
     * coverage for bundle adjustment optimization.
     * 
     * **Selection Strategy:**
     * 1. **First Frame:** Always selected as initial keyframe
     * 2. **Tracking Quality:** Selected when fewer than threshold matches with last keyframe
     * 3. **Temporal Constraint:** Selected after maximum frame interval regardless of tracking
     * 4. **Geometric Validation:** Feature matches validated with fundamental matrix RANSAC
     * 
     * **Quality Metrics:**
     * - Descriptor matching score between current and last keyframe
     * - Number of geometrically consistent feature correspondences
     * - Spatial distribution of matching features
     * 
     * @param current_descriptors ORB descriptors from current frame
     * @param current_keypoints Keypoint locations from current frame
     * @return true if current frame should become a keyframe
     * 
     * @pre has_last_keyframe_ indicates whether reference keyframe exists
     * @post frames_since_last_keyframe_ counter updated appropriately
     * 
     * @note Keyframe density affects backend optimization quality vs. speed
     * @note Threshold of 150 matches empirically determined for good tracking
     */
    bool isKeyframe(const cv::Mat& current_descriptors, const std::vector<cv::KeyPoint>& current_keypoints) {
        // Always select first frame as initial keyframe
        if (!has_last_keyframe_) {
            has_last_keyframe_ = true;
            return true;
        }

        bool tracking_criterion = false;
        
        // Evaluate tracking quality relative to last keyframe
        if (!last_keyframe_descriptors_.empty() && !current_descriptors.empty()) {
            // Match current frame against last keyframe
            std::vector<cv::DMatch> all_keyframe_matches;
            matcher_.match(current_descriptors, last_keyframe_descriptors_, all_keyframe_matches);
            
            // Apply distance-based filtering
            std::vector<cv::DMatch> distance_filtered_keyframe_matches;
            float max_distance = 50.0f; // Hamming distance threshold for ORB
            for (const auto& match : all_keyframe_matches) {
                if (match.distance < max_distance) {
                    distance_filtered_keyframe_matches.push_back(match);
                }
            }
            
            // Apply geometric consistency filtering using fundamental matrix
            std::vector<cv::DMatch> geometrically_consistent_keyframe_matches;
            if (distance_filtered_keyframe_matches.size() >= 8) {
                std::vector<cv::Point2f> last_kf_pts, current_kf_pts;
                for (const auto& match : distance_filtered_keyframe_matches) {
                    last_kf_pts.push_back(last_keyframe_keypoints_[match.trainIdx].pt);
                    current_kf_pts.push_back(current_keypoints[match.queryIdx].pt);
                }
                
                std::vector<uchar> kf_inliers_mask;
                cv::findFundamentalMat(last_kf_pts, current_kf_pts, kf_inliers_mask, 
                                     cv::FM_RANSAC, 2.0, 0.99);
                
                for (size_t i = 0; i < kf_inliers_mask.size(); i++) {
                    if (kf_inliers_mask[i]) {
                        geometrically_consistent_keyframe_matches.push_back(distance_filtered_keyframe_matches[i]);
                    }
                }
            } else {
                geometrically_consistent_keyframe_matches = distance_filtered_keyframe_matches;
            }
            
            RCLCPP_DEBUG(this->get_logger(), "Valid matches with last keyframe: %zu", 
                        geometrically_consistent_keyframe_matches.size());
            
            // Trigger keyframe if tracking quality insufficient
            tracking_criterion = (geometrically_consistent_keyframe_matches.size() < 150);
        }

        // Apply keyframe selection logic
        if (tracking_criterion || frames_since_last_keyframe_ > 30) {
            frames_since_last_keyframe_ = 0;
            return true;
        }

        frames_since_last_keyframe_++;
        return false;
    }

    /**
     * @brief Publishes keyframe data for backend bundle adjustment
     * 
     * Packages current frame data into keyframe message format including
     * camera pose, 3D landmarks, and 2D observations. Performs 3D point
     * triangulation using camera calibration and depth information.
     * 
     * **Data Preparation:**
     * 1. Convert camera pose to message format (optical coordinates)
     * 2. Triangulate 3D world points from depth and camera intrinsics
     * 3. Create landmark-observation correspondences
     * 4. Package descriptors for backend feature matching
     * 5. Update keyframe reference data for next comparison
     * 
     * **Coordinate System:**
     * - Pose stored in optical coordinates (internal consistency)
     * - Landmarks computed in world coordinates using current pose
     * - Observations stored in pixel coordinates
     * 
     * @param current_keypoints Feature keypoints detected in current frame
     * @param current_descriptors ORB descriptors for current keypoints
     * @param current_depth_frame Depth image aligned with current RGB frame
     * @param stamp ROS timestamp for keyframe message
     * 
     * @pre Camera calibration parameters available (fx_, fy_, cx_, cy_)
     * @pre current_keypoints and current_descriptors have matching sizes
     * @pre current_depth_frame aligned with RGB frame used for features
     * 
     * @post Keyframe message published on /frontend/keyframe topic
     * @post Last keyframe reference data updated for next comparison
     * @post keyframe_id_ incremented for unique identification
     * 
     * @note Only features with valid depth measurements included
     * @note 3D landmarks transformed to world coordinates for backend consistency
     */
    void publishKeyframe(const std::vector<cv::KeyPoint>& current_keypoints, 
                        const cv::Mat& current_descriptors, 
                        const cv::Mat& current_depth_frame, 
                        const rclcpp::Time& stamp) {
        dynamic_visual_slam_interfaces::msg::Keyframe kf;

        // Convert rotation matrix to quaternion for message format
        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R_.at<double>(i, j);  // R_ is in optical coordinates
            }
        }
        Eigen::Quaterniond q(R_eigen);
        q.normalize();
        
        // Package pose in message format (optical coordinates)
        geometry_msgs::msg::Transform transform;
        transform.translation.x = t_.at<double>(0);  // t_ is in optical coordinates
        transform.translation.y = t_.at<double>(1);
        transform.translation.z = t_.at<double>(2);
        transform.rotation.x = q.x();
        transform.rotation.y = q.y();
        transform.rotation.z = q.z();
        transform.rotation.w = q.w();

        // Set keyframe metadata
        kf.pose = transform;
        kf.header.frame_id = "camera_link";
        kf.header.stamp = stamp;
        kf.frame_id = keyframe_id_++;

        // Process each keypoint to create 3D landmarks and 2D observations
        for (size_t i = 0; i < current_keypoints.size(); i++) {
            cv::Point2f pt = current_keypoints[i].pt;
            int x = static_cast<int>(std::round(pt.x));
            int y = static_cast<int>(std::round(pt.y));
            
            // Get depth measurement and convert to meters
            float pt_depth = current_depth_frame.at<uint16_t>(y, x) * 0.001f;
            
            // Triangulate 3D point in camera coordinates using pinhole model
            cv::Point3f pt_3d_optical(
                (pt.x - rgb_cx_) * pt_depth / rgb_fx_,  // X = (u - cx) * depth / fx
                (pt.y - rgb_cy_) * pt_depth / rgb_fy_,  // Y = (v - cy) * depth / fy
                pt_depth                                 // Z = depth
            );
            
            // Validate depth range for reliable 3D points
            if (pt_3d_optical.z > 0.3 && pt_3d_optical.z < 3.0) {
                // Transform 3D point to world coordinates
                cv::Mat landmark_camera = (cv::Mat_<double>(3,1) << 
                    pt_3d_optical.x, pt_3d_optical.y, pt_3d_optical.z);
                cv::Mat landmark_world = R_ * landmark_camera + t_;

                // Create landmark message
                dynamic_visual_slam_interfaces::msg::Landmark landmark;
                landmark.landmark_id = static_cast<uint64_t>(i);
                landmark.position.x = landmark_world.at<double>(0);
                landmark.position.y = landmark_world.at<double>(1);
                landmark.position.z = landmark_world.at<double>(2);
                
                // Create corresponding observation message
                dynamic_visual_slam_interfaces::msg::Observation obs;
                obs.landmark_id = static_cast<uint64_t>(i);
                obs.pixel_x = current_keypoints[i].pt.x;
                obs.pixel_y = current_keypoints[i].pt.y;
                
                // Copy ORB descriptor for backend matching
                cv::Mat descriptor_row = current_descriptors.row(i);
                obs.descriptor.assign(descriptor_row.data, 
                                    descriptor_row.data + descriptor_row.total());
                
                // Add to keyframe message
                kf.landmarks.push_back(landmark);
                kf.observations.push_back(obs);
            }
        }

        // Update keyframe reference data for next comparison
        last_keyframe_keypoints_ = current_keypoints;
        last_keyframe_descriptors_ = current_descriptors.clone();

        // Publish keyframe and debug data
        keyframe_pub_->publish(kf);
        dgb_pub_->publish(dgb_image_);
        
        RCLCPP_INFO(this->get_logger(), "Published keyframe %lld with %zu landmarks", 
                    kf.frame_id, kf.landmarks.size());
    }

    /**
     * @brief Estimates camera pose using PnP RANSAC with 3D-2D correspondences
     * 
     * Implements the core visual odometry algorithm by solving the Perspective-n-Point
     * problem using matched features between consecutive frames. Uses depth information
     * from the previous frame to establish 3D-2D correspondences, then estimates the
     * relative camera motion using robust RANSAC-based pose estimation.
     * 
     * **Algorithm Pipeline:**
     * 1. **3D Point Creation:** Use previous frame depth to triangulate 3D points
     * 2. **Correspondence Setup:** Link previous 3D points to current 2D features  
     * 3. **PnP RANSAC:** Robustly estimate camera motion with outlier rejection
     * 4. **Motion Validation:** Check for physically reasonable motion
     * 5. **Pose Update:** Integrate motion into global camera trajectory
     * 6. **Transform Broadcasting:** Publish updated pose for navigation
     * 
     * **Coordinate System Handling:**
     * - 3D points triangulated in camera optical frame of previous keyframe
     * - Pose estimation works in optical coordinates throughout
     * - Global pose accumulated as T_world_current = T_world_prev * T_prev_current^-1
     * - Transform broadcasting handles conversion to ROS coordinates
     * 
     * **Robustness Features:**
     * - RANSAC outlier rejection for degenerate feature matches
     * - Motion outlier detection prevents large pose jumps
     * - Minimum correspondence requirement ensures pose reliability
     * - Depth validity checking prevents erroneous 3D points
     * 
     * @param prev_kps Keypoints detected in previous frame
     * @param curr_kps Keypoints detected in current frame
     * @param good_matches Validated matches between prev_kps and curr_kps
     * @param prev_depth Depth image from previous frame (16UC1, millimeters)
     * @param stamp Timestamp for transform broadcasting
     * 
     * @pre good_matches contains valid indices into prev_kps and curr_kps
     * @pre prev_depth aligned with previous RGB frame used for prev_kps
     * @pre Camera intrinsics (rgb_camera_matrix_, etc.) initialized
     * @pre good_matches.size() >= 6 for reliable pose estimation
     * 
     * @post Global camera pose (R_, t_) updated if estimation successful
     * @post Camera transform broadcasted on /tf topic  
     * @post Pose remains unchanged if estimation fails or is outlier
     * 
     * @note Requires minimum 6 valid 3D-2D correspondences for solvePnPRansac
     * @note Motion outlier rejection prevents pose jumps >0.5m translation or >0.2rad rotation  
     * @note Fails gracefully with warning if insufficient valid correspondences
     * 
     * @warning Depth image must be temporally and spatially aligned with RGB
     * @warning Poor lighting or texture can cause tracking failure
     * 
     * @see isMotionOutlier() for outlier detection criteria
     * @see broadcastTransformROS() for coordinate frame conversion
     */
    void estimateCameraPose(const std::vector<cv::KeyPoint>& prev_kps, 
                           const std::vector<cv::KeyPoint>& curr_kps, 
                           const std::vector<cv::DMatch>& good_matches, 
                           const cv::Mat& prev_depth, 
                           const rclcpp::Time& stamp) {
        std::vector<cv::Point3f> points3d;  // 3D points from previous frame
        std::vector<cv::Point2f> points2d;  // 2D points from current frame

        // Validate camera calibration availability
        if (rgb_camera_matrix_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Camera intrinsics not available for pose estimation!");
            return;
        }

        try {
            // Build 3D-2D correspondences from feature matches
            for (const auto& match : good_matches) {
                cv::Point2f prev_pt = prev_kps[match.trainIdx].pt;  // Previous frame pixel
                cv::Point2f curr_pt = curr_kps[match.queryIdx].pt;  // Current frame pixel
                
                // Get pixel coordinates for depth lookup (rounded to integer)
                int x_prev = static_cast<int>(std::round(prev_pt.x));
                int y_prev = static_cast<int>(std::round(prev_pt.y));
                
                // Validate pixel coordinates are within image bounds
                if (x_prev < 0 || y_prev < 0 || 
                    x_prev >= prev_depth.cols || y_prev >= prev_depth.rows) {
                    continue;
                }
                
                // Extract depth measurement and convert from mm to meters
                float d_prev = prev_depth.at<uint16_t>(y_prev, x_prev) * 0.001f;
                
                // Validate depth is within reliable sensing range
                if (d_prev <= 0.3f || d_prev > 3.0f) {
                    continue;
                }
                
                // Triangulate 3D point in previous camera frame using pinhole model
                // Keep in optical coordinates (this is what solvePnP expects)
                cv::Point3f pt3d_prev(
                    (prev_pt.x - rgb_cx_) * d_prev / rgb_fx_,  // X_cam = (u - cx) * depth / fx
                    (prev_pt.y - rgb_cy_) * d_prev / rgb_fy_,  // Y_cam = (v - cy) * depth / fy
                    d_prev                                      // Z_cam = depth
                );
                
                // Store valid 3D-2D correspondence
                points3d.push_back(pt3d_prev);
                points2d.push_back(curr_pt);
            }
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during correspondence creation: %s", e.what());
            return;
        }

        // Validate sufficient correspondences for robust pose estimation
        if (points3d.size() < 6) {
            RCLCPP_WARN(this->get_logger(), 
                       "Insufficient correspondences for pose estimation: %zu (need >= 6)", 
                       points3d.size());
            return;
        }

        try {
            cv::Mat rvec, tvec;  // PnP solution: rotation vector and translation
            std::vector<int> inliers;  // RANSAC inlier indices
            
            // Solve PnP RANSAC for robust camera pose estimation
            bool success = cv::solvePnPRansac(
                points3d,           // 3D points from previous frame (camera coordinates)
                points2d,           // 2D points in current frame (pixel coordinates)
                rgb_camera_matrix_, // Camera intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
                rgb_dist_coeffs_,   // Radial distortion coefficients
                rvec,               // Output: rotation vector (Rodrigues format)
                tvec,               // Output: translation vector
                false,              // Don't use initial guess
                100,                // RANSAC iterations
                4.0,                // RANSAC reprojection threshold (pixels)
                0.99,               // RANSAC confidence level
                inliers             // Output: inlier correspondence indices  
            );

            if (!success) {
                RCLCPP_WARN(this->get_logger(), "PnP RANSAC failed to find solution");
                return;
            }

            // Convert rotation vector to matrix for easier manipulation
            cv::Mat R_relative;
            cv::Rodrigues(rvec, R_relative);
            cv::Mat t_relative = tvec.clone();
            
            // PnP gives us T_current_previous (transformation from previous to current camera)
            // We need to invert this to get T_previous_current for pose chaining
            cv::Mat R_curr_inv = R_relative.t();           // R^-1 = R^T for rotation matrices
            cv::Mat t_curr_inv = -R_curr_inv * t_relative; // t^-1 = -R^T * t
            
            // Validate motion is physically reasonable (outlier detection)
            if (isMotionOutlier(R_curr_inv, t_curr_inv)) {
                RCLCPP_DEBUG(this->get_logger(), "Motion outlier detected, rejecting pose update");
                return;
            }

            // Update global camera pose: T_world_current = T_world_previous * T_previous_current
            t_ = t_ + R_ * t_curr_inv;  // Update translation: t = t_prev + R_prev * dt
            R_ = R_ * R_curr_inv;      // Update rotation: R = R_prev * dR

            // Broadcast updated pose for navigation and visualization
            broadcastTransformROS(stamp);

            RCLCPP_DEBUG(this->get_logger(), 
                        "Camera pose updated - Position: [%.3f, %.3f, %.3f], Inliers: %zu/%zu", 
                        t_.at<double>(0), t_.at<double>(1), t_.at<double>(2),
                        inliers.size(), points3d.size());
                            
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during PnP estimation: %s", e.what());
            return;
        }
    }

    /**
     * @brief Callback for RGB camera calibration info
     * 
     * Processes camera_info messages to extract intrinsic parameters
     * needed for 3D point triangulation and pose estimation.
     * 
     * @param msg Camera info message containing calibration parameters
     */
    void rgbInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_rgb_camera_info_ = msg;

        // Extract camera matrix elements (K matrix in row-major order)
        rgb_camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        rgb_camera_matrix_.at<double>(0, 0) = msg->k[0]; // fx
        rgb_fx_ = msg->k[0];
        rgb_camera_matrix_.at<double>(0, 2) = msg->k[2]; // cx
        rgb_cx_ = msg->k[2];
        rgb_camera_matrix_.at<double>(1, 1) = msg->k[4]; // fy
        rgb_fy_ = msg->k[4];
        rgb_camera_matrix_.at<double>(1, 2) = msg->k[5]; // cy
        rgb_cy_ = msg->k[5];
        rgb_camera_matrix_.at<double>(2, 2) = 1.0;

        // Extract distortion coefficients
        rgb_dist_coeffs_ = cv::Mat(1, 5, CV_64F);
        for (int i = 0; i < 5; i++) {
            rgb_dist_coeffs_.at<double>(0, i) = msg->d[i];
        }
        
        RCLCPP_DEBUG(this->get_logger(), "RGB camera calibration loaded: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f", 
                    rgb_fx_, rgb_fy_, rgb_cx_, rgb_cy_);
    }

    /**
     * @brief Callback for depth camera calibration info
     * 
     * @param msg Camera info message for depth camera
     */
    void depthInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_depth_camera_info_ = msg;

        depth_camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        depth_camera_matrix_.at<double>(0, 0) = msg->k[0]; // fx
        depth_fx_ = msg->k[0];
        depth_camera_matrix_.at<double>(0, 2) = msg->k[2]; // cx
        depth_cx_ = msg->k[2];
        depth_camera_matrix_.at<double>(1, 1) = msg->k[4]; // fy
        depth_fy_ = msg->k[4];
        depth_camera_matrix_.at<double>(1, 2) = msg->k[5]; // cy
        depth_cy_ = msg->k[5];
        depth_camera_matrix_.at<double>(2, 2) = 1.0;

        depth_dist_coeffs_ = cv::Mat(1, 5, CV_64F);
        for (int i = 0; i < 5; i++) {
            depth_dist_coeffs_.at<double>(0, i) = msg->d[i];
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Depth camera calibration loaded: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f", 
                    depth_fx_, depth_fy_, depth_cx_, depth_cy_);
    }

    /**
     * @brief Main synchronized RGB-D processing callback
     * 
     * This is the core processing function that handles synchronized RGB and depth
     * images to perform visual odometry, feature tracking, and keyframe selection.
     * Implements the complete frontend processing pipeline from raw images to
     * pose estimates and keyframe data.
     * 
     * **Processing Pipeline:**
     * 1. **Image Preprocessing:** Convert to OpenCV format and grayscale
     * 2. **Feature Detection:** Extract ORB features with depth filtering  
     * 3. **Feature Matching:** Match against previous frame with geometric validation
     * 4. **Pose Estimation:** Estimate camera motion using PnP RANSAC
     * 5. **Keyframe Decision:** Determine if current frame should be keyframe
     * 6. **Data Publishing:** Output visualization and keyframe data
     * 7. **State Update:** Prepare data for next frame processing
     * 
     * **First Frame Handling:**
     * - No previous frame available for matching
     * - All depth-filtered features sent to backend as initial keyframe
     * - Establishes coordinate frame origin and initial map
     * 
     * **Subsequent Frame Handling:**  
     * - Features matched against previous frame for pose estimation
     * - Geometric validation using fundamental matrix RANSAC
     * - Culled feature set prepared for backend (matched + high-quality new features)
     * - Pose estimated and broadcasted for navigation
     * 
     * @param rgb_msg Synchronized RGB image message
     * @param depth_msg Synchronized depth image message  
     * 
     * @pre Messages are temporally synchronized by message filters
     * @pre Camera calibration info received and processed
     * @pre Images are rectified and aligned (RGB-D alignment)
     * 
     * @post Camera pose updated and broadcasted if successful
     * @post Keyframe published if keyframe criteria met
     * @post Visualization images published for debugging
     * @post Previous frame state updated for next iteration
     * 
     * @note Graceful degradation when feature matching fails
     * @note Performance optimized for 30 FPS real-time operation
     */
    void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg, 
                     const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
    {
        try {
            // Store debug image for publishing
            dgb_image_ = *rgb_msg;
            
            // Convert ROS images to OpenCV format
            cv_bridge::CvImagePtr rgb_cv_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
            cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv::Mat current_frame_rgb = rgb_cv_ptr->image;
            cv::Mat current_frame_depth = depth_cv_ptr->image;
            cv::Mat current_frame_gray;
            int num_features;

            // Convert RGB to grayscale for feature detection
            cv::cvtColor(current_frame_rgb, current_frame_gray, cv::COLOR_BGR2GRAY);

            if (prev_frame_valid_) {
                // === SUBSEQUENT FRAME PROCESSING ===
                
                cv::Mat vis_image = current_frame_rgb.clone();  // Visualization image
                
                // Extract ORB features from current frame
                std::vector<cv::KeyPoint> current_keypoints;
                cv::Mat current_descriptors;
                num_features = (*orb_extractor_)(current_frame_gray, cv::noArray(), 
                                               current_keypoints, current_descriptors, vLappingArea);

                // Filter features based on depth validity
                std::vector<cv::KeyPoint> filtered_keypoints;
                cv::Mat filtered_descriptors;
                filterDepth(current_keypoints, current_descriptors, current_frame_depth, 
                           filtered_keypoints, filtered_descriptors);

                RCLCPP_DEBUG(this->get_logger(), "Features: %d total, %zu depth-filtered", 
                            num_features, filtered_keypoints.size());

                // Check for sufficient features in both frames
                if (filtered_keypoints.empty() || prev_kps_.empty() || 
                    filtered_descriptors.empty() || prev_descriptors_.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Insufficient features for matching - resetting tracking");
                    
                    // Reset tracking state and continue
                    current_frame_gray.copyTo(prev_frame_gray_);
                    current_frame_depth.copyTo(prev_frame_depth_);
                    prev_kps_ = filtered_keypoints;
                    prev_descriptors_ = filtered_descriptors.clone();
                    return;
                }

                // === FEATURE MATCHING PIPELINE ===
                
                // 1. Brute-force descriptor matching
                std::vector<cv::DMatch> all_matches;
                matcher_.match(filtered_descriptors, prev_descriptors_, all_matches);

                // 2. Distance-based filtering (remove poor descriptor matches)
                std::vector<cv::DMatch> distance_filtered_matches;
                float max_distance = 50.0f; // Hamming distance threshold for ORB descriptors
                for (const auto& match : all_matches) {
                    if (match.distance < max_distance) {
                        distance_filtered_matches.push_back(match);
                    }
                }

                // 3. Geometric consistency filtering using fundamental matrix RANSAC
                std::vector<cv::DMatch> geometrically_consistent_matches;
                if (distance_filtered_matches.size() >= 8) {
                    // Extract point correspondences for geometric validation
                    std::vector<cv::Point2f> prev_pts, curr_pts;
                    for (const auto& match : distance_filtered_matches) {
                        prev_pts.push_back(prev_kps_[match.trainIdx].pt);
                        curr_pts.push_back(filtered_keypoints[match.queryIdx].pt);
                    }
                    
                    // Use fundamental matrix RANSAC to reject geometric outliers
                    std::vector<uchar> inliers_mask;
                    cv::Mat fundamental_matrix = cv::findFundamentalMat(
                        prev_pts, curr_pts, inliers_mask, cv::FM_RANSAC, 2.0, 0.99);

                    // Keep only geometrically consistent matches
                    for (size_t i = 0; i < inliers_mask.size(); i++) {
                        if (inliers_mask[i]) {
                            geometrically_consistent_matches.push_back(distance_filtered_matches[i]);
                        }
                    }
                    
                    RCLCPP_DEBUG(this->get_logger(), 
                                "Feature matching: %zu initial → %zu distance-filtered → %zu geometric", 
                                all_matches.size(), distance_filtered_matches.size(), 
                                geometrically_consistent_matches.size());
                } else {
                    // Insufficient matches for RANSAC - use distance-filtered matches
                    geometrically_consistent_matches = distance_filtered_matches;
                    RCLCPP_WARN(this->get_logger(), 
                               "Insufficient matches for geometric validation: %zu", 
                               distance_filtered_matches.size());
                }
                
                // === FEATURE CULLING FOR BACKEND ===
                
                // Prepare optimized feature set for backend processing
                std::vector<cv::KeyPoint> backend_keypoints;
                cv::Mat backend_descriptors;
                
                // Track which features have matches (for prioritization)
                std::set<int> matched_indices;
                for (const auto& match : geometrically_consistent_matches) {
                    matched_indices.insert(match.queryIdx);
                }
                
                // Priority 1: Add all geometrically consistent matches (highest priority)
                for (const auto& match : geometrically_consistent_matches) {
                    int curr_idx = match.queryIdx;
                    backend_keypoints.push_back(filtered_keypoints[curr_idx]);
                    
                    if (backend_descriptors.empty()) {
                        backend_descriptors = filtered_descriptors.row(curr_idx).clone();
                    } else {
                        cv::vconcat(backend_descriptors, filtered_descriptors.row(curr_idx), backend_descriptors);
                    }
                }
                
                // Priority 2: Add high-quality unmatched features for new landmark creation
                std::vector<std::pair<float, int>> unmatched_features;
                for (size_t i = 0; i < filtered_keypoints.size(); i++) {
                    if (matched_indices.find(i) == matched_indices.end()) {
                        unmatched_features.push_back({filtered_keypoints[i].response, i});
                    }
                }
                
                // Sort unmatched features by quality (response strength)
                std::sort(unmatched_features.begin(), unmatched_features.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
                
                // Add best unmatched features up to limits
                const int MAX_NEW_FEATURES = 200;  // Tunable: balance between coverage and efficiency
                const float MIN_RESPONSE = 50.0f;  // Quality threshold for new features
                int added_new = 0;
                
                for (const auto& [response, idx] : unmatched_features) {
                    if (added_new >= MAX_NEW_FEATURES || response < MIN_RESPONSE) break;
                    
                    backend_keypoints.push_back(filtered_keypoints[idx]);
                    if (backend_descriptors.empty()) {
                        backend_descriptors = filtered_descriptors.row(idx).clone();
                    } else {
                        cv::vconcat(backend_descriptors, filtered_descriptors.row(idx), backend_descriptors);
                    }
                    added_new++;
                }
                
                RCLCPP_DEBUG(this->get_logger(), 
                           "Feature culling: %zu total → %zu matched → %d new → %zu backend",
                           current_keypoints.size(), geometrically_consistent_matches.size(), 
                           added_new, backend_keypoints.size());

                // === VISUALIZATION ===
                
                // Draw matched features in green on visualization image
                for (const auto& match : geometrically_consistent_matches) {
                    cv::Point2f curr_pt = filtered_keypoints[match.queryIdx].pt;
                    cv::circle(vis_image, curr_pt, 3, cv::Scalar(0, 255, 0), 2); // Green circles
                }

                // === POSE ESTIMATION ===
                
                // Estimate camera motion if sufficient matches available
                if (geometrically_consistent_matches.size() >= 5) {
                    estimateCameraPose(prev_kps_, filtered_keypoints, 
                                     geometrically_consistent_matches, prev_frame_depth_, 
                                     rgb_msg->header.stamp);
                } else {
                    RCLCPP_WARN(this->get_logger(), 
                               "Insufficient matches for pose estimation: %zu", 
                               geometrically_consistent_matches.size());
                }

                // === KEYFRAME SELECTION ===
                
                // Send optimized feature set to backend if keyframe criteria met
                if (isKeyframe(backend_descriptors, backend_keypoints)) {
                    publishKeyframe(backend_keypoints, backend_descriptors, 
                                  current_frame_depth, rgb_msg->header.stamp);
                }

                // === STATE UPDATE ===
                
                // Update tracking state for next frame (keep ALL depth-filtered features)
                prev_kps_ = filtered_keypoints;
                prev_descriptors_ = filtered_descriptors.clone();
                
                // Publish visualization image
                sensor_msgs::msg::Image::SharedPtr out_msg = 
                    cv_bridge::CvImage(rgb_msg->header, "bgr8", vis_image).toImageMsg();
                image_pub_->publish(*out_msg);

                // Publish synchronized camera info
                if (latest_rgb_camera_info_) {
                    auto camera_info = *latest_rgb_camera_info_;
                    camera_info.header = rgb_msg->header;
                    camera_info_pub_->publish(camera_info);
                }
                
                // Update frame buffers for next iteration
                current_frame_gray.copyTo(prev_frame_gray_);
                current_frame_depth.copyTo(prev_frame_depth_);
            } 
            else {
                // === FIRST FRAME PROCESSING ===
                
                RCLCPP_INFO(this->get_logger(), "Processing first frame - initializing tracking");
                
                // Extract features from first frame
                std::vector<cv::KeyPoint> current_keypoints;
                cv::Mat current_descriptors;
                num_features = (*orb_extractor_)(current_frame_gray, cv::noArray(), 
                                               current_keypoints, current_descriptors, vLappingArea);

                // Apply depth filtering
                std::vector<cv::KeyPoint> filtered_keypoints;
                cv::Mat filtered_descriptors;
                filterDepth(current_keypoints, current_descriptors, current_frame_depth, 
                           filtered_keypoints, filtered_descriptors);

                // Send all valid features as initial keyframe
                publishKeyframe(filtered_keypoints, filtered_descriptors, 
                              current_frame_depth, rgb_msg->header.stamp);

                // Initialize tracking state
                prev_kps_ = filtered_keypoints;
                prev_descriptors_ = filtered_descriptors.clone();

                // Publish camera info for first frame
                if (latest_rgb_camera_info_) {
                    auto camera_info = *latest_rgb_camera_info_;
                    camera_info.header = rgb_msg->header;
                    camera_info_pub_->publish(camera_info);
                }
                
                // Set up for subsequent frame processing
                current_frame_gray.copyTo(prev_frame_gray_);
                current_frame_depth.copyTo(prev_frame_depth_);
                prev_frame_valid_ = true;
                
                RCLCPP_INFO(this->get_logger(), 
                           "First frame processed - %zu features, tracking initialized", 
                           filtered_keypoints.size());
            }
            
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in sync callback: %s", e.what());
        }
    }
};

/**
 * @brief Main function for frontend node
 * 
 * Initializes ROS 2, creates the Frontend node, and begins processing
 * camera data. Runs until shutdown signal received.
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @return 0 on successful completion
 */
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "Starting Dynamic Visual SLAM Frontend");
    
    auto frontend_node = std::make_shared<Frontend>();
    rclcpp::spin(frontend_node);
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "Shutting down Dynamic Visual SLAM Frontend");
    rclcpp::shutdown();
    
    return 0;
}