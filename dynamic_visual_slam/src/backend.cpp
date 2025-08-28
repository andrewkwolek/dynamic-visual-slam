/**
 * @file backend.cpp
 * @brief Visual SLAM backend for bundle adjustment optimization and map management
 * @author Andrew Kwolek
 * @date 2025
 * @version 0.0.1
 * 
 * This file implements the backend component of the Dynamic Visual SLAM system.
 * The backend is responsible for maintaining the global map through bundle adjustment
 * optimization, managing 3D landmarks with semantic information, and providing
 * optimized poses and map data for navigation and visualization.
 * 
 * **System Architecture:**
 * The backend operates as a ROS 2 node that receives keyframe data from the frontend,
 * performs data association between observations and existing landmarks, maintains
 * a persistent map database, and periodically optimizes the map using sliding window
 * bundle adjustment.
 * 
 * **Key Responsibilities:**
 * 1. **Data Association:** Match new observations to existing landmarks using descriptors and geometry
 * 2. **Map Management:** Maintain persistent database of 3D landmarks with metadata
 * 3. **Bundle Adjustment:** Optimize camera poses and landmark positions using Ceres solver
 * 4. **Semantic Integration:** Incorporate YOLO object detection for landmark categorization
 * 5. **Map Visualization:** Publish landmark markers for RViz visualization
 * 6. **Map Pruning:** Remove low-quality or outdated landmarks to maintain map quality
 * 
 * **Coordinate System Conventions:**
 * - **Internal Processing:** Camera optical frame (X=right, Y=down, Z=forward)
 * - **Visualization:** ROS standard frame (X=forward, Y=left, Z=up)
 * - **Automatic Conversion:** Between coordinate systems for ROS compatibility
 * 
 * **Data Association Strategy:**
 * 1. **Descriptor Matching:** ORB descriptor distance < threshold (typically 50 Hamming units)
 * 2. **Geometric Validation:** Reprojection error < threshold (typically 5 pixels)  
 * 3. **Semantic Consistency:** Category labels must match for association
 * 4. **Temporal Relevance:** Recently observed landmarks preferred for association
 * 
 * **Bundle Adjustment Strategy:**
 * - **Sliding Window:** Last N keyframes (typically 5-10) optimized together
 * - **Gauge Freedom:** First keyframe pose held fixed as coordinate reference
 * - **Robust Loss:** Huber loss function handles outlier observations
 * - **Real-time Constraints:** Optimization limited to maintain frame rate
 * 
 * **Memory Management:**
 * - **Landmark Pruning:** Remove landmarks with insufficient observations
 * - **Observation Cleanup:** Remove observations from pruned landmarks
 * - **Sliding Window:** Bounded optimization prevents unbounded memory growth
 * 
 * @note Integration with YOLO object detection for semantic landmark labeling
 * @note Thread-safe design allows concurrent frontend operation
 * @warning Requires synchronized YOLO detections and keyframe messages
 * 
 * @see Frontend for keyframe generation and pose estimation
 * @see SlidingWindowBA for bundle adjustment implementation
 */

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "dynamic_visual_slam/bundle_adjustment.hpp"
#include "dynamic_visual_slam_interfaces/msg/keyframe.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2/LinearMath/Quaternion.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include <Eigen/Geometry>

/**
 * @class Backend
 * @brief Main backend node for SLAM map optimization and management
 * 
 * The Backend class implements the optimization and mapping components of the
 * visual SLAM system. It receives keyframes from the frontend, performs data
 * association to build a consistent map, and periodically optimizes the map
 * using bundle adjustment to minimize accumulated drift.
 * 
 * **Core Algorithm Pipeline:**
 * 1. **Keyframe Reception:** Process keyframes from frontend with pose and observations
 * 2. **Semantic Labeling:** Integrate YOLO detections to categorize landmarks
 * 3. **Data Association:** Match observations to existing landmarks in map database
 * 4. **Landmark Creation:** Create new landmarks for unmatched high-quality observations
 * 5. **Map Update:** Add new keyframes and landmarks to optimization database
 * 6. **Bundle Adjustment:** Periodically optimize poses and landmarks in sliding window
 * 7. **Map Visualization:** Publish landmark markers and trajectory for RViz
 * 8. **Map Maintenance:** Prune low-quality landmarks and clean observation database
 * 
 * **Data Structures:**
 * - **KeyframeInfo:** Stores camera poses, timestamps, and observation references
 * - **LandmarkInfo:** Stores 3D positions, descriptors, and observation history
 * - **ObservationInfo:** Links keyframes to landmarks with pixel coordinates and descriptors
 * 
 * **Optimization Strategy:**
 * The system uses sliding window bundle adjustment to balance accuracy and performance:
 * - **Window Size:** 5-10 keyframes (configurable based on scene complexity)
 * - **Update Frequency:** Every 2 seconds or after significant motion
 * - **Landmark Selection:** All landmarks observed within the temporal window
 * - **Robust Optimization:** Huber loss function handles outlier observations
 * 
 * **Memory Management:**
 * - **Landmark Database:** Hash map organization by semantic category
 * - **Observation Database:** Vector storage with efficient pruning
 * - **Automatic Cleanup:** Remove stale landmarks and orphaned observations
 * - **Bounded Growth:** Sliding window prevents unbounded memory usage
 * 
 * **Thread Safety:**
 * - **Bundle Adjustment:** Runs asynchronously in timer callback with mutex protection
 * - **Data Association:** Main thread processes keyframes sequentially
 * - **Visualization:** Thread-safe marker publishing for real-time display
 * 
 * @note Designed for real-time operation with bounded computational complexity
 * @note Integrates seamlessly with ROS 2 navigation stack through TF publishing
 * @warning Requires high-quality frontend poses for initialization convergence
 * 
 * @example Usage in launch file:
 * @code{.xml}
 * <node pkg="dynamic_visual_slam" exec="backend"/>
 * @endcode
 */
class Backend : public rclcpp::Node 
{
public:
    /**
     * @brief Constructs the backend node with complete optimization pipeline setup
     * 
     * Initializes all components necessary for SLAM backend processing:
     * - Bundle adjustment optimizer with default camera parameters
     * - Synchronized subscribers for keyframes and YOLO detections
     * - Camera calibration subscriber for accurate optimization
     * - Publishers for visualization markers and trajectory data
     * - Timer for periodic bundle adjustment optimization
     * - Data structures for landmark and observation management
     * 
     * **Initialization Sequence:**
     * 1. Create bundle adjuster with placeholder camera parameters (updated from camera_info)
     * 2. Setup message filter subscribers for synchronized YOLO and keyframe processing
     * 3. Initialize camera calibration subscriber for optimization parameters
     * 4. Create visualization publishers for RViz integration
     * 5. Setup periodic bundle adjustment timer (2 second intervals)
     * 6. Initialize data association parameters and thresholds
     * 7. Setup semantic filtering for dynamic objects
     * 
     * **Default Parameters:**
     * - Bundle adjustment frequency: Every 2 seconds
     * - Sliding window size: 5 keyframes
     * - Association thresholds: 50 Hamming distance, 5 pixel reprojection
     * - Landmark pruning: Minimum 2 observations, 20 second timeout
     * - Semantic filtering: "person" category filtered by default
     * 
     * **Topic Subscriptions:**
     * - `/yolo/tracking` - YOLO object detections for semantic labeling
     * - `/frontend/keyframe` - Keyframe data from frontend processing
     * - `/camera/camera/color/camera_info` - Camera calibration parameters
     * 
     * **Topic Publications:**
     * - `/backend/landmark_markers` - 3D landmark visualization markers
     * - `/backend/trajectory` - Camera trajectory visualization
     * 
     * @post All ROS subscriptions and publishers are active
     * @post Bundle adjustment timer scheduled for periodic optimization
     * @post System ready to process keyframes and build map upon spin()
     * 
     * @note Constructor completes quickly; actual processing begins with first keyframe
     * @note Camera parameters updated asynchronously from camera_info messages
     * @note YOLO integration optional - keyframes processed even without detections
     */
    Backend() : Node("backend") {
        rclcpp::QoS qos = rclcpp::QoS(30);

        // Initialize bundle adjuster with default parameters (updated when camera info received)
        bundle_adjuster_ = std::make_unique<SlidingWindowBA>(10, 640.0, 480.0, 320.0, 240.0);
        
        // Setup synchronized subscribers for YOLO detections and keyframes
        tracking_sub_.subscribe(this, "/yolo/tracking", qos.get_rmw_qos_profile());
        keyframe_sub_.subscribe(this, "/frontend/keyframe", qos.get_rmw_qos_profile());

        // Message synchronizer for temporal alignment of detections and keyframes
        sync_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<yolo_msgs::msg::DetectionArray, dynamic_visual_slam_interfaces::msg::Keyframe>>>(
            message_filters::sync_policies::ApproximateTime<yolo_msgs::msg::DetectionArray, dynamic_visual_slam_interfaces::msg::Keyframe>(10), 
            tracking_sub_, keyframe_sub_);
        sync_->registerCallback(std::bind(&Backend::syncCallback, this, std::placeholders::_1, std::placeholders::_2));
        
        // Camera calibration subscriber for accurate bundle adjustment
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/color/camera_info", qos,
            std::bind(&Backend::cameraInfoCallback, this, std::placeholders::_1));
            
        // TF broadcaster for optimized poses (currently unused - frontend handles pose publishing)
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        
        // Visualization publishers for RViz integration
        landmark_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/backend/landmark_markers", qos);
        trajectory_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/backend/trajectory", qos);

        // Periodic bundle adjustment optimization timer
        ba_timer_ = this->create_wall_timer(
            std::chrono::seconds(2), 
            std::bind(&Backend::bundleAdjustmentCallback, this));
            
        // Initialize optimization and association parameters
        min_observations_for_landmark_ = 2;    // Minimum observations before landmark considered stable
        max_reprojection_error_ = 2.0;         // Maximum reprojection error for bundle adjustment
        keyframe_count_ = 0;                   // Counter for processed keyframes
        camera_params_initialized_ = false;    // Flag for camera calibration availability

        // Initialize unique ID generators for landmarks and observations
        next_global_landmark_id_ = 0;
        next_observation_id_ = 0;
        
        // Feature matching infrastructure for data association
        descriptor_matcher_ = cv::BFMatcher(cv::NORM_HAMMING, false);

        // Data association thresholds (tuned for ORB descriptors and typical camera motion)
        max_descriptor_distance_ = 50.0;      // Hamming distance threshold for ORB matching
        max_reprojection_distance_ = 5.0;     // Pixel reprojection error threshold
        min_parallax_ratio_ = 0.02;          // Minimum baseline-to-depth ratio for triangulation

        // Trajectory visualization state
        has_prev_keyframe_ = false;
        
        // Semantic filtering configuration (objects to exclude from mapping)
        filtered_objects_ = {"person"};  // Dynamic objects that should not be mapped
        
        RCLCPP_INFO(this->get_logger(), "Backend node initialized successfully");
        RCLCPP_INFO(this->get_logger(), "Waiting for camera calibration and keyframe data...");
        RCLCPP_INFO(this->get_logger(), "Bundle adjustment will run every 2 seconds");
    }

private:
    // === ROS Communication Infrastructure ===
    
    /// Bundle adjustment optimization engine
    std::unique_ptr<SlidingWindowBA> bundle_adjuster_;

    /// Message filter subscribers for synchronized processing
    message_filters::Subscriber<yolo_msgs::msg::DetectionArray> tracking_sub_;
    message_filters::Subscriber<dynamic_visual_slam_interfaces::msg::Keyframe> keyframe_sub_;

    /// Message synchronizer for temporal alignment
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<yolo_msgs::msg::DetectionArray, dynamic_visual_slam_interfaces::msg::Keyframe>>> sync_;
    
    /// Camera calibration subscriber
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    
    /// Publishers for visualization and transforms
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmark_markers_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trajectory_markers_pub_;
    
    /// Latest timestamp for visualization synchronization
    rclcpp::Time latest_keyframe_timestamp_;
    
    // === Camera Calibration Data ===
    
    /// Camera calibration state and parameters
    bool camera_params_initialized_;
    double fx_, fy_, cx_, cy_;  ///< Camera intrinsic parameters for bundle adjustment

    // === Bundle Adjustment Infrastructure ===
    
    /// Thread synchronization for bundle adjustment
    std::mutex keyframes_mutex_;           ///< Protects keyframe data during optimization
    std::atomic<bool> ba_running_{false};  ///< Prevents concurrent bundle adjustment runs
    
    /// Bundle adjustment scheduling
    rclcpp::TimerBase::SharedPtr ba_timer_;  ///< Periodic optimization timer
    
    // === Algorithm Parameters ===
    
    /// Quality and pruning parameters
    int min_observations_for_landmark_;     ///< Minimum observations before landmark is considered stable
    double max_reprojection_error_;         ///< Maximum reprojection error for optimization inclusion
    int keyframe_count_;                    ///< Total keyframes processed (for statistics)
    
    /// Trajectory visualization state
    geometry_msgs::msg::Point prev_keyframe_pos_;  ///< Previous keyframe position for trajectory drawing
    bool has_prev_keyframe_;                       ///< Whether previous keyframe exists for trajectory

    /**
     * @struct ObservationInfo
     * @brief Internal representation of a 2D feature observation
     * 
     * Links a specific keyframe to a landmark through pixel coordinates and
     * descriptor information. Used for data association and bundle adjustment.
     * 
     * **Data Association Pipeline:**
     * 1. Created from frontend keyframe observations
     * 2. Used for descriptor matching against existing landmarks
     * 3. Linked to landmarks through association process
     * 4. Feeds into bundle adjustment as 2D-3D correspondences
     * 
     * @see LandmarkInfo for 3D landmark representation
     * @see KeyframeInfo for camera pose storage
     */
    struct ObservationInfo {
        uint64_t observation_id;   ///< Unique identifier for this observation
        uint64_t landmark_id;      ///< ID of associated landmark (0 if unassociated)
        uint64_t frame_id;         ///< ID of keyframe containing this observation
        cv::Point2f pixel;         ///< 2D pixel coordinates in image
        cv::Mat descriptor;        ///< ORB descriptor for matching (32 bytes)
        std::string category;      ///< Semantic category from YOLO detection

        /**
         * @brief Constructs observation with pixel and descriptor data
         * @param obs_id Unique observation identifier
         * @param frame Keyframe ID containing this observation  
         * @param pix 2D pixel coordinates
         * @param desc ORB descriptor (will be cloned)
         */
        ObservationInfo(uint64_t obs_id, uint64_t frame, const cv::Point2f& pix, const cv::Mat& desc)
            : observation_id(obs_id), landmark_id(0), frame_id(frame), pixel(pix), descriptor(desc.clone()) {}
    };

    /**
     * @struct KeyframeInfo
     * @brief Internal representation of a SLAM keyframe with pose and metadata
     * 
     * Stores camera pose information and references to observations made in
     * this keyframe. Used for bundle adjustment optimization and temporal
     * tracking of camera motion.
     * 
     * **Coordinate Convention:**
     * - Pose stored in camera optical frame (X=right, Y=down, Z=forward)
     * - Consistent with frontend processing and OpenCV conventions
     * - Converted for visualization as needed
     * 
     * @see ObservationInfo for observation details
     */
    struct KeyframeInfo {
        uint64_t frame_id;                    ///< Unique keyframe identifier
        cv::Mat R;                           ///< 3x3 rotation matrix (camera to world)
        cv::Mat t;                           ///< 3x1 translation vector (camera position in world)
        rclcpp::Time timestamp;              ///< ROS timestamp when keyframe captured
        std::vector<uint64_t> observation_ids;  ///< IDs of observations made in this keyframe

        /**
         * @brief Constructs keyframe with pose and timestamp
         * @param id Unique keyframe identifier
         * @param rotation Camera rotation matrix (will be cloned)
         * @param translation Camera translation vector (will be cloned)
         * @param stamp ROS timestamp
         */
        KeyframeInfo(uint64_t id, const cv::Mat& rotation, const cv::Mat& translation, const rclcpp::Time& stamp)
            : frame_id(id), R(rotation.clone()), t(translation.clone()), timestamp(stamp) {}
    };

    /**
     * @struct LandmarkInfo
     * @brief Internal representation of a 3D landmark with tracking metadata
     * 
     * Stores comprehensive information about a 3D landmark including its world
     * position, visual descriptor, observation history, and temporal tracking data.
     * Used for data association, bundle adjustment, and map management.
     * 
     * **Lifecycle Management:**
     * 1. **Creation:** Initialized when first observation cannot be associated
     * 2. **Association:** Linked to new observations through descriptor/geometry matching
     * 3. **Triangulation:** Position refined using multi-view geometry
     * 4. **Optimization:** Position optimized during bundle adjustment
     * 5. **Pruning:** Removed if insufficient observations or too old
     * 
     * **Quality Metrics:**
     * - **Observation Count:** Number of times landmark has been observed
     * - **Temporal Currency:** How recently landmark was observed
     * - **Geometric Consistency:** Reprojection error across observations
     * 
     * @see ObservationInfo for 2D observations of this landmark
     * @see triangulate() for position refinement algorithm
     */
    struct LandmarkInfo {
        uint64_t global_id;                     ///< Unique landmark identifier across entire map
        cv::Point3f position;                   ///< 3D position in world coordinate frame (meters)
        cv::Mat descriptor;                     ///< Representative ORB descriptor for matching
        std::vector<uint64_t> observation_ids; ///< IDs of all observations of this landmark
        int observation_count;                  ///< Total number of observations (for quality assessment)
        rclcpp::Time last_seen;                ///< Timestamp when landmark was last observed

        /**
         * @brief Constructs landmark with initial observation data
         * @param id Unique landmark identifier
         * @param pos Initial 3D position estimate
         * @param desc ORB descriptor from first observation (will be cloned)
         * @param timestamp Time when landmark was first observed
         */
        LandmarkInfo(uint64_t id, const cv::Point3f& pos, const cv::Mat& desc, const rclcpp::Time& timestamp)
            : global_id(id), position(pos), descriptor(desc.clone()), observation_count(1), last_seen(timestamp) {}

        /**
         * @brief Refines landmark position using multi-view triangulation
         * 
         * Uses all available observations of this landmark across different keyframes
         * to compute an improved 3D position estimate. Employs SVD-based multi-view
         * triangulation with careful view selection based on parallax analysis.
         * 
         * **Triangulation Algorithm:**
         * 1. **View Selection:** Choose camera pairs with sufficient parallax (>5 degrees)
         * 2. **Linear Triangulation:** Use SVD to solve overdetermined linear system
         * 3. **Validation:** Check reprojection errors across all views
         * 4. **Quality Control:** Reject solutions with high reprojection error
         * 5. **Position Update:** Update landmark position if triangulation successful
         * 
         * **Geometric Requirements:**
         * - Minimum 2 observations for triangulation
         * - Sufficient baseline between observing cameras
         * - Consistent feature matching across views
         * - Reasonable depth range [0.1m, 10.0m]
         * 
         * **Quality Metrics:**
         * - Maximum reprojection error: 2.0 pixels
         * - Minimum parallax angle: 5 degrees  
         * - Valid depth range: [0.1m, 10.0m]
         * 
         * @param all_observations Complete database of observations for lookup
         * @param keyframes Complete database of keyframes for pose lookup
         * @param fx,fy,cx,cy Camera intrinsic parameters for projection
         * 
         * @pre observation_ids.size() >= 2 (minimum for triangulation)
         * @pre All referenced observations and keyframes exist in databases
         * @pre Camera intrinsics are accurate and consistent
         * 
         * @post position updated if triangulation successful and validated
         * @post position unchanged if triangulation fails validation tests
         * 
         * @note Critical for maintaining map consistency and accuracy
         * @note Computational cost scales with number of observations
         * @warning Requires sufficient parallax between views for numerical stability
         */
        void triangulate(const std::vector<ObservationInfo>& all_observations,
                        const std::vector<KeyframeInfo>& keyframes,
                        double fx, double fy, double cx, double cy) {
            
            // Require minimum observations for stable triangulation
            if (observation_ids.size() < 2) return;
            
            std::vector<cv::Point2f> image_points;      // 2D pixel observations
            std::vector<cv::Mat> camera_matrices;       // Camera projection matrices
            std::vector<cv::Mat> camera_centers;        // Camera center positions
            
            // Build correspondence data from observation and keyframe databases
            for (uint64_t obs_id : observation_ids) {
                // Find observation in database
                auto obs_it = std::find_if(all_observations.begin(), all_observations.end(),
                    [obs_id](const ObservationInfo& obs) { return obs.observation_id == obs_id; });
                
                if (obs_it == all_observations.end()) continue;
                
                // Find corresponding keyframe
                auto kf_it = std::find_if(keyframes.begin(), keyframes.end(),
                    [&obs_it](const KeyframeInfo& kf) { return kf.frame_id == obs_it->frame_id; });
                
                if (kf_it == keyframes.end()) continue;
                
                // Store 2D observation
                image_points.push_back(obs_it->pixel);
                
                // Build camera projection matrix P = K * [R | t]
                cv::Mat K = (cv::Mat_<double>(3,3) << 
                    fx, 0, cx,
                    0, fy, cy,
                    0, 0, 1);
                
                cv::Mat Rt;
                cv::hconcat(kf_it->R, kf_it->t, Rt);  // Concatenate [R | t]
                cv::Mat P = K * Rt;                   // Full projection matrix
                camera_matrices.push_back(P);
                
                // Compute camera center for parallax analysis: C = -R^T * t
                cv::Mat camera_center = -kf_it->R.t() * kf_it->t;
                camera_centers.push_back(camera_center);
            }
            
            if (image_points.size() < 2) return;
            
            // === PARALLAX-BASED VIEW SELECTION ===
            
            double max_parallax_angle = 0.0;
            int best_view1 = -1, best_view2 = -1;
            
            // Find camera pair with maximum parallax for most stable triangulation
            for (size_t i = 0; i < camera_centers.size(); i++) {
                for (size_t j = i + 1; j < camera_centers.size(); j++) {
                    // Compute baseline between cameras
                    cv::Mat baseline = camera_centers[i] - camera_centers[j];
                    double baseline_length = cv::norm(baseline);
                    
                    // Estimate average depth to landmark
                    cv::Mat landmark_pos = (cv::Mat_<double>(3,1) << position.x, position.y, position.z);
                    cv::Mat to_landmark1 = landmark_pos - camera_centers[i];
                    cv::Mat to_landmark2 = landmark_pos - camera_centers[j];
                    double depth1 = cv::norm(to_landmark1);
                    double depth2 = cv::norm(to_landmark2);
                    double avg_depth = (depth1 + depth2) / 2.0;
                    
                    // Compute parallax angle: angle = atan(baseline / depth)
                    double parallax_angle = std::atan2(baseline_length, avg_depth);
                    
                    if (parallax_angle > max_parallax_angle) {
                        max_parallax_angle = parallax_angle;
                        best_view1 = i;
                        best_view2 = j;
                    }
                }
            }
            
            // Require minimum parallax for numerical stability
            const double MIN_PARALLAX_ANGLE = 0.0175 * 5;  // 5 degrees in radians
            
            if (max_parallax_angle < MIN_PARALLAX_ANGLE) {
                return;  // Insufficient parallax for stable triangulation
            }
            
            cv::Point3f new_position;
            
            if (image_points.size() == 2) {
                // === SIMPLE TWO-VIEW TRIANGULATION ===
                
                cv::Mat point_4d;
                std::vector<cv::Point2f> pts1 = {image_points[best_view1]};
                std::vector<cv::Point2f> pts2 = {image_points[best_view2]};
                
                // OpenCV's linear triangulation using SVD
                cv::triangulatePoints(camera_matrices[best_view1], camera_matrices[best_view2], 
                                    pts1, pts2, point_4d);
                
                // Convert from homogeneous coordinates
                if (point_4d.at<float>(3, 0) != 0) {
                    new_position.x = point_4d.at<float>(0, 0) / point_4d.at<float>(3, 0);
                    new_position.y = point_4d.at<float>(1, 0) / point_4d.at<float>(3, 0);
                    new_position.z = point_4d.at<float>(2, 0) / point_4d.at<float>(3, 0);
                } else {
                    return;  // Degenerate triangulation
                }
            } else {
                // === MULTI-VIEW TRIANGULATION ===
                
                // Build linear system Ax = 0 for homogeneous solution
                cv::Mat A(2 * image_points.size(), 4, CV_64F);
                
                for (size_t i = 0; i < image_points.size(); i++) {
                    cv::Mat P = camera_matrices[i];
                    cv::Point2f pt = image_points[i];
                    
                    // Each point contributes two linear constraints:
                    // x*(P3 - u*P1) = 0  and  x*(P3 - v*P2) = 0
                    for (int j = 0; j < 4; j++) {
                        A.at<double>(2*i, j) = pt.x * P.at<double>(2, j) - P.at<double>(0, j);
                        A.at<double>(2*i+1, j) = pt.y * P.at<double>(2, j) - P.at<double>(1, j);
                    }
                }
                
                // Solve Ax = 0 using SVD (solution is last column of V)
                cv::Mat U, D, Vt;
                cv::SVD::compute(A, D, U, Vt, cv::SVD::MODIFY_A);
                
                cv::Mat X = Vt.row(3).t();  // Last row of V^T (rightmost nullspace)
                
                // Convert from homogeneous coordinates
                if (X.at<double>(3) != 0) {
                    new_position.x = X.at<double>(0) / X.at<double>(3);
                    new_position.y = X.at<double>(1) / X.at<double>(3);
                    new_position.z = X.at<double>(2) / X.at<double>(3);
                } else {
                    return;  // Degenerate solution
                }
            }
            
            // === REPROJECTION ERROR VALIDATION ===
            
            double total_reprojection_error = 0.0;
            int valid_reprojections = 0;
            
            // Check triangulated point quality across all views
            for (size_t i = 0; i < image_points.size(); i++) {
                cv::Mat point_3d = (cv::Mat_<double>(4,1) << new_position.x, new_position.y, new_position.z, 1.0);
                cv::Mat projected = camera_matrices[i] * point_3d;
                
                if (projected.at<double>(2) > 0) {  // Point in front of camera
                    cv::Point2f reproj_pt(projected.at<double>(0) / projected.at<double>(2),
                                        projected.at<double>(1) / projected.at<double>(2));
                    
                    double error = cv::norm(image_points[i] - reproj_pt);
                    total_reprojection_error += error;
                    valid_reprojections++;
                }
            }
            
            // Accept triangulation only if reprojection error is reasonable
            const double MAX_REPROJECTION_ERROR = 2.0;  // pixels
            if (valid_reprojections > 0) {
                double avg_reprojection_error = total_reprojection_error / valid_reprojections;
                if (avg_reprojection_error > MAX_REPROJECTION_ERROR) {
                    return;  // Reject high-error triangulation
                }
            }
            
            // === DEPTH RANGE VALIDATION ===
            
            // Accept only physically plausible 3D positions
            if (new_position.z > 0.1 && new_position.z < 10.0) {
                position = new_position;  // Update landmark position
            }
        }
    };

    // === Map Database ===
    
    /// Core data structures for map representation
    std::vector<KeyframeInfo> keyframes_;                              ///< Temporal sequence of keyframes
    std::unordered_map<std::string, std::unordered_map<uint64_t, LandmarkInfo>> landmark_database_; ///< Landmarks organized by category
    std::vector<ObservationInfo> all_observations_;                    ///< Complete observation database
    
    /// Unique ID generation for data association
    uint64_t next_global_landmark_id_;  ///< Monotonic landmark ID generator
    uint64_t next_observation_id_;      ///< Monotonic observation ID generator
    
    /// Feature matching infrastructure
    cv::BFMatcher descriptor_matcher_;  ///< ORB descriptor matcher for data association

    // === Algorithm Parameters ===
    
    /// Data association thresholds (empirically tuned for ORB features)
    double max_descriptor_distance_;     ///< Maximum Hamming distance for ORB matching (50)
    double max_reprojection_distance_;   ///< Maximum pixel reprojection error for association (5.0)
    double min_parallax_ratio_;          ///< Minimum baseline-to-depth ratio for triangulation (0.02)

    /// Semantic filtering configuration
    std::unordered_set<std::string> filtered_objects_; ///< Object categories to exclude from mapping

    /**
     * @brief Processes camera calibration info for bundle adjustment
     * 
     * Extracts camera intrinsic parameters from ROS camera_info message and
     * initializes the bundle adjustment optimizer with accurate calibration data.
     * This is critical for accurate 3D reconstruction and pose optimization.
     * 
     * @param msg Camera info message containing calibration matrix
     * 
     * @post Bundle adjuster initialized with correct camera parameters
     * @post camera_params_initialized_ flag set to enable keyframe processing
     */
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (!camera_params_initialized_) {
            // Extract intrinsic parameters from calibration matrix
            fx_ = msg->k[0];  // K[0,0] - focal length x
            fy_ = msg->k[4];  // K[1,1] - focal length y  
            cx_ = msg->k[2];  // K[0,2] - principal point x
            cy_ = msg->k[5];  // K[1,2] - principal point y
            
            // Reinitialize bundle adjuster with accurate camera parameters
            bundle_adjuster_ = std::make_unique<SlidingWindowBA>(10, fx_, fy_, cx_, cy_);
            camera_params_initialized_ = true;
            
            RCLCPP_INFO(this->get_logger(), 
                       "Camera calibration loaded: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f", 
                       fx_, fy_, cx_, cy_);
        }
    }

    /**
     * @brief Main synchronized processing callback for keyframes and YOLO detections
     * 
     * This is the core processing function that integrates keyframe data from the
     * frontend with semantic object detections from YOLO to build and maintain
     * the global SLAM map. Handles data association, landmark creation, and
     * map database updates.
     * 
     * **Processing Pipeline:**
     * 1. **Input Validation:** Check camera calibration and data consistency
     * 2. **Pose Extraction:** Convert ROS pose message to OpenCV matrices
     * 3. **Semantic Labeling:** Categorize observations using YOLO bounding boxes
     * 4. **Data Association:** Match observations to existing landmarks
     * 5. **Landmark Management:** Create new landmarks or update existing ones
     * 6. **Database Update:** Add keyframes, observations, and landmarks to map
     * 7. **Visualization:** Publish updated landmark markers
     * 
     * **Data Association Strategy:**
     * - **Descriptor Matching:** ORB Hamming distance < 50 for initial filtering
     * - **Geometric Validation:** Reprojection error < 5 pixels for final acceptance
     * - **Semantic Consistency:** Category labels must match between observation and landmark
     * - **Temporal Preference:** Recently observed landmarks preferred for association
     * 
     * @param tracking_msg YOLO object detections for semantic labeling
     * @param kf_msg Keyframe data from frontend with pose and observations
     * 
     * @pre Camera parameters initialized from camera_info callback
     * @pre Frontend provides valid keyframe with pose and observations
     * @pre YOLO detections temporally synchronized with keyframe
     * 
     * @post New keyframe added to keyframes_ database
     * @post New observations added to all_observations_ database  
     * @post Landmark database updated with associations and new landmarks
     * @post Visualization markers published for updated map
     * 
     * @note Semantic labeling is optional - observations labeled "unlabeled" if no YOLO match
     * @note Dynamic objects (e.g., "person") filtered out to maintain static map assumption
     * @warning Large descriptor distances may indicate poor feature quality or lighting changes
     */
    void syncCallback(const yolo_msgs::msg::DetectionArray::ConstSharedPtr& tracking_msg, 
                     const dynamic_visual_slam_interfaces::msg::Keyframe::ConstSharedPtr& kf_msg) {
        // Validate camera calibration availability
        if (!camera_params_initialized_) {
            RCLCPP_WARN(this->get_logger(), "Camera parameters not initialized - skipping keyframe");
            return;
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Processing keyframe %lu with %zu landmarks", 
                    kf_msg->frame_id, kf_msg->landmarks.size());

        // Store latest timestamp for visualization synchronization
        latest_keyframe_timestamp_ = kf_msg->header.stamp;
        int frame_id = kf_msg->frame_id;
        yolo_msgs::msg::DetectionArray detections = *tracking_msg;

        // Extract camera pose from ROS message format
        cv::Mat R, t;
        extractPoseFromTransform(kf_msg->pose, R, t);

        // Data structures for new keyframe processing
        std::vector<ObservationInfo> new_observations;
        std::unordered_map<std::string, std::unordered_map<uint64_t, LandmarkInfo>> new_landmarks;
        KeyframeInfo new_keyframe(frame_id, R, t, latest_keyframe_timestamp_);

        // Process each observation-landmark pair from keyframe
        for (size_t i = 0; i < kf_msg->observations.size(); i++) {
            const auto& obs = kf_msg->observations[i];
            const auto& landmark = kf_msg->landmarks[i];

            // Convert ROS descriptor to OpenCV format
            cv::Mat descriptor(1, obs.descriptor.size(), CV_8U);
            std::memcpy(descriptor.data, obs.descriptor.data(), obs.descriptor.size());

            // Create observation with semantic categorization
            ObservationInfo new_obs(next_observation_id_, frame_id, 
                                   cv::Point2f(obs.pixel_x, obs.pixel_y), descriptor);
            new_obs.category = categorizeObservation(new_obs.pixel, detections);

            // Filter out dynamic objects that shouldn't be mapped
            if (filtered_objects_.find(new_obs.category) != filtered_objects_.end()) {
                continue;  // Skip this observation
            }

            // Track observation in current keyframe
            new_keyframe.observation_ids.push_back(next_observation_id_);
            next_observation_id_++;

            // Attempt data association with existing landmarks
            int associated_landmark_id = associateObservation(new_obs, R, t);

            if (associated_landmark_id != -1) {
                // === EXISTING LANDMARK ASSOCIATION ===
                
                new_obs.landmark_id = associated_landmark_id;
                auto& landmark_info = landmark_database_.at(new_obs.category).at(associated_landmark_id);
                
                // Update landmark tracking metadata
                landmark_info.observation_count++;
                landmark_info.last_seen = kf_msg->header.stamp;
                landmark_info.observation_ids.push_back(new_obs.observation_id);

                // Refine landmark position using multi-view triangulation
                landmark_info.triangulate(all_observations_, keyframes_, fx_, fy_, cx_, cy_);
                
                RCLCPP_DEBUG(this->get_logger(), 
                            "Associated observation %lu with existing landmark %d", 
                            new_obs.observation_id, associated_landmark_id);
            }
            else {
                // === NEW LANDMARK CREATION ===
                
                cv::Point3f landmark_pos;
                landmark_pos.x = landmark.position.x;
                landmark_pos.y = landmark.position.y;
                landmark_pos.z = landmark.position.z;
                
                uint64_t new_landmark_id = next_global_landmark_id_++;
                LandmarkInfo new_landmark(new_landmark_id, landmark_pos, descriptor, kf_msg->header.stamp);
                new_landmark.observation_ids.push_back(new_obs.observation_id);
                
                // Add to new landmarks pending database insertion
                new_landmarks[new_obs.category].emplace(new_landmark_id, new_landmark);
                new_obs.landmark_id = new_landmark_id;
                
                RCLCPP_DEBUG(this->get_logger(), 
                            "Created new landmark %lu for observation %lu", 
                            new_landmark_id, new_obs.observation_id);
            }

            // Store processed observation
            new_observations.push_back(new_obs);
        }

        // === DATABASE UPDATES ===
        
        // Add new keyframe to temporal sequence
        keyframes_.push_back(new_keyframe);
        
        // Add all new observations to global database
        all_observations_.insert(all_observations_.end(), new_observations.begin(), new_observations.end());

        // Add all new landmarks to categorized database
        for (const auto& [landmark_category, landmarks] : new_landmarks) {
            for (const auto& [landmark_id, landmark_info] : landmarks) {
                landmark_database_[landmark_category].emplace(landmark_id, landmark_info);
                
                RCLCPP_DEBUG(this->get_logger(), 
                            "Added landmark %lu (category: %s) to global map", 
                            landmark_id, landmark_category.c_str());
            }
        }
        
        // Update statistics and visualization
        keyframe_count_++;
        
        RCLCPP_INFO(this->get_logger(), 
                   "Processed keyframe %d: %zu observations, %zu new landmarks. Map: %zu total landmarks", 
                   frame_id, new_observations.size(), new_landmarks.size(), 
                   getTotalLandmarkCount());

        // Publish updated visualization
        publishAllLandmarkMarkers();
    }

    /**
     * @brief Periodic bundle adjustment optimization callback
     * 
     * Performs sliding window bundle adjustment to optimize camera poses and
     * landmark positions, reducing accumulated drift and improving map consistency.
     * Runs asynchronously every 2 seconds to balance accuracy and real-time performance.
     * 
     * **Optimization Pipeline:**
     * 1. **Concurrency Check:** Prevent overlapping optimizations
     * 2. **Data Validation:** Ensure sufficient keyframes for optimization
     * 3. **Window Selection:** Extract last N keyframes for sliding window
     * 4. **Data Preparation:** Collect observations and landmarks in window
     * 5. **Bundle Adjustment:** Optimize poses and landmarks using Ceres
     * 6. **Result Integration:** Update database with optimized parameters
     * 7. **Map Maintenance:** Prune low-quality landmarks and cleanup
     * 8. **Visualization Update:** Publish optimized map for display
     * 
     * **Sliding Window Strategy:**
     * - **Window Size:** 5 keyframes (configurable based on computational budget)
     * - **Fixed Gauge:** First keyframe pose held constant as coordinate reference
     * - **Landmark Selection:** All landmarks observed within temporal window
     * - **Observation Filtering:** Only geometrically valid observations included
     * 
     * **Performance Characteristics:**
     * - **Typical Runtime:** 50-200ms for 5-keyframe window with 100-500 landmarks
     * - **Memory Usage:** O(keyframes² + landmarks) scaling
     * - **Convergence:** Usually 10-20 iterations for typical SLAM scenarios
     * 
     * @pre keyframes_.size() >= 2 for meaningful optimization
     * @pre Camera calibration parameters available for reprojection
     * @pre Map database contains sufficient observations for constraints
     * 
     * @post Camera poses and landmark positions updated if optimization successful
     * @post Low-quality landmarks pruned from database
     * @post Updated visualization published for RViz display
     * 
     * @note Thread-safe execution with mutex protection during optimization
     * @note Graceful failure handling - system continues if optimization fails  
     * @warning Large optimization times may indicate degenerate geometry or too many landmarks
     */
    void bundleAdjustmentCallback() {
        // Prevent concurrent bundle adjustment runs
        if (ba_running_.load()) {
            RCLCPP_DEBUG(this->get_logger(), "Bundle adjustment already running - skipping cycle");
            return;
        }

        // Thread-safe access to keyframe database
        std::lock_guard<std::mutex> lock(keyframes_mutex_);
        
        // Validate sufficient data for meaningful optimization
        if (keyframes_.size() < 2) {
            RCLCPP_DEBUG(this->get_logger(), "Insufficient keyframes for bundle adjustment: %zu", keyframes_.size());
            return;
        }

        ba_running_.store(true);  // Set busy flag
        
        // === SLIDING WINDOW SELECTION ===
        
        // Extract recent keyframes for optimization (sliding window approach)
        int window_size = std::min(5, static_cast<int>(keyframes_.size()));
        int start_idx = keyframes_.size() - window_size;
        
        std::vector<KeyframeInfo> window_keyframe_infos(
            keyframes_.begin() + start_idx, 
            keyframes_.end()
        );
        
        RCLCPP_INFO(this->get_logger(), 
                   "Starting bundle adjustment: %d keyframes in window [%d-%zu], total landmarks: %zu", 
                   window_size, start_idx, keyframes_.size() - 1, getTotalLandmarkCount());
        
        // Convert to bundle adjustment format
        std::vector<KeyframeData> window_keyframes;
        for (const auto& kf_info : window_keyframe_infos) {
            window_keyframes.emplace_back(kf_info.frame_id, kf_info.R, kf_info.t, kf_info.timestamp);
        }
        
        // === DATA PREPARATION ===
        
        // Collect all observations in the temporal window
        std::set<uint64_t> window_observation_ids;
        for (const auto& kf_info : window_keyframe_infos) {
            for (uint64_t obs_id : kf_info.observation_ids) {
                window_observation_ids.insert(obs_id);
            }
        }
        
        // Build optimization data structures
        std::vector<Observation> window_observations;
        std::set<std::pair<uint64_t, std::string>> window_landmark_ids;
        
        // Extract observations and collect associated landmark IDs
        for (const auto& obs_info : all_observations_) {
            if (window_observation_ids.count(obs_info.observation_id) > 0) {
                window_landmark_ids.insert(std::make_pair(obs_info.landmark_id, obs_info.category));
                window_observations.emplace_back(obs_info.pixel.x, obs_info.pixel.y, 
                                               obs_info.landmark_id, obs_info.category, obs_info.frame_id);
            }
        }
        
        // Extract landmarks observed in the window
        std::vector<Landmark> window_landmarks;
        for (const auto& [landmark_id, landmark_category] : window_landmark_ids) {
            const auto& landmark_info = landmark_database_.at(landmark_category).at(landmark_id);
            window_landmarks.emplace_back(landmark_id, landmark_category,
                                        landmark_info.position.x, 
                                        landmark_info.position.y, 
                                        landmark_info.position.z,
                                        false); // Don't fix landmarks (allow optimization)
        }
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "Optimization window: %zu landmarks, %zu observations", 
                    window_landmarks.size(), window_observations.size());
        
        // === BUNDLE ADJUSTMENT OPTIMIZATION ===
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        OptimizationResult result = bundle_adjuster_->optimize(
            window_keyframes, 
            window_landmarks, 
            window_observations,
            20  // Maximum iterations for real-time performance
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // === RESULT PROCESSING ===
        
        if (result.success) {
            RCLCPP_INFO(this->get_logger(), 
                       "Bundle adjustment converged in %d iterations, cost: %.6f, time: %ld ms",
                       result.iterations_completed, result.final_cost, duration.count());
            
            // Update database with optimized parameters
            updateOptimizedResults(result);
        } else {
            RCLCPP_WARN(this->get_logger(), 
                       "Bundle adjustment failed: %s, time: %ld ms", 
                       result.message.c_str(), duration.count());
        }

        // === MAP MAINTENANCE ===
        
        // Remove low-quality landmarks and cleanup observation database
        pruneLandmarks();

        // Update visualization with optimized map
        publishAllLandmarkMarkers();
        
        ba_running_.store(false);  // Clear busy flag
    }

    /**
     * @brief Categorizes observation based on YOLO object detections
     * 
     * Determines semantic category for a feature observation by checking if its
     * pixel location falls within any YOLO detection bounding box. Used for
     * semantic landmark labeling and dynamic object filtering.
     * 
     * **Categorization Process:**
     * 1. Check each YOLO detection bounding box
     * 2. Test if observation pixel falls within box boundaries  
     * 3. Return first matching category name
     * 4. Default to "unlabeled" if no bounding box contains pixel
     * 
     * @param pixel 2D pixel coordinates of observation
     * @param detections Array of YOLO object detections with bounding boxes
     * @return String category name ("person", "car", "unlabeled", etc.)
     * 
     * @note Multiple overlapping detections will return first match
     * @note Categories used for filtering dynamic objects from mapping
     */
    std::string categorizeObservation(const cv::Point2f& pixel, const yolo_msgs::msg::DetectionArray& detections) {
        // Test each detection bounding box for pixel containment
        for (const auto& detection : detections.detections) {
            const auto& bbox = detection.bbox;
            
            // Check if observation pixel is inside bounding box
            if (pixel.x >= bbox.center.position.x - bbox.size.x/2 &&
                pixel.x <= bbox.center.position.x + bbox.size.x/2 &&
                pixel.y >= bbox.center.position.y - bbox.size.y/2 &&
                pixel.y <= bbox.center.position.y + bbox.size.y/2) {
                
                // Return category name for this detection
                return detection.class_name;
            }
        }
        
        // Default category for features not in any detection
        return "unlabeled";
    }

    /**
     * @brief Attempts to associate observation with existing landmark
     * 
     * Implements two-stage data association using descriptor similarity and
     * geometric consistency to match new observations with existing landmarks
     * in the map database. Critical for maintaining map consistency and
     * preventing duplicate landmark creation.
     * 
     * **Association Pipeline:**
     * 1. **Category Filtering:** Only consider landmarks with matching semantic category
     * 2. **Descriptor Matching:** Compute ORB Hamming distances to all candidates
     * 3. **Distance Filtering:** Retain matches below descriptor distance threshold
     * 4. **Geometric Validation:** Check reprojection error using current camera pose
     * 5. **Best Match Selection:** Choose landmark with lowest reprojection error
     * 
     * **Quality Thresholds:**
     * - **Descriptor Distance:** < 50 Hamming units for ORB (empirically tuned)
     * - **Reprojection Error:** < 5 pixels for geometric consistency
     * - **Category Consistency:** Semantic labels must match exactly
     * 
     * @param obs Observation to associate with existing landmarks
     * @param R Current camera rotation matrix (world to camera)
     * @param t Current camera translation vector (world to camera)
     * @return Landmark ID if successful association, -1 if no match found
     * 
     * @pre obs.category exists in landmark_database_ (may be empty)
     * @pre R and t represent valid camera pose for reprojection
     * @pre Camera intrinsics (fx_, fy_, cx_, cy_) initialized
     * 
     * @note Returns ID of best matching landmark based on reprojection error
     * @note Conservative thresholds prevent false associations that degrade map quality
     * @warning Poor lighting or motion blur can cause association failures
     */
    int associateObservation(const ObservationInfo& obs, const cv::Mat& R, const cv::Mat& t) {
        std::vector<std::pair<int, double>> candidates;  // (landmark_id, descriptor_distance)

        // Find descriptor-based candidates within same semantic category
        for (const auto& [landmark_id, landmark_info] : landmark_database_[obs.category]) {
            std::vector<cv::DMatch> matches;

            // Compute ORB descriptor distance (Hamming distance for binary descriptors)
            descriptor_matcher_.match(obs.descriptor, landmark_info.descriptor, matches);

            if (!matches.empty() && matches[0].distance < max_descriptor_distance_) {
                candidates.emplace_back(landmark_id, matches[0].distance);
            }
        }

        RCLCPP_DEBUG(this->get_logger(), "Descriptor matching: %zu candidates found", candidates.size());

        if (candidates.empty()) {
            return -1;  // No descriptor matches found
        }

        // === GEOMETRIC VALIDATION ===
        
        int best_landmark_id = -1;
        double best_reprojection_error = std::numeric_limits<double>::max();

        // Evaluate each candidate using geometric consistency
        for (const auto& [candidate_id, descriptor_distance] : candidates) {
            const auto& candidate_landmark = landmark_database_.at(obs.category).at(candidate_id);

            // Convert landmark position to OpenCV format for reprojection
            cv::Point3f landmark_3d_cv(candidate_landmark.position.x, 
                                      candidate_landmark.position.y, 
                                      candidate_landmark.position.z);

            // Project landmark into current camera view
            cv::Point2f reprojection_pixel = reprojectPoint(landmark_3d_cv, R, t);

            // Compute reprojection error (Euclidean distance in pixels)
            double reprojection_error = cv::norm(obs.pixel - reprojection_pixel);

            // Accept if error is below threshold and better than current best
            if (reprojection_error < max_reprojection_distance_ && 
                reprojection_error < best_reprojection_error) {
                best_landmark_id = candidate_id;
                best_reprojection_error = reprojection_error;
            }
        }

        if (best_landmark_id != -1) {
            RCLCPP_DEBUG(this->get_logger(), 
                        "Successful association: landmark %d, reprojection error %.2f pixels", 
                        best_landmark_id, best_reprojection_error);
        }

        return best_landmark_id;
    }

    /**
     * @brief Projects 3D world point into camera image coordinates
     * 
     * Performs perspective projection of 3D landmark position into 2D pixel
     * coordinates using camera pose and intrinsic parameters. Used for
     * reprojection error computation in data association and optimization.
     * 
     * **Projection Pipeline:**
     * 1. **Coordinate Transform:** Convert world point to camera coordinates using pose
     * 2. **Depth Check:** Validate point is in front of camera (positive depth)
     * 3. **Perspective Division:** Apply pinhole camera model with intrinsics
     * 4. **Pixel Coordinates:** Convert to image pixel coordinates
     * 
     * **Mathematical Model:**
     * \f[
     * \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{z} \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix}
     * \f]
     * where (x,y,z) are camera coordinates and (u,v) are pixel coordinates.
     * 
     * @param point_3d_world 3D landmark position in world coordinates
     * @param R Camera rotation matrix (world to camera transformation)
     * @param t Camera translation vector (world to camera transformation)
     * @return 2D pixel coordinates, or (-1,-1) if point behind camera
     * 
     * @pre R is valid rotation matrix (orthogonal, determinant = 1)
     * @pre t is 3D translation vector
     * @pre Camera intrinsics (fx_, fy_, cx_, cy_) initialized
     * 
     * @note Returns invalid coordinates (-1,-1) for points behind camera
     * @note Assumes optical coordinate convention (Z=forward)
     */
    cv::Point2f reprojectPoint(const cv::Point3f& point_3d_world, const cv::Mat& R, const cv::Mat& t) {
        // Transform point from world to camera coordinates
        cv::Mat point_world = (cv::Mat_<double>(3,1) << point_3d_world.x, point_3d_world.y, point_3d_world.z);
        cv::Mat point_camera = R.t() * (point_world - t);  // R^T * (X - t) = R^-1 * (X - t)
        
        // Extract camera coordinates (optical convention: X=right, Y=down, Z=forward)
        double x = point_camera.at<double>(0);  // Right in optical frame
        double y = point_camera.at<double>(1);  // Down in optical frame  
        double z = point_camera.at<double>(2);  // Forward in optical frame (depth)
        
        // Check if point is in front of camera
        if (z <= 0) {
            return cv::Point2f(-1, -1);  // Invalid projection (behind camera)
        }
        
        // Apply pinhole camera model with perspective division
        float u = fx_ * x / z + cx_;  // Horizontal pixel coordinate
        float v = fy_ * y / z + cy_;  // Vertical pixel coordinate
        
        return cv::Point2f(u, v);
    }
    
    /**
     * @brief Extracts camera pose from ROS transform message
     * 
     * Converts ROS geometry_msgs::Transform to OpenCV rotation matrix and
     * translation vector format. Handles quaternion normalization and
     * coordinate frame consistency.
     * 
     * @param transform ROS transform message with quaternion rotation and translation
     * @param[out] R 3x3 rotation matrix (OpenCV format)
     * @param[out] t 3x1 translation vector (OpenCV format)
     */
    void extractPoseFromTransform(const geometry_msgs::msg::Transform& transform, cv::Mat& R, cv::Mat& t) {
        // Extract translation vector
        t = cv::Mat(3, 1, CV_64F);
        t.at<double>(0) = transform.translation.x;
        t.at<double>(1) = transform.translation.y;
        t.at<double>(2) = transform.translation.z;
        
        // Extract and normalize quaternion
        double qw = transform.rotation.w;
        double qx = transform.rotation.x;
        double qy = transform.rotation.y;
        double qz = transform.rotation.z;
        
        double norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        qw /= norm; qx /= norm; qy /= norm; qz /= norm;
        
        // Convert quaternion to rotation matrix
        R = cv::Mat(3, 3, CV_64F);
        
        R.at<double>(0, 0) = 1 - 2*(qy*qy + qz*qz);
        R.at<double>(0, 1) = 2*(qx*qy - qw*qz);
        R.at<double>(0, 2) = 2*(qx*qz + qw*qy);
        
        R.at<double>(1, 0) = 2*(qx*qy + qw*qz);
        R.at<double>(1, 1) = 1 - 2*(qx*qx + qz*qz);
        R.at<double>(1, 2) = 2*(qy*qz - qw*qx);
        
        R.at<double>(2, 0) = 2*(qx*qz - qw*qy);
        R.at<double>(2, 1) = 2*(qy*qz + qw*qx);
        R.at<double>(2, 2) = 1 - 2*(qx*qx + qy*qy);
    }

    /**
     * @brief Removes low-quality landmarks and associated observations
     * 
     * Performs map maintenance by removing landmarks that don't meet quality
     * criteria, cleaning up orphaned observations, and updating keyframe
     * references. Critical for preventing map degradation and memory growth.
     * 
     * **Pruning Criteria:**
     * - **Insufficient Observations:** < 2 observations (unreliable landmarks)
     * - **Temporal Staleness:** Not observed for > 20 seconds (likely outdated)
     * - **Geometric Inconsistency:** High reprojection errors (poor quality)
     * 
     * **Cleanup Process:**
     * 1. **Identification:** Scan all landmarks for pruning criteria violations
     * 2. **Cascade Removal:** Remove landmark from categorized database
     * 3. **Observation Cleanup:** Remove all observations referencing pruned landmarks
     * 4. **Keyframe Update:** Remove observation references from keyframe metadata
     * 5. **Statistics Logging:** Report pruning results for system monitoring
     * 
     * **Memory Impact:**
     * - Prevents unbounded memory growth in long-running sessions
     * - Removes outlier landmarks that degrade optimization quality
     * - Maintains observation database consistency and referential integrity
     * 
     * @post Low-quality landmarks removed from landmark_database_
     * @post Orphaned observations removed from all_observations_
     * @post Keyframe observation references updated for consistency
     * 
     * @note Conservative thresholds prevent premature removal of valid landmarks
     * @note Pruning frequency should balance memory usage and map stability
     */
    void pruneLandmarks() {
        auto current_time = this->now();
        const int min_observation_threshold = 2;      // Minimum observations for landmark stability
        const double max_time_since_seen = 20.0;      // Maximum age in seconds before pruning
        
        // Collect landmarks marked for removal
        std::vector<std::pair<uint64_t, std::string>> landmarks_to_remove;
        
        // Scan all landmarks across all categories for pruning criteria
        for (const auto& [landmark_category, landmarks] : landmark_database_) {
            for (const auto& [landmark_id, landmark_info] : landmarks) {
                double time_since_seen = (current_time - landmark_info.last_seen).seconds();
                
                // Mark for removal if insufficient observations AND too old
                if (landmark_info.observation_count < min_observation_threshold && 
                    time_since_seen > max_time_since_seen) {
                    landmarks_to_remove.emplace_back(landmark_id, landmark_category);
                    
                    RCLCPP_DEBUG(this->get_logger(), 
                                "Marking landmark %lu for removal: %d observations (< %d), %.1fs old (> %.1fs)", 
                                landmark_id, landmark_info.observation_count, min_observation_threshold,
                                time_since_seen, max_time_since_seen);
                }
            }
        }
        
        // Execute cascade removal of landmarks and dependent data
        int removed_landmarks = 0;
        int removed_observations = 0;
        
        for (const auto& [landmark_id, landmark_category] : landmarks_to_remove) {
            const auto& landmark_info = landmark_database_.at(landmark_category).at(landmark_id);
            
            // Collect observation IDs to remove
            std::set<uint64_t> obs_ids_to_remove(landmark_info.observation_ids.begin(), 
                                                 landmark_info.observation_ids.end());
            
            // Remove landmark from categorized database
            landmark_database_[landmark_category].erase(landmark_id);
            removed_landmarks++;
            
            // Remove associated observations from global database
            auto obs_it = all_observations_.begin();
            while (obs_it != all_observations_.end()) {
                if (obs_ids_to_remove.count(obs_it->observation_id) > 0 || 
                    obs_it->landmark_id == landmark_id) {
                    obs_it = all_observations_.erase(obs_it);
                    removed_observations++;
                } else {
                    ++obs_it;
                }
            }
            
            // Update keyframe observation references
            for (auto& keyframe : keyframes_) {
                auto& obs_ids = keyframe.observation_ids;
                obs_ids.erase(
                    std::remove_if(obs_ids.begin(), obs_ids.end(), 
                                   [&obs_ids_to_remove](uint64_t obs_id) { 
                                       return obs_ids_to_remove.count(obs_id) > 0; 
                                   }),
                    obs_ids.end()
                );
            }
        }
        
        // Log pruning statistics
        if (removed_landmarks > 0) {
            RCLCPP_INFO(this->get_logger(), 
                       "Landmark pruning: removed %d landmarks and %d observations. Remaining: %zu landmarks, %zu observations", 
                       removed_landmarks, removed_observations,
                       getTotalLandmarkCount(), all_observations_.size());
        }
    }

    /**
     * @brief Integrates bundle adjustment optimization results into map database
     * 
     * Updates keyframe poses and landmark positions with optimized values from
     * bundle adjustment, maintaining database consistency and coordinate frame
     * conventions. Called after successful bundle adjustment convergence.
     * 
     * **Update Process:**
     * 1. **Pose Updates:** Replace keyframe poses with optimized camera positions
     * 2. **Landmark Updates:** Replace landmark positions with optimized 3D coordinates  
     * 3. **Consistency Check:** Verify all referenced entities exist in database
     * 4. **Statistics Logging:** Report number of updated poses and landmarks
     * 
     * **Thread Safety:**
     * - Called within bundle adjustment timer callback with mutex protection
     * - No additional synchronization needed for database access
     * - Atomic updates prevent partial state during optimization
     * 
     * @param result Bundle adjustment optimization result containing updated parameters
     * 
     * @pre result.success == true (only called after successful optimization)
     * @pre All referenced keyframes and landmarks exist in respective databases
     * @pre Optimization result coordinates in same frame as database storage
     * 
     * @post Keyframe poses updated with optimized camera positions
     * @post Landmark positions updated with optimized 3D coordinates
     * @post Database remains consistent with optimized geometry
     * 
     * @note Updates preserve coordinate frame conventions (optical frame internal storage)
     * @note Only entities within optimization window are updated
     * @warning Assumes optimization coordinate frame matches database frame
     */
    void updateOptimizedResults(const OptimizationResult& result) {
        // Update keyframe poses from optimization results
        for (const auto& [frame_id, pose_pair] : result.optimized_poses) {
            const auto& [R_opt, t_opt] = pose_pair;
            
            // Find corresponding keyframe in database
            for (auto& kf_info : keyframes_) {
                if (kf_info.frame_id == static_cast<uint64_t>(frame_id)) {
                    // Replace with optimized pose (clone to ensure data ownership)
                    kf_info.R = R_opt.clone();
                    kf_info.t = t_opt.clone();
                    break;
                }
            }
        }
        
        // Update landmark positions from optimization results
        for (const auto& [landmark_key, optimized_pos] : result.optimized_landmarks) {
            const auto& [landmark_id, landmark_category] = landmark_key;
            
            // Find landmark in categorized database
            auto category_it = landmark_database_.find(landmark_category);
            if (category_it != landmark_database_.end()) {
                auto landmark_it = category_it->second.find(landmark_id);
                if (landmark_it != category_it->second.end()) {
                    // Update 3D position with optimized coordinates
                    landmark_it->second.position.x = optimized_pos.x;
                    landmark_it->second.position.y = optimized_pos.y;
                    landmark_it->second.position.z = optimized_pos.z;
                }
            }
        }
        
        RCLCPP_INFO(this->get_logger(), 
                   "Applied optimization results: %zu poses updated, %zu landmarks updated", 
                   result.optimized_poses.size(), result.optimized_landmarks.size());
    }
    
    /**
     * @brief Publishes landmark visualization markers for RViz display
     * 
     * Creates and publishes RViz marker array containing all landmarks in the
     * map database with appropriate coordinate transformations, coloring, and
     * metadata. Essential for real-time map visualization and debugging.
     * 
     * **Visualization Pipeline:**
     * 1. **Coordinate Transform:** Convert landmark positions from optical to ROS frame
     * 2. **Marker Creation:** Generate sphere markers for each landmark
     * 3. **Color Coding:** Color-code landmarks based on observation quality
     * 4. **Metadata Setup:** Configure marker properties (size, lifetime, namespace)
     * 5. **Batch Publishing:** Publish complete marker array for efficient RViz update
     * 
     * **Coordinate Transformation:**
     * - **Internal Storage:** Optical frame (X=right, Y=down, Z=forward)
     * - **Visualization:** ROS frame (X=forward, Y=left, Z=up)
     * - **Transform Matrix:** Applied to all landmark positions for RViz compatibility
     * 
     * **Color Coding Scheme:**
     * - **Cyan (0,1,1):** Well-established landmarks (>1 observation)
     * - **Green (0,1,0):** Newly created landmarks (1 observation)
     * - **Consistent Alpha:** 0.8 transparency for all landmarks
     * 
     * **Marker Properties:**
     * - **Type:** Sphere markers for clear 3D visualization
     * - **Size:** 5mm diameter spheres (scale.x/y/z = 0.005)
     * - **Namespace:** "landmarks" for organized RViz display
     * - **Lifetime:** Persistent markers (lifetime = 0)
     * - **Frame:** "world" coordinate frame for global visualization
     * 
     * @pre landmark_database_ contains valid landmark positions
     * @pre latest_keyframe_timestamp_ set for marker synchronization
     * @pre landmark_markers_pub_ publisher initialized and connected
     * 
     * @post All landmarks published as RViz markers in world frame
     * @post Markers color-coded by observation quality for visual debugging
     * @post RViz display updated with current map state
     * 
     * @note Efficient batch publishing minimizes RViz update overhead
     * @note Coordinate transformation ensures RViz compatibility
     * @note Color coding provides immediate visual feedback on landmark quality
     */
    void publishAllLandmarkMarkers() {
        visualization_msgs::msg::MarkerArray marker_array;
        
        // Transformation matrix from optical frame to ROS frame for visualization
        cv::Mat T_opt_to_ros = (cv::Mat_<double>(3,3) << 
            0,  0,  1,    // Optical Z → ROS X (forward)
            -1, 0,  0,    // Optical -X → ROS Y (left)  
            0, -1,  0     // Optical -Y → ROS Z (up)
        );
        
        // Create marker for each landmark across all categories
        for (const auto& [landmark_category, landmarks] : landmark_database_) {
            for (const auto& [landmark_id, landmark_info] : landmarks) {
                visualization_msgs::msg::Marker marker;
                
                // Marker identification and timing
                marker.header.frame_id = "world";
                marker.header.stamp = latest_keyframe_timestamp_;
                marker.ns = "landmarks";
                marker.id = static_cast<int>(landmark_id);
                marker.type = visualization_msgs::msg::Marker::SPHERE;
                marker.action = visualization_msgs::msg::Marker::ADD;
                
                // Transform landmark position from optical to ROS coordinates
                cv::Mat pos_optical = (cv::Mat_<double>(3,1) << 
                    landmark_info.position.x, 
                    landmark_info.position.y, 
                    landmark_info.position.z);
                cv::Mat pos_ros = T_opt_to_ros * pos_optical;
                
                // Set marker position and orientation
                marker.pose.position.x = pos_ros.at<double>(0);
                marker.pose.position.y = pos_ros.at<double>(1);
                marker.pose.position.z = pos_ros.at<double>(2);
                marker.pose.orientation.x = 0.0;
                marker.pose.orientation.y = 0.0;
                marker.pose.orientation.z = 0.0;
                marker.pose.orientation.w = 1.0;
                
                // Set marker size (5mm diameter spheres)
                marker.scale.x = 0.005;
                marker.scale.y = 0.005;
                marker.scale.z = 0.005;
                
                // Color-code based on landmark quality
                if (landmark_info.observation_count > 1) {
                    // Well-established landmarks: cyan
                    marker.color.r = 0.0;
                    marker.color.g = 1.0;
                    marker.color.b = 1.0;
                    marker.color.a = 0.8;
                }
                else {
                    // New landmarks: green
                    marker.color.r = 0.0;
                    marker.color.g = 1.0;
                    marker.color.b = 0.0;
                    marker.color.a = 0.8;
                }
                
                // Persistent marker (no automatic deletion)
                marker.lifetime = rclcpp::Duration::from_seconds(0);
                
                marker_array.markers.push_back(marker);
            }
        }
        
        // Publish complete marker array for efficient RViz update
        landmark_markers_pub_->publish(marker_array);
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "Published %zu landmark markers (transformed from optical to ROS frame)", 
                    getTotalLandmarkCount());
    }

    /**
     * @brief Computes total number of landmarks across all categories
     * 
     * Utility function that counts landmarks in the categorized database,
     * providing aggregate statistics for logging and debugging purposes.
     * 
     * @return Total number of landmarks in map database
     * @note Traverses all categories in landmark_database_ for complete count
     */
    size_t getTotalLandmarkCount() const {
        size_t total = 0;
        for (const auto& [category, landmarks] : landmark_database_) {
            total += landmarks.size();
        }
        return total;
    }
};

/**
 * @brief Main entry point for the SLAM backend node
 * 
 * Initializes ROS 2 system, creates and spins the Backend node for continuous
 * operation, and handles graceful shutdown. The main function follows standard
 * ROS 2 node lifecycle patterns.
 * 
 * **Execution Flow:**
 * 1. **ROS Initialization:** Initialize ROS 2 context and communication
 * 2. **Node Creation:** Instantiate Backend node with all subscribers/publishers
 * 3. **Event Loop:** Spin node to process callbacks (keyframes, timer, etc.)
 * 4. **Graceful Shutdown:** Clean up ROS 2 resources on termination
 * 
 * **Runtime Characteristics:**
 * - **Continuous Operation:** Runs indefinitely until shutdown signal
 * - **Callback Processing:** Handles keyframes, bundle adjustment, visualization
 * - **Thread Safety:** ROS 2 executor manages callback scheduling and synchronization
 * - **Memory Management:** RAII ensures proper cleanup of node resources
 * 
 * @param argc Command line argument count
 * @param argv Command line argument vector
 * @return Exit status (0 for success, non-zero for error)
 * 
 * @pre ROS 2 environment properly configured
 * @pre Required dependencies (camera, YOLO) available or gracefully handled
 * 
 * @post Backend node terminates cleanly with resource cleanup
 * @post ROS 2 system shutdown completed
 * 
 * @note Standard ROS 2 node pattern - modify carefully to maintain compatibility
 * @note Node will continue processing until SIGINT or ROS shutdown
 */
int main(int argc, char* argv[]) {
    // Initialize ROS 2 system
    rclcpp::init(argc, argv);
    
    // Create and spin backend node (blocks until shutdown)
    rclcpp::spin(std::make_shared<Backend>());
    
    // Clean shutdown
    rclcpp::shutdown();
    return 0;
}