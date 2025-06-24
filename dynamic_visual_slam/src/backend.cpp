#include <opencv2/opencv.hpp>
#include <memory>
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
#include <Eigen/Geometry>

class Backend : public rclcpp::Node 
{
public:
    Backend() : Node("backend") {
        // subscribe to keyframe topic
        rclcpp::QoS qos = rclcpp::QoS(30);

        // Initialize bundle adjuster with default camera parameters
        // These will be updated when we receive camera info
        bundle_adjuster_ = std::make_unique<SlidingWindowBA>(10, fx_, fy_, cx_, cy_, 0.5);
        
        // Track global landmark IDs
        next_global_landmark_id_ = 0;
        
        keyframe_sub_ = this->create_subscription<dynamic_visual_slam_interfaces::msg::Keyframe>(
            "/frontend/keyframe", qos, 
            std::bind(&Backend::keyframeCallback, this, std::placeholders::_1));
            
        // Subscribe to camera info to get proper camera parameters
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/color/camera_info", qos,
            std::bind(&Backend::cameraInfoCallback, this, std::placeholders::_1));
            
        // Create TF broadcaster for optimized poses
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        
        // Create marker publishers for landmark visualization
        landmark_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/backend/landmark_markers", qos);
            
        // Parameters
        min_observations_for_landmark_ = 2;
        max_reprojection_error_ = 2.0;
        bundle_adjustment_frequency_ = 10; // Run BA every 10 keyframes
        keyframe_count_ = 0;
        camera_params_initialized_ = false;
        
        // Initialize landmark map clearing flag
        map_cleared_ = false;
        
        RCLCPP_INFO(this->get_logger(), "Backend initialized - keeping all landmarks for mapping");
    }

private:
    std::unique_ptr<SlidingWindowBA> bundle_adjuster_;
    
    rclcpp::Subscription<dynamic_visual_slam_interfaces::msg::Keyframe>::SharedPtr keyframe_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    
    // Publishers
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmark_markers_pub_;
    
    // Store latest timestamp for TF broadcasting
    rclcpp::Time latest_keyframe_timestamp_;
    
    // Camera parameters
    bool camera_params_initialized_;
    double fx_, fy_, cx_, cy_;
    
    // Landmark management - persistent mapping
    uint64_t next_global_landmark_id_;
    std::unordered_map<uint64_t, uint64_t> temp_to_global_landmark_map_;
    
    // Persistent landmark storage for mapping
    std::unordered_map<uint64_t, geometry_msgs::msg::Point> all_landmarks_;
    std::unordered_map<uint64_t, int> landmark_observation_count_;
    std::unordered_map<uint64_t, rclcpp::Time> landmark_first_seen_;
    std::vector<geometry_msgs::msg::Pose> keyframe_poses_;
    
    // Map management
    bool map_cleared_;
    
    // Parameters
    int min_observations_for_landmark_;
    double max_reprojection_error_;
    int bundle_adjustment_frequency_;
    int keyframe_count_;

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (!camera_params_initialized_) {
            fx_ = msg->k[0];  // K[0,0]
            fy_ = msg->k[4];  // K[1,1] 
            cx_ = msg->k[2];  // K[0,2]
            cy_ = msg->k[5];  // K[1,2]
            
            bundle_adjuster_ = std::make_unique<SlidingWindowBA>(10, fx_, fy_, cx_, cy_);
            camera_params_initialized_ = true;
            
            RCLCPP_INFO(this->get_logger(), "Camera parameters initialized: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f", 
                        fx_, fy_, cx_, cy_);
        }
    }

    void keyframeCallback(const dynamic_visual_slam_interfaces::msg::Keyframe::ConstSharedPtr& msg) {
        if (!camera_params_initialized_) {
            RCLCPP_WARN(this->get_logger(), "Camera parameters not yet initialized, skipping keyframe");
            return;
        }
        
        // RCLCPP_INFO(this->get_logger(), "Processing keyframe %lu with %zu landmarks (Total map size: %zu)", 
        //             msg->frame_id, msg->landmarks.size(), all_landmarks_.size());

        keyframe_count_++;

        latest_keyframe_timestamp_ = msg->header.stamp;

        cv::Mat R, t;
        extractPoseFromTransform(msg->pose, R, t);

        geometry_msgs::msg::Pose keyframe_pose;
        keyframe_pose.position.x = msg->pose.translation.x;
        keyframe_pose.position.y = msg->pose.translation.y;
        keyframe_pose.position.z = msg->pose.translation.z;
        keyframe_pose.orientation = msg->pose.rotation;
        keyframe_poses_.push_back(keyframe_pose);
        
        int frame_id = bundle_adjuster_->addFrame(R, t);
        
        processLandmarksAndObservations(msg, frame_id, R, t);

        RCLCPP_INFO(this->get_logger(), "Running bundle adjustment...");
        auto start = std::chrono::high_resolution_clock::now();
        
        bundle_adjuster_->optimize(3);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        RCLCPP_INFO(this->get_logger(), "Bundle adjustment completed in %ld ms", duration.count());
        
        broadcastOptimizedTransform();
        logOptimizedPose();
        
        // Republish landmarks after optimization (positions may have been refined)
        publishAllLandmarkMarkers();
    }
    
    void extractPoseFromTransform(const geometry_msgs::msg::Transform& transform, cv::Mat& R, cv::Mat& t) {
        t = cv::Mat(3, 1, CV_64F);
        t.at<double>(0) = transform.translation.x;
        t.at<double>(1) = transform.translation.y;
        t.at<double>(2) = transform.translation.z;
        
        double qw = transform.rotation.w;
        double qx = transform.rotation.x;
        double qy = transform.rotation.y;
        double qz = transform.rotation.z;
        
        R = cv::Mat(3, 3, CV_64F);
        
        double norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        qw /= norm; qx /= norm; qy /= norm; qz /= norm;
        
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
    
    void processLandmarksAndObservations(const dynamic_visual_slam_interfaces::msg::Keyframe::ConstSharedPtr& msg, int frame_id, const cv::Mat& R, const cv::Mat& t) {
        for (size_t i = 0; i < msg->landmarks.size(); i++) {
            const auto& landmark = msg->landmarks[i];
            const auto& observation = msg->observations[i];
            
            if (landmark.position.z <= 0.1 || landmark.position.z > 10.0) {
                continue; // Skip landmarks with invalid depth
            }
            
            uint64_t temp_landmark_id = landmark.landmark_id;
            uint64_t global_landmark_id;
            
            if (landmark.is_new) {
                global_landmark_id = next_global_landmark_id_++;
                temp_to_global_landmark_map_[temp_landmark_id] = global_landmark_id;

                cv::Mat landmark_optical = (cv::Mat_<double>(3,1) << landmark.position.x, landmark.position.y, landmark.position.z);
                cv::Mat landmark_ros_camera = (cv::Mat_<double>(3,1) << 
                    landmark_optical.at<double>(2),   // X_ros = Z_optical (forward)
                    -landmark_optical.at<double>(0),  // Y_ros = -X_optical (left)
                    -landmark_optical.at<double>(1)   // Z_ros = -Y_optical (up)
                );

                cv::Mat landmark_world = R * landmark_ros_camera + t;

                geometry_msgs::msg::Point landmark_pos;
                landmark_pos.x = landmark_world.at<double>(0);
                landmark_pos.y = landmark_world.at<double>(1);
                landmark_pos.z = landmark_world.at<double>(2);
                all_landmarks_[global_landmark_id] = landmark_pos;
                
                RCLCPP_DEBUG(this->get_logger(), "Landmark %lu: Optical [%.3f,%.3f,%.3f] -> ROS [%.3f,%.3f,%.3f] -> World [%.3f,%.3f,%.3f]",
                            global_landmark_id,
                            landmark.position.x, landmark.position.y, landmark.position.z,
                            landmark_ros_camera.at<double>(0), landmark_ros_camera.at<double>(1), landmark_ros_camera.at<double>(2),
                            landmark_pos.x, landmark_pos.y, landmark_pos.z);
                
                landmark_observation_count_[global_landmark_id] = 1;
                landmark_first_seen_[global_landmark_id] = latest_keyframe_timestamp_;
                
                bundle_adjuster_->addObservation(
                    frame_id,
                    observation.pixel_x, observation.pixel_y,
                    landmark.position.x, landmark.position.y, landmark.position.z
                );
                
                RCLCPP_DEBUG(this->get_logger(), "Added new landmark %lu (map size: %zu)", 
                            global_landmark_id, all_landmarks_.size());
            } else {
                auto it = temp_to_global_landmark_map_.find(temp_landmark_id);
                if (it != temp_to_global_landmark_map_.end()) {
                    global_landmark_id = it->second;
                    
                    landmark_observation_count_[global_landmark_id]++;
                    
                    bundle_adjuster_->addObservation(
                        frame_id, static_cast<int>(global_landmark_id),
                        observation.pixel_x, observation.pixel_y
                    );
                    
                    RCLCPP_DEBUG(this->get_logger(), "Added observation for existing landmark %lu (observed %d times)", 
                                global_landmark_id, landmark_observation_count_[global_landmark_id]);
                }
            }
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Processed %zu landmarks for frame %d", 
                    msg->landmarks.size(), frame_id);
    }
    
    void publishAllLandmarkMarkers() {
        visualization_msgs::msg::MarkerArray marker_array;
        
        if (!map_cleared_) {
            visualization_msgs::msg::Marker delete_all_marker;
            delete_all_marker.header.frame_id = "world";
            delete_all_marker.header.stamp = latest_keyframe_timestamp_;
            delete_all_marker.ns = "landmarks";
            delete_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
            marker_array.markers.push_back(delete_all_marker);
            map_cleared_ = true;
        }
        
        for (const auto& [landmark_id, position] : all_landmarks_) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = latest_keyframe_timestamp_;
            marker.ns = "landmarks";
            marker.id = static_cast<int>(landmark_id);
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            marker.pose.position = position;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            
            marker.scale.x = 0.005;
            marker.scale.y = 0.005;
            marker.scale.z = 0.005;
            
            int obs_count = landmark_observation_count_[landmark_id];
            if (obs_count == 1) {
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                marker.color.a = 0.8;
            } else if (obs_count < 5) {
                marker.color.r = 1.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                marker.color.a = 0.8;
            } else {
                marker.color.r = 0.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0;
                marker.color.a = 0.9;
            }
            
            marker.lifetime = rclcpp::Duration::from_seconds(0);
            
            marker_array.markers.push_back(marker);
        }
        
        landmark_markers_pub_->publish(marker_array);
        RCLCPP_DEBUG(this->get_logger(), "Published %zu persistent landmark markers", all_landmarks_.size());
    }
    
    void logOptimizedPose() {
        auto [R_opt, t_opt] = bundle_adjuster_->getLatestPose();
        
        RCLCPP_DEBUG(this->get_logger(), "Latest optimized pose - Translation: [%.3f, %.3f, %.3f] | Map size: %zu landmarks", 
                    t_opt.at<double>(0), t_opt.at<double>(1), t_opt.at<double>(2), all_landmarks_.size());
                    
        cv::Vec3f euler_angles;
        cv::Mat rvec;
        cv::Rodrigues(R_opt, rvec);
        double angle = cv::norm(rvec);
        
        RCLCPP_DEBUG(this->get_logger(), "Latest optimized pose - Rotation angle: %.3f rad", angle);
    }
    
    void broadcastOptimizedTransform() {
        auto [R_opt, t_opt] = bundle_adjuster_->getLatestPose();
        
        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R_opt.at<double>(i, j);
            }
        }
        
        Eigen::Quaterniond q(R_eigen);
        q.normalize();
        
        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = latest_keyframe_timestamp_;
        transform_stamped.header.frame_id = "odom";
        transform_stamped.child_frame_id = "camera_link";
        
        transform_stamped.transform.translation.x = t_opt.at<double>(0);
        transform_stamped.transform.translation.y = t_opt.at<double>(1);
        transform_stamped.transform.translation.z = t_opt.at<double>(2);
        
        transform_stamped.transform.rotation.x = q.x();
        transform_stamped.transform.rotation.y = q.y();
        transform_stamped.transform.rotation.z = q.z();
        transform_stamped.transform.rotation.w = q.w();
        
        tf_broadcaster_->sendTransform(transform_stamped);
        
        RCLCPP_DEBUG(this->get_logger(), "Broadcasting optimized camera_link transform");
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Backend>());
    rclcpp::shutdown();
    return 0;
}