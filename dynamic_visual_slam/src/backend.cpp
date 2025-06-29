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
#include <Eigen/Geometry>

class Backend : public rclcpp::Node 
{
public:
    Backend() : Node("backend") {
        // subscribe to keyframe topic
        rclcpp::QoS qos = rclcpp::QoS(30);

        // Initialize bundle adjuster with default camera parameters
        // These will be updated when we receive camera info
        bundle_adjuster_ = std::make_unique<SlidingWindowBA>(10, 640.0, 480.0, 320.0, 240.0);
        
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

        next_global_landmark_id_ = 1;
        next_observation_id_ = 1;
        descriptor_matcher_ = cv::BFMatcher(cv::NORM_HAMMING, false);

        // Association parameters
        max_descriptor_distance_ = 50.0;  // Hamming distance for ORB
        max_reprojection_distance_ = 3.0;  // pixels
        min_parallax_angle_ = 5.0;  // degrees
        
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
    
    // Persistent landmark storage for mapping
    std::unordered_map<uint64_t, geometry_msgs::msg::Point> all_landmarks_;
    std::unordered_map<uint64_t, int> landmark_observation_count_;
    std::unordered_map<uint64_t, rclcpp::Time> landmark_first_seen_;
    std::vector<geometry_msgs::msg::Pose> keyframe_poses_;
    std::vector<cv::Mat> keyframe_descriptors_;

    
    // Map management
    bool map_cleared_;
    
    // Parameters
    int min_observations_for_landmark_;
    double max_reprojection_error_;
    int bundle_adjustment_frequency_;
    int keyframe_count_;

    struct LandmarkInfo {
        uint64_t global_id;
        geometry_msgs::msg::Point position;
        cv::Mat descriptor;
        std::vector<uint64_t> observation_ids;
        int observation_count;
        rclcpp::Time last_seen;

        LandmarkInfo(uint64_t id, const geometry_msgs::msg::Point& pos, const cv::Mat& desc, const rclcpp::Time& timestamp)
            : global_id(id), position(pos), descriptor(desc.clone()), observation_count(1), last_seen(timestamp) {}
    };

    struct ObservationInfo {
        uint64_t observation_id;
        uint64_t landmark_id;
        uint64_t frame_id;
        cv::Point2f pixel;
        cv::Mat descriptor;

        ObservationInfo(uint64_t obs_id, uint64_t frame, const cv::Point2f& pix, const cv::Mat& desc)
            : observation_id(obs_id), landmark_id(0), frame_id(frame), pixel(pix), descriptor(desc.clone()) {}
    };

    std::unordered_map<uint64_t, LandmarkInfo> landmark_database_;
    std::vector<ObservationInfo> all_observations_;
    uint64_t next_global_landmark_id_;
    uint64_t next_observation_id_;
    cv::BFMatcher descriptor_matcher_;

    // Parameters for association
    double max_descriptor_distance_;
    double max_reprojection_distance_;
    double min_parallax_angle_;

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
        
        RCLCPP_INFO(this->get_logger(), "Processing keyframe %lu with %zu landmarks (Total map size: %zu)", 
                    msg->frame_id, msg->landmarks.size(), all_landmarks_.size());

        latest_keyframe_timestamp_ = msg->header.stamp;

        cv::Mat R, t;
        extractPoseFromTransform(msg->pose, R, t);

        geometry_msgs::msg::Pose keyframe_pose;
        keyframe_pose.position.x = msg->pose.translation.x;
        keyframe_pose.position.y = msg->pose.translation.y;
        keyframe_pose.position.z = msg->pose.translation.z;
        keyframe_pose.orientation = msg->pose.rotation;
        keyframe_poses_.push_back(keyframe_pose);

        int frame_id = msg->frame_id;

        std::vector<ObservationInfo> new_observations;

        // Need to organize observations and landmarks
        for (size_t i = 0; i < msg->observations.size(); i++) {
            const auto& obs = msg->observations[i];
            const auto& landmark = msg->landmarks[i];

            cv::Mat descriptor(1, obs.descriptor.size(), CV_8U);
            std::memcpy(descriptor.data, obs.descriptor.data(), obs.descriptor.size());

            ObservationInfo new_obs(next_observation_id_++, frame_id, cv::Point2f(obs.pixel_x, obs.pixel_y), descriptor);

            // Associate observation with existing landmarks
            int associated_landmark_id = associateObservation(new_obs, R, t);

            if (associated_landmark_id != -1) {
                new_obs.landmark_id = associated_landmark_id;
                auto& landmark_info = landmark_database_.at(associated_landmark_id);  // Use at() instead of []
                landmark_info.observation_count++;
                landmark_info.last_seen = msg->header.stamp;
                landmark_info.observation_ids.push_back(new_obs.observation_id);
            }
            else {
                // Create new landmark
                geometry_msgs::msg::Point landmark_pos;
                landmark_pos.x = landmark.position.x;
                landmark_pos.y = landmark.position.y;
                landmark_pos.z = landmark.position.z;
                
                uint64_t new_landmark_id = next_global_landmark_id_++;
                LandmarkInfo new_landmark(new_landmark_id, landmark_pos, descriptor, msg->header.stamp);
                new_landmark.observation_ids.push_back(new_obs.observation_id);
                
                landmark_database_.emplace(new_landmark_id, new_landmark);
                new_obs.landmark_id = new_landmark_id;
                
                RCLCPP_DEBUG(this->get_logger(), "Created new landmark %lu for observation %lu", 
                            new_landmark_id, new_obs.observation_id);
            }
            
            new_observations.push_back(new_obs);
        }
        
        // Add all new observations to database
        all_observations_.insert(all_observations_.end(), new_observations.begin(), new_observations.end());
        
        RCLCPP_INFO(this->get_logger(), "Keyframe processed. Total landmarks: %zu, Total observations: %zu", 
                    landmark_database_.size(), all_observations_.size());
    }

    int associateObservation(const ObservationInfo& obs, const cv::Mat& R, const cv::Mat& t) {
        std::vector<std::pair<int, double>> candidates;

        // Find all candidates based on descriptor
        for (const auto& [landmark_id, landmark_info] : landmark_database_) {
            std::vector<cv::DMatch> matches;

            descriptor_matcher_.match(obs.descriptor, landmark_info.descriptor, matches);

            if (!matches.empty() && matches[0].distance < max_descriptor_distance_) {
                candidates.emplace_back(landmark_id, matches[0].distance);
            }
        }

        if (candidates.empty()) {
            RCLCPP_DEBUG(this->get_logger(), "No matches, new landmark!");
            return -1;
        }

        int best_landmark_id = -1;
        double best_reprojection_error = std::numeric_limits<double>::max();

        for (const auto& [candidate_id, descriptor_distance] : candidates) {
            const auto& candidate_landmark = landmark_database_.at(candidate_id);

            cv::Point3f landmark_3d_cv(candidate_landmark.position.x, candidate_landmark.position.y, candidate_landmark.position.z);

            cv::Point2f reprojection_pixel = reprojectPoint(landmark_3d_cv, R, t);

            double reprojection_error = cv::norm(obs.pixel - reprojection_pixel);

            if (reprojection_error < max_reprojection_distance_ && reprojection_error < best_reprojection_error) {
                best_landmark_id = candidate_id;
                best_reprojection_error = reprojection_error;
            }
        }

        if (best_landmark_id != -1) {
            RCLCPP_DEBUG(this->get_logger(), "Found association with reprojection error: %.2f pixels", best_reprojection_error);
        }

        return best_landmark_id;
    }

    cv::Point2f reprojectPoint(const cv::Point3f& point_3d_ros, const cv::Mat& R, const cv::Mat& t) {
        // Transform point from world to camera ROS frame
        cv::Mat point_cam_ros = R.t() * (cv::Mat_<double>(3,1) << point_3d_ros.x, point_3d_ros.y, point_3d_ros.z) - R.t() * t;
        
        double x = point_cam_ros.at<double>(0);
        double y = point_cam_ros.at<double>(1); 
        double z = point_cam_ros.at<double>(2);
        
        if (x <= 0) {
            return cv::Point2f(-1, -1);  // Invalid projection
        }
        
        float u = fx_ * -y / x + cx_;
        float v = fy_ * -z / x + cy_;
        
        return cv::Point2f(u, v);
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
        
        for (const auto& [landmark_id, landmark_info] : landmark_database_) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = latest_keyframe_timestamp_;
            marker.ns = "landmarks";
            marker.id = static_cast<int>(landmark_id);
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            marker.pose.position = landmark_info.position;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            
            marker.scale.x = 0.005;
            marker.scale.y = 0.005;
            marker.scale.z = 0.005;
            
            if (landmark_database_.at(landmark_id).observation_count > 1) {
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 1.0;
                marker.color.a = 0.8;
            }
            else {
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                marker.color.a = 0.8;
            }
            
            marker.lifetime = rclcpp::Duration::from_seconds(0);
            
            marker_array.markers.push_back(marker);
        }
        
        landmark_markers_pub_->publish(marker_array);
        RCLCPP_DEBUG(this->get_logger(), "Published %zu persistent landmark markers", all_landmarks_.size());
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Backend>());
    rclcpp::shutdown();
    return 0;
}