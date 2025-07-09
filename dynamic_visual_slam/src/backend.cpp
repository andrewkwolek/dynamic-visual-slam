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

        // BA callback
        ba_timer_ = this->create_wall_timer(std::chrono::seconds(2), std::bind(&Backend::bundleAdjustmentCallback, this));
            
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
        max_descriptor_distance_ = 25.0;  // Hamming distance for ORB
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

    // Bundle adjustment data structures
    std::mutex keyframes_mutex_;
    std::atomic<bool> ba_running_{false};

    // Timer for bundle adjustment
    rclcpp::TimerBase::SharedPtr ba_timer_;

    // Map management
    bool map_cleared_;
    
    // Parameters
    int min_observations_for_landmark_;
    double max_reprojection_error_;
    int bundle_adjustment_frequency_;
    int keyframe_count_;

    struct LandmarkInfo {
        uint64_t global_id;
        cv::Point3f position;
        cv::Mat descriptor;
        std::vector<uint64_t> observation_ids;
        int observation_count;
        rclcpp::Time last_seen;

        LandmarkInfo(uint64_t id, const cv::Point3f& pos, const cv::Mat& desc, const rclcpp::Time& timestamp)
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

    struct KeyframeInfo {
        uint64_t frame_id;
        cv::Mat R;
        cv::Mat t;
        rclcpp::Time timestamp;
        std::vector<uint64_t> observation_ids;

        KeyframeInfo(uint64_t id, const cv::Mat& rotation, const cv::Mat& translation, const rclcpp::Time& stamp)
            : frame_id(id), R(rotation.clone()), t(translation.clone()), timestamp(stamp) {}
    };

    std::vector<KeyframeInfo> keyframes_;
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
                    msg->frame_id, msg->landmarks.size(), landmark_database_.size());

        latest_keyframe_timestamp_ = msg->header.stamp;

        cv::Mat R, t;
        extractPoseFromTransform(msg->pose, R, t);

        int frame_id = msg->frame_id;

        std::vector<ObservationInfo> new_observations;
        std::unordered_map<uint64_t, LandmarkInfo> new_landmarks;
        KeyframeInfo new_keyframe(frame_id, R, t, latest_keyframe_timestamp_);

        // Need to organize observations and landmarks
        for (size_t i = 0; i < msg->observations.size(); i++) {
            const auto& obs = msg->observations[i];
            const auto& landmark = msg->landmarks[i];

            cv::Mat descriptor(1, obs.descriptor.size(), CV_8U);
            std::memcpy(descriptor.data, obs.descriptor.data(), obs.descriptor.size());

            ObservationInfo new_obs(next_observation_id_, frame_id, cv::Point2f(obs.pixel_x, obs.pixel_y), descriptor);
            new_keyframe.observation_ids.push_back(next_observation_id_);
            next_observation_id_++;

            // Associate observation with existing landmarks
            int associated_landmark_id = associateObservation(new_obs, R, t);

            if (associated_landmark_id != -1) {
                new_obs.landmark_id = associated_landmark_id;
                auto& landmark_info = landmark_database_.at(associated_landmark_id);
                landmark_info.observation_count++;
                landmark_info.last_seen = msg->header.stamp;
                landmark_info.observation_ids.push_back(new_obs.observation_id);
            }
            else {
                // Create new landmark
                cv::Point3f landmark_pos;
                landmark_pos.x = landmark.position.x;
                landmark_pos.y = landmark.position.y;
                landmark_pos.z = landmark.position.z;
                
                uint64_t new_landmark_id = next_global_landmark_id_++;
                LandmarkInfo new_landmark(new_landmark_id, landmark_pos, descriptor, msg->header.stamp);
                new_landmark.observation_ids.push_back(new_obs.observation_id);
                
                new_landmarks.emplace(new_landmark_id, new_landmark);
                new_obs.landmark_id = new_landmark_id;
                
                RCLCPP_DEBUG(this->get_logger(), "Created new landmark %lu for observation %lu", 
                            new_landmark_id, new_obs.observation_id);
            }

            new_observations.push_back(new_obs);
        }

        keyframes_.push_back(new_keyframe);
        
        // Add all new observations to database
        all_observations_.insert(all_observations_.end(), new_observations.begin(), new_observations.end());

        for (const auto& [landmark_id, landmark_info] : new_landmarks) {
            landmark_database_.emplace(landmark_id, landmark_info);
            RCLCPP_DEBUG(this->get_logger(), "Added landmark %lu to global map", landmark_id);
        }
        
        RCLCPP_INFO(this->get_logger(), "Keyframe processed. Total landmarks: %zu, Total observations: %zu", 
                    landmark_database_.size(), all_observations_.size());

        publishAllLandmarkMarkers();
    }

    void bundleAdjustmentCallback() {
        if (ba_running_.load()) {
            RCLCPP_INFO(this->get_logger(), "Bundle adjustment already running, skipping this cycle");
            return;
        }

        std::lock_guard<std::mutex> lock(keyframes_mutex_);
        
        if (keyframes_.size() < 2) {
            RCLCPP_INFO(this->get_logger(), "Not enough keyframes for BA: %zu", keyframes_.size());
            return;
        }

        ba_running_.store(true);
        
        // Create sliding window (last 5 keyframes)
        int window_size = std::min(5, static_cast<int>(keyframes_.size()));
        int start_idx = keyframes_.size() - window_size;
        
        std::vector<KeyframeInfo> window_keyframe_infos(
            keyframes_.begin() + start_idx, 
            keyframes_.end()
        );
        
        RCLCPP_INFO(this->get_logger(), "Starting bundle adjustment with %d keyframes (window %d-%zu)", 
                    window_size, start_idx, keyframes_.size() - 1);
        
        // Extract KeyframeData vector for BA
        std::vector<KeyframeData> window_keyframes;
        for (const auto& kf_info : window_keyframe_infos) {
            window_keyframes.emplace_back(kf_info.frame_id, kf_info.R, kf_info.t, kf_info.timestamp);
        }
        
        // Collect window observation IDs efficiently
        std::set<uint64_t> window_observation_ids;
        for (const auto& kf_info : window_keyframe_infos) {
            for (uint64_t obs_id : kf_info.observation_ids) {
                window_observation_ids.insert(obs_id);
            }
        }
        
        // Extract observations and landmarks for the window
        std::vector<Observation> window_observations;
        std::set<uint64_t> window_landmark_ids;
        
        for (const auto& obs_info : all_observations_) {
            if (window_observation_ids.count(obs_info.observation_id) > 0) {
                window_landmark_ids.insert(obs_info.landmark_id);
                window_observations.emplace_back(obs_info.pixel.x, obs_info.pixel.y, obs_info.landmark_id, obs_info.frame_id);
            }
        }
        
        // Extract landmarks for the window
        std::vector<Landmark> window_landmarks;
        for (uint64_t landmark_id : window_landmark_ids) {
            const auto& landmark_info = landmark_database_.at(landmark_id);
            window_landmarks.emplace_back(landmark_id, 
                                        landmark_info.position.x, 
                                        landmark_info.position.y, 
                                        landmark_info.position.z,
                                        false); // Don't fix landmarks
        }
        
        RCLCPP_INFO(this->get_logger(), "Window contains %zu landmarks and %zu observations", 
                    window_landmarks.size(), window_observations.size());
        
        // Run bundle adjustment directly in the timer callback
        auto start_time = std::chrono::high_resolution_clock::now();
        
        OptimizationResult result = bundle_adjuster_->optimize(
            window_keyframes, 
            window_landmarks, 
            window_observations,
            20  // max iterations
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (result.success) {
            RCLCPP_INFO(this->get_logger(), 
                        "Bundle adjustment converged in %d iterations, final cost: %.6f, time: %ld ms",
                        result.iterations_completed, result.final_cost, duration.count());
            
            // Update optimized poses and landmarks
            updateOptimizedResults(result);
        } else {
            RCLCPP_WARN(this->get_logger(), 
                        "Bundle adjustment failed: %s, time: %ld ms", 
                        result.message.c_str(), duration.count());
        }

        pruneLandmarks();

        publishAllLandmarkMarkers();
        
        ba_running_.store(false);
    }

    void updateOptimizedResults(const OptimizationResult& result) {
        // No need for mutex since we're already in the timer callback with the lock held
        
        // Update keyframe poses in the vector
        for (const auto& [frame_id, pose_pair] : result.optimized_poses) {
            const auto& [R_opt, t_opt] = pose_pair;
            
            // Find and update the corresponding keyframe
            for (auto& kf_info : keyframes_) {
                if (kf_info.frame_id == (uint64_t)frame_id) {
                    kf_info.R = R_opt.clone();
                    kf_info.t = t_opt.clone();
                    break;
                }
            }
        }
        
        // Update landmark positions in the database
        for (const auto& [landmark_id, optimized_pos] : result.optimized_landmarks) {
            auto it = landmark_database_.find(landmark_id);
            if (it != landmark_database_.end()) {
                it->second.position.x = optimized_pos.x;
                it->second.position.y = optimized_pos.y;
                it->second.position.z = optimized_pos.z;
            }
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Updated %zu poses and %zu landmarks from BA", 
                    result.optimized_poses.size(), result.optimized_landmarks.size());
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

    cv::Point2f reprojectPoint(const cv::Point3f& point_3d_world, const cv::Mat& R, const cv::Mat& t) {
        cv::Mat point_world = (cv::Mat_<double>(3,1) << point_3d_world.x, point_3d_world.y, point_3d_world.z);
        cv::Mat point_camera = R.t() * (point_world - t);
        
        double x = point_camera.at<double>(0);  // Right in optical frame
        double y = point_camera.at<double>(1);  // Down in optical frame
        double z = point_camera.at<double>(2);  // Forward in optical frame
        
        if (z <= 0) {
            return cv::Point2f(-1, -1);
        }
        
        float u = fx_ * x / z + cx_;
        float v = fy_ * y / z + cy_;
        
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
        
        double norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        qw /= norm; qx /= norm; qy /= norm; qz /= norm;
        
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

    void pruneLandmarks() {
        auto current_time = this->now();
        const int min_observation_threshold = 2;
        const double max_time_since_seen = 20.0;
        
        std::vector<uint64_t> landmarks_to_remove;
        
        for (const auto& [landmark_id, landmark_info] : landmark_database_) {
            double time_since_seen = (current_time - landmark_info.last_seen).seconds();
            if (landmark_info.observation_count < min_observation_threshold && time_since_seen > max_time_since_seen) {
                landmarks_to_remove.push_back(landmark_id);
                RCLCPP_DEBUG(this->get_logger(), "Marking landmark %lu for removal: insufficient observations (%d < %d)", 
                            landmark_id, landmark_info.observation_count, min_observation_threshold);
            }
        }
        
        int removed_landmarks = 0;
        int removed_observations = 0;
        
        for (uint64_t landmark_id : landmarks_to_remove) {
            const auto& landmark_info = landmark_database_.at(landmark_id);
            std::set<uint64_t> obs_ids_to_remove(landmark_info.observation_ids.begin(), 
                                                 landmark_info.observation_ids.end());
            
            landmark_database_.erase(landmark_id);
            removed_landmarks++;
            
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
            
            for (auto& keyframe : keyframes_) {
                auto& obs_ids = keyframe.observation_ids;
                obs_ids.erase(
                    std::remove_if(obs_ids.begin(), obs_ids.end(), [&obs_ids_to_remove](uint64_t obs_id) { return obs_ids_to_remove.count(obs_id) > 0; }),
                    obs_ids.end()
                );
            }
        }
        
        if (removed_landmarks > 0) {
            RCLCPP_INFO(this->get_logger(), 
                        "Landmark pruning completed: removed %d landmarks and %d observations. "
                        "Remaining landmarks: %zu, observations: %zu", 
                        removed_landmarks, removed_observations,
                        landmark_database_.size(), all_observations_.size());
        } else {
            RCLCPP_INFO(this->get_logger(), "Landmark pruning: no landmarks removed");
        }
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
        
        // Transformation matrix from optical to ROS for visualization
        cv::Mat T_opt_to_ros = (cv::Mat_<double>(3,3) << 
            0,  0,  1,    // Optical Z → ROS X (forward)
            -1, 0,  0,    // Optical -X → ROS Y (left)
            0, -1,  0     // Optical -Y → ROS Z (up)
        );
        
        for (const auto& [landmark_id, landmark_info] : landmark_database_) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = latest_keyframe_timestamp_;
            marker.ns = "landmarks";
            marker.id = static_cast<int>(landmark_id);
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            // Convert landmark position from optical to ROS for visualization
            cv::Mat pos_optical = (cv::Mat_<double>(3,1) << 
                landmark_info.position.x, 
                landmark_info.position.y, 
                landmark_info.position.z);
            cv::Mat pos_ros = T_opt_to_ros * pos_optical;
            
            marker.pose.position.x = pos_ros.at<double>(0);
            marker.pose.position.y = pos_ros.at<double>(1);
            marker.pose.position.z = pos_ros.at<double>(2);
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
        RCLCPP_DEBUG(this->get_logger(), "Published %zu landmark markers (converted from optical to ROS)", 
                    landmark_database_.size());
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Backend>());
    rclcpp::shutdown();
    return 0;
}