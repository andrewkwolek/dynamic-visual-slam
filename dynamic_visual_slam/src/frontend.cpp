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

class Frontend : public rclcpp::Node
{
public:
    Frontend() : Node("frontend")
    {
        rclcpp::QoS qos = rclcpp::QoS(30);

        // Create message filter subscribers
        rgb_sub_.subscribe(this, "/camera/camera/color/image_raw", qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, "/camera/camera/aligned_depth_to_color/image_raw", qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>(10), rgb_sub_, depth_sub_);
        sync_->registerCallback(std::bind(&Frontend::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        // Create subscriptions
        rgb_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera/camera/color/camera_info", qos, std::bind(&Frontend::rgbInfoCallback, this, std::placeholders::_1));
        depth_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera/camera/aligned_depth_to_color/camera_info", qos, std::bind(&Frontend::depthInfoCallback, this, std::placeholders::_1));

        // Create publishers
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/feature_detector/features_image", qos);
        camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/feature_detector/camera_info", qos);
        keyframe_pub_ = this->create_publisher<dynamic_visual_slam_interfaces::msg::Keyframe>("/frontend/keyframe", qos);
        dgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/frontend/dgb_image", qos);
        
        // Initialize ORB feature detector
        orb_extractor_ = std::make_unique<ORB_SLAM3::ORBextractor>(1000, 1.2f, 8, 20, 7);

        prev_frame_valid_ = false;

        // Create tf broadcasters
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        matcher_ = cv::BFMatcher(cv::NORM_HAMMING);

        // Create odom frame
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

        // Create camera_link frame
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

        R_ = cv::Mat::eye(3, 3, CV_64F);
        t_ = cv::Mat::zeros(3, 1, CV_64F);

        rgb_camera_matrix_ = cv::Mat();
        rgb_dist_coeffs_ = cv::Mat();
        depth_camera_matrix_ = cv::Mat();
        depth_dist_coeffs_ = cv::Mat();

        keyframe_id_ = 0;
        frames_since_last_keyframe_ = 0;
        has_last_keyframe_ = false;

        MAX_DEPTH = 3.0f;
        MIN_DEPTH = 0.3f;
            
        RCLCPP_DEBUG(this->get_logger(), "Image processor node initialized");
    }

private:
    // Message filter subscriptions
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr rgb_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;

    //Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
    rclcpp::Publisher<dynamic_visual_slam_interfaces::msg::Keyframe>::SharedPtr keyframe_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr dgb_pub_;

    // Camera info
    sensor_msgs::msg::CameraInfo::SharedPtr latest_rgb_camera_info_;
    sensor_msgs::msg::CameraInfo::SharedPtr latest_depth_camera_info_;
    cv::Mat rgb_camera_matrix_;
    cv::Mat rgb_dist_coeffs_;
    cv::Mat depth_camera_matrix_;
    cv::Mat depth_dist_coeffs_;
    float rgb_fx_;
    float rgb_fy_;
    float rgb_cx_;
    float rgb_cy_;
    float depth_fx_;
    float depth_fy_;
    float depth_cx_;
    float depth_cy_;
    sensor_msgs::msg::Image dgb_image_;

    // Depth filter
    float MAX_DEPTH;
    float MIN_DEPTH;

    // ORB Feature detector
    std::vector<int> vLappingArea = {0, 0};
    std::unique_ptr<ORB_SLAM3::ORBextractor> orb_extractor_;

    // BF feature matcher
    cv::BFMatcher matcher_;

    // Message filter synchronizer
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,sensor_msgs::msg::Image>>> sync_;

    // Previous frame info
    cv::Mat prev_frame_gray_;
    cv::Mat prev_frame_depth_;
    std::vector<cv::KeyPoint> prev_kps_;
    std::vector<cv::Point2f> prev_points_;
    cv::Mat prev_descriptors_;
    bool prev_frame_valid_;

    // TF broadcasters
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;

    // Camera transform
    cv::Mat R_;
    cv::Mat t_;

    // Keyframe detection
    long long int keyframe_id_;
    int frames_since_last_keyframe_;
    cv::Mat last_keyframe_descriptors_;
    std::vector<cv::KeyPoint> last_keyframe_keypoints_;
    cv::Mat last_keyframe_depth_;
    bool has_last_keyframe_;

    void broadcastTransformROS(const rclcpp::Time& stamp) {
        // Transformation matrix from optical frame to ROS frame
        // Optical: X=right, Y=down, Z=forward
        // ROS:     X=forward, Y=left, Z=up
        cv::Mat T_opt_to_ros = (cv::Mat_<double>(3,3) << 
            0,  0,  1,    // Optical Z → ROS X (forward)
            -1, 0,  0,    // Optical -X → ROS Y (left)
            0, -1,  0     // Optical -Y → ROS Z (up)
        );
        
        cv::Mat R_ros = T_opt_to_ros * R_ * T_opt_to_ros.t();
        cv::Mat t_ros = T_opt_to_ros * t_;
        
        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R_ros.at<double>(i, j);
            }
        }
        
        Eigen::Quaterniond q(R_eigen);
        
        q.normalize();
        
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
                    "Broadcasted TF - Position: [%.3f, %.3f, %.3f], Quat: [%.3f, %.3f, %.3f, %.3f]",
                    t_ros.at<double>(0), t_ros.at<double>(1), t_ros.at<double>(2),
                    q.x(), q.y(), q.z(), q.w());
    }

    bool isValidDepth(const cv::Mat& depth_image, int x, int y) {
        if (x < 0 || y < 0 || x >= depth_image.cols || y >= depth_image.rows) {
            return false;
        }

        float depth = depth_image.at<uint16_t>(y, x) * 0.001f;

        if (depth < MIN_DEPTH || depth > MAX_DEPTH || std::isnan(depth) || std::isinf(depth) || depth < 0.0f) {
            return false;
        }

        return true;
    }

    void filterDepth(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, const cv::Mat& depth_image, std::vector<cv::KeyPoint>& filtered_keypoints, cv::Mat& filtered_descriptors) {
        std::vector<int> original_indices;

        for (size_t i = 0; i < keypoints.size(); i++) {
            cv::Point2f pt = keypoints[i].pt;
            int x = static_cast<int>(std::round(pt.x));
            int y = static_cast<int>(std::round(pt.y));

            if (isValidDepth(depth_image, x, y)) {
                filtered_keypoints.push_back(keypoints[i]);
                original_indices.push_back(i);
            }
        }

        if (!filtered_keypoints.empty() && !descriptors.empty()) {
            filtered_descriptors = cv::Mat(filtered_keypoints.size(), descriptors.cols, descriptors.type());
            for (size_t i = 0; i < original_indices.size(); i++) {
                descriptors.row(original_indices[i]).copyTo(filtered_descriptors.row(i));
            }
        }
    }

    bool isMotionOutlier(const cv::Mat& R_new, const cv::Mat& t_new) {
        const double MAX_TRANSLATION = 0.5;
        const double MAX_ROTATION = 0.2;
        
        double translation_norm = cv::norm(t_new);
        if (translation_norm > MAX_TRANSLATION) {
            // RCLCPP_WARN(this->get_logger(), "Translation outlier detected: %f m", translation_norm);
            

            cv::Mat rvec;
            cv::Rodrigues(R_new, rvec);
            double rotation_angle = cv::norm(rvec);
            if (rotation_angle > MAX_ROTATION) {
                // RCLCPP_WARN(this->get_logger(), "Rotation outlier detected: %f rad", rotation_angle);
            }

            return true;
        }
        
        return false;
    }

    bool isKeyframe(const cv::Mat& current_descriptors, const std::vector<cv::KeyPoint>& current_keypoints) {
        if (!has_last_keyframe_) {
            has_last_keyframe_ = true;
            return true;
        }

        bool tracking_criterion = false;
        if (!last_keyframe_descriptors_.empty() && !current_descriptors.empty()) {
            std::vector<cv::DMatch> all_keyframe_matches;
            matcher_.match(current_descriptors, last_keyframe_descriptors_, all_keyframe_matches);
            
            std::vector<cv::DMatch> distance_filtered_keyframe_matches;
            float max_distance = 50.0f;
            for (const auto& match : all_keyframe_matches) {
                if (match.distance < max_distance) {
                    distance_filtered_keyframe_matches.push_back(match);
                }
            }
            
            std::vector<cv::DMatch> geometrically_consistent_keyframe_matches;
            if (distance_filtered_keyframe_matches.size() >= 8) {
                std::vector<cv::Point2f> last_kf_pts, current_kf_pts;
                for (const auto& match : distance_filtered_keyframe_matches) {
                    last_kf_pts.push_back(last_keyframe_keypoints_[match.trainIdx].pt);
                    current_kf_pts.push_back(current_keypoints[match.queryIdx].pt);
                }
                
                std::vector<uchar> kf_inliers_mask;
                cv::findFundamentalMat(last_kf_pts, current_kf_pts, kf_inliers_mask, cv::FM_RANSAC, 2.0, 0.99);
                
                for (size_t i = 0; i < kf_inliers_mask.size(); i++) {
                    if (kf_inliers_mask[i]) {
                        geometrically_consistent_keyframe_matches.push_back(distance_filtered_keyframe_matches[i]);
                    }
                }
            } else {
                geometrically_consistent_keyframe_matches = distance_filtered_keyframe_matches;
            }
            
            RCLCPP_DEBUG(this->get_logger(), "Valid matches with last keyframe: %zu", geometrically_consistent_keyframe_matches.size());
            
            tracking_criterion = (geometrically_consistent_keyframe_matches.size() < 150);
        }

        if (tracking_criterion || frames_since_last_keyframe_ > 30) {
            frames_since_last_keyframe_ = 0;
            return true;
        }

        frames_since_last_keyframe_++;
        return false;
    }

    void publishKeyframe(const std::vector<cv::KeyPoint>& current_keypoints, const cv::Mat& current_descriptors, const cv::Mat& current_depth_frame, const rclcpp::Time& stamp) {
        dynamic_visual_slam_interfaces::msg::Keyframe kf;

        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R_.at<double>(i, j);  // R_ is in optical
            }
        }
        Eigen::Quaterniond q(R_eigen);
        q.normalize();
        
        geometry_msgs::msg::Transform transform;
        transform.translation.x = t_.at<double>(0);  // t_ is in optical
        transform.translation.y = t_.at<double>(1);
        transform.translation.z = t_.at<double>(2);
        transform.rotation.x = q.x();
        transform.rotation.y = q.y();
        transform.rotation.z = q.z();
        transform.rotation.w = q.w();

        kf.pose = transform;
        kf.header.frame_id = "camera_link";
        kf.header.stamp = stamp;
        kf.frame_id = keyframe_id_++;

        for (size_t i = 0; i < current_keypoints.size(); i++) {
            cv::Point2f pt = current_keypoints[i].pt;
            int x = static_cast<int>(std::round(pt.x));
            int y = static_cast<int>(std::round(pt.y));
            float pt_depth = current_depth_frame.at<uint16_t>(y, x) * 0.001f;
            
            cv::Point3f pt_3d_optical((pt.x - rgb_cx_) * pt_depth / rgb_fx_, 
                                    (pt.y - rgb_cy_) * pt_depth / rgb_fy_, 
                                    pt_depth);
            
            if (pt_3d_optical.z > 0.3 && pt_3d_optical.z < 3.0) {
                cv::Mat landmark_camera = (cv::Mat_<double>(3,1) << 
                    pt_3d_optical.x, pt_3d_optical.y, pt_3d_optical.z);
                cv::Mat landmark_world = R_ * landmark_camera + t_;

                dynamic_visual_slam_interfaces::msg::Landmark landmark;
                landmark.landmark_id = static_cast<uint64_t>(i);
                landmark.position.x = landmark_world.at<double>(0);
                landmark.position.y = landmark_world.at<double>(1);
                landmark.position.z = landmark_world.at<double>(2);
                
                dynamic_visual_slam_interfaces::msg::Observation obs;
                obs.landmark_id = static_cast<uint64_t>(i);
                obs.pixel_x = current_keypoints[i].pt.x;
                obs.pixel_y = current_keypoints[i].pt.y;
                cv::Mat descriptor_row = current_descriptors.row(i);
                obs.descriptor.assign(descriptor_row.data, descriptor_row.data + descriptor_row.total());
                
                kf.landmarks.push_back(landmark);
                kf.observations.push_back(obs);
            }
        }

        last_keyframe_keypoints_ = current_keypoints;
        last_keyframe_descriptors_ = current_descriptors.clone();

        keyframe_pub_->publish(kf);
        dgb_pub_->publish(dgb_image_);
    }

    void estimateCameraPose(const std::vector<cv::KeyPoint>& prev_kps, const std::vector<cv::KeyPoint>& curr_kps, const std::vector<cv::DMatch>& good_matches, const cv::Mat& prev_depth, const rclcpp::Time& stamp) {
        std::vector<cv::Point3f> points3d;
        std::vector<cv::Point2f> points2d;

        if (rgb_camera_matrix_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "RGB camera matrix is empty!");
            return;
        }

        try {
            for (const auto& match : good_matches) {
                cv::Point2f prev_pt = prev_kps[match.trainIdx].pt;
                cv::Point2f curr_pt = curr_kps[match.queryIdx].pt;
                
                int x_prev = static_cast<int>(std::round(prev_pt.x));
                int y_prev = static_cast<int>(std::round(prev_pt.y));
                
                if (x_prev < 0 || y_prev < 0 || 
                    x_prev >= prev_depth.cols || y_prev >= prev_depth.rows) {
                    continue;
                }
                
                float d_prev = prev_depth.at<uint16_t>(y_prev, x_prev) * 0.001f;
                
                if (d_prev <= 0.3f || d_prev > 3.0f) {
                    continue;
                }
                
                // Keep in optical coordinates (this is what solvePnP expects)
                cv::Point3f pt3d_prev((prev_pt.x - rgb_cx_) * d_prev / rgb_fx_, 
                                    (prev_pt.y - rgb_cy_) * d_prev / rgb_fy_, 
                                    d_prev);
                
                points3d.push_back(pt3d_prev);
                points2d.push_back(curr_pt);
            }
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during point processing: %s", e.what());
            return;
        }

        if (points3d.size() < 6) {
            RCLCPP_WARN(this->get_logger(), "Not enough matching points for pose estimation: %zu", points3d.size());
            return;
        }

        try {
            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            
            bool success = cv::solvePnPRansac(
                points3d,           // 3D points from previous frame (optical)
                points2d,           // 2D points in current frame
                rgb_camera_matrix_, // Camera matrix
                rgb_dist_coeffs_,   // Distortion coefficients
                rvec,               // Output rotation vector
                tvec,               // Output translation vector
                false               // Don't use extrinsic guess
            );

            if (!success) {
                RCLCPP_WARN(this->get_logger(), "PnP RANSAC failed");
                return;
            }

            cv::Mat R_relative;
            cv::Rodrigues(rvec, R_relative);
            cv::Mat t_relative = tvec.clone();
            
            cv::Mat R_curr_inv = R_relative.t();
            cv::Mat t_curr_inv = -R_curr_inv * t_relative;

            if (isMotionOutlier(R_curr_inv, t_curr_inv)) {
                return;
            }

            t_ = t_ + R_ * t_curr_inv;
            R_ = R_ * R_curr_inv;

            broadcastTransformROS(stamp);

            RCLCPP_DEBUG(this->get_logger(), "Camera position (optical): [%f, %f, %f]", 
                        t_.at<double>(0), t_.at<double>(1), t_.at<double>(2));
                            
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during PnP estimation: %s", e.what());
            return;
        }
    }

    void rgbInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_rgb_camera_info_ = msg;

        rgb_camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        rgb_camera_matrix_.at<double>(0, 0)= msg->k[0];
        rgb_fx_ = msg->k[0];
        rgb_camera_matrix_.at<double>(0, 2) = msg->k[2];
        rgb_cx_ = msg->k[2];
        rgb_camera_matrix_.at<double>(1, 1) = msg->k[4];
        rgb_fy_ = msg->k[4];
        rgb_camera_matrix_.at<double>(1, 2) = msg->k[5];
        rgb_cy_ = msg->k[5];
        rgb_camera_matrix_.at<double>(2, 2) = 1.0;

        rgb_dist_coeffs_ = cv::Mat(1, 5, CV_64F);
        for (int i = 0; i < 5; i++) {
            rgb_dist_coeffs_.at<double>(0, i) = msg->d[i];
        }
    }

    void depthInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_depth_camera_info_ = msg;

        depth_camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        depth_camera_matrix_.at<double>(0, 0) = msg->k[0];
        depth_fx_ = msg->k[0];
        depth_camera_matrix_.at<double>(0, 2) = msg->k[2];
        depth_cx_ = msg->k[2];
        depth_camera_matrix_.at<double>(1, 1) = msg->k[4];
        depth_fy_ = msg->k[4];
        depth_camera_matrix_.at<double>(1, 2) = msg->k[5];
        depth_cy_ = msg->k[5];
        depth_camera_matrix_.at<double>(2, 2) = 1.0;

        depth_dist_coeffs_ = cv::Mat(1, 5, CV_64F);
        for (int i = 0; i < 5; i++) {
            depth_dist_coeffs_.at<double>(0, i) = msg->d[i];
        }
    }

    void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
    {
        try {
            dgb_image_ = *rgb_msg;
            cv_bridge::CvImagePtr rgb_cv_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
            cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv::Mat current_frame_rgb = rgb_cv_ptr->image;
            cv::Mat current_frame_depth = depth_cv_ptr->image;
            cv::Mat current_frame_gray;
            int num_features;

            cv::cvtColor(current_frame_rgb, current_frame_gray, cv::COLOR_BGR2GRAY);

            if (prev_frame_valid_) {
                cv::Mat vis_image = current_frame_rgb.clone();
                
                // Extract all features
                std::vector<cv::KeyPoint> current_keypoints;
                cv::Mat current_descriptors;
                num_features = (*orb_extractor_)(current_frame_gray, cv::noArray(), current_keypoints, current_descriptors, vLappingArea);

                // Apply depth filtering to all features
                std::vector<cv::KeyPoint> filtered_keypoints;
                cv::Mat filtered_descriptors;
                filterDepth(current_keypoints, current_descriptors, current_frame_depth, filtered_keypoints, filtered_descriptors);

                RCLCPP_DEBUG(this->get_logger(), "Features: %d total, %zu depth-filtered", 
                            num_features, filtered_keypoints.size());

                if (filtered_keypoints.empty() || prev_kps_.empty() || filtered_descriptors.empty() || prev_descriptors_.empty()) {
                    RCLCPP_WARN(this->get_logger(), "No features detected in one of the frames.");
                    current_frame_gray.copyTo(prev_frame_gray_);
                    current_frame_depth.copyTo(prev_frame_depth_);
                    prev_kps_ = filtered_keypoints;
                    prev_descriptors_ = filtered_descriptors.clone();
                    return;
                }

                // Match against previous frame
                std::vector<cv::DMatch> all_matches;
                matcher_.match(filtered_descriptors, prev_descriptors_, all_matches);

                // Distance filtering
                std::vector<cv::DMatch> distance_filtered_matches;
                float max_distance = 50.0f; 
                for (const auto& match : all_matches) {
                    if (match.distance < max_distance) {
                        distance_filtered_matches.push_back(match);
                    }
                }

                // Geometric filtering
                std::vector<cv::DMatch> geometrically_consistent_matches;
                if (distance_filtered_matches.size() >= 8) {
                    std::vector<cv::Point2f> prev_pts, curr_pts;
                    for (const auto& match : distance_filtered_matches) {
                        prev_pts.push_back(prev_kps_[match.trainIdx].pt);
                        curr_pts.push_back(filtered_keypoints[match.queryIdx].pt);
                    }
                    
                    std::vector<uchar> inliers_mask;
                    cv::Mat fundamental_matrix = cv::findFundamentalMat(prev_pts, curr_pts, inliers_mask, cv::FM_RANSAC, 2.0, 0.99);

                    for (size_t i = 0; i < inliers_mask.size(); i++) {
                        if (inliers_mask[i]) {
                            geometrically_consistent_matches.push_back(distance_filtered_matches[i]);
                        }
                    }
                    
                    RCLCPP_DEBUG(this->get_logger(), "Matches: %zu initial -> %zu distance -> %zu geometric", 
                                all_matches.size(), distance_filtered_matches.size(), geometrically_consistent_matches.size());
                } else {
                    geometrically_consistent_matches = distance_filtered_matches;
                    RCLCPP_WARN(this->get_logger(), "Insufficient matches for RANSAC: %zu", distance_filtered_matches.size());
                }
                
                // Prepare data for backend (culled features)
                std::vector<cv::KeyPoint> backend_keypoints;
                cv::Mat backend_descriptors;
                
                // Track which features are matched
                std::set<int> matched_indices;
                for (const auto& match : geometrically_consistent_matches) {
                    matched_indices.insert(match.queryIdx);
                }
                
                // Step 1: Add all matched features (highest priority)
                for (const auto& match : geometrically_consistent_matches) {
                    int curr_idx = match.queryIdx;
                    backend_keypoints.push_back(filtered_keypoints[curr_idx]);
                    
                    if (backend_descriptors.empty()) {
                        backend_descriptors = filtered_descriptors.row(curr_idx).clone();
                    } else {
                        cv::vconcat(backend_descriptors, filtered_descriptors.row(curr_idx), backend_descriptors);
                    }
                }
                
                // Step 2: Add high-quality unmatched features for new landmarks
                std::vector<std::pair<float, int>> unmatched_features;
                for (size_t i = 0; i < filtered_keypoints.size(); i++) {
                    if (matched_indices.find(i) == matched_indices.end()) {
                        unmatched_features.push_back({filtered_keypoints[i].response, i});
                    }
                }
                
                // Sort by quality and take best unmatched features
                std::sort(unmatched_features.begin(), unmatched_features.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
                
                const int MAX_NEW_FEATURES = 200;  // Tunable parameter
                const float MIN_RESPONSE = 50.0f;
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
                        "Culling: %zu total -> %zu depth-filtered -> %zu matched -> %d new -> %zu backend",
                        current_keypoints.size(), filtered_keypoints.size(), 
                        geometrically_consistent_matches.size(), added_new, backend_keypoints.size());

                // Visualization - draw matched features
                for (const auto& match : geometrically_consistent_matches) {
                    cv::Point2f curr_pt = filtered_keypoints[match.queryIdx].pt;
                    cv::circle(vis_image, curr_pt, 3, cv::Scalar(0, 255, 0), 2);
                }

                // Estimate pose using all geometric matches (for frontend tracking)
                if (geometrically_consistent_matches.size() >= 5) {
                    estimateCameraPose(prev_kps_, filtered_keypoints, geometrically_consistent_matches, prev_frame_depth_, rgb_msg->header.stamp);
                }

                // Send culled features to backend
                if (isKeyframe(backend_descriptors, backend_keypoints)) {
                    publishKeyframe(backend_keypoints, backend_descriptors, current_frame_depth, rgb_msg->header.stamp);
                }

                // Update for next frame (keep ALL depth-filtered features for matching)
                prev_kps_ = filtered_keypoints;
                prev_descriptors_ = filtered_descriptors.clone();
                
                sensor_msgs::msg::Image::SharedPtr out_msg = cv_bridge::CvImage(rgb_msg->header, "bgr8", vis_image).toImageMsg();
                image_pub_->publish(*out_msg);

                if (latest_rgb_camera_info_) {
                    auto camera_info = *latest_rgb_camera_info_;
                    camera_info.header = rgb_msg->header;
                    camera_info_pub_->publish(camera_info);
                }
                
                current_frame_gray.copyTo(prev_frame_gray_);
                current_frame_depth.copyTo(prev_frame_depth_);
            } 
            else {
                // First frame - extract and send all depth-filtered features
                std::vector<cv::KeyPoint> current_keypoints;
                cv::Mat current_descriptors;
                RCLCPP_DEBUG(this->get_logger(), "Running ORB extractor for first frame!");
                num_features = (*orb_extractor_)(current_frame_gray, cv::noArray(), current_keypoints, current_descriptors, vLappingArea);

                std::vector<cv::KeyPoint> filtered_keypoints;
                cv::Mat filtered_descriptors;
                filterDepth(current_keypoints, current_descriptors, current_frame_depth, filtered_keypoints, filtered_descriptors);

                // For first frame, send all filtered features to backend
                publishKeyframe(filtered_keypoints, filtered_descriptors, current_frame_depth, rgb_msg->header.stamp);

                prev_kps_ = filtered_keypoints;
                prev_descriptors_ = filtered_descriptors.clone();

                if (latest_rgb_camera_info_) {
                    auto camera_info = *latest_rgb_camera_info_;
                    camera_info.header = rgb_msg->header;
                    camera_info_pub_->publish(camera_info);
                }
                
                current_frame_gray.copyTo(prev_frame_gray_);
                current_frame_depth.copyTo(prev_frame_depth_);
                prev_frame_valid_ = true;
            }
            
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Frontend>());
    rclcpp::shutdown();
    return 0;
}