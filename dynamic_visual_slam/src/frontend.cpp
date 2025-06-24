#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
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
        
        // Initialize ORB feature detector
        orb_detector_ = cv::ORB::create(800);

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
            
        RCLCPP_INFO(this->get_logger(), "Image processor node initialized");
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

    // ORB Feature detector
    cv::Ptr<cv::ORB> orb_detector_;

    // FLANN feature matcher
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

    void broadcastTransform(const rclcpp::Time& stamp) {
        // First convert rotation matrix to quaternion
        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R_.at<double>(i, j);
            }
        }
        
        Eigen::Quaterniond q(R_eigen);
        
        // Create the transform message for odom -> camera
        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = stamp;
        transform_stamped.header.frame_id = "odom";
        transform_stamped.child_frame_id = "camera_link";
        
        // Set the translation
        transform_stamped.transform.translation.x = t_.at<double>(0);
        transform_stamped.transform.translation.y = t_.at<double>(1);
        transform_stamped.transform.translation.z = t_.at<double>(2);
        
        // Set the rotation
        transform_stamped.transform.rotation.x = q.x();
        transform_stamped.transform.rotation.y = q.y();
        transform_stamped.transform.rotation.z = q.z();
        transform_stamped.transform.rotation.w = q.w();
        
        // Broadcast the transform
        tf_broadcaster_->sendTransform(transform_stamped);
    }

    bool isMotionOutlier(const cv::Mat& R_new, const cv::Mat& t_new) {
        const double MAX_TRANSLATION = 0.5;
        const double MAX_ROTATION = 0.2;
        
        double translation_norm = cv::norm(t_new);
        if (translation_norm > MAX_TRANSLATION) {
            RCLCPP_WARN(this->get_logger(), "Translation outlier detected: %f m", translation_norm);
            

            cv::Mat rvec;
            cv::Rodrigues(R_new, rvec);
            double rotation_angle = cv::norm(rvec);
            if (rotation_angle > MAX_ROTATION) {
                RCLCPP_WARN(this->get_logger(), "Rotation outlier detected: %f rad", rotation_angle);
            }

            return true;
        }
        
        return false;
    }

    bool isKeyframe(const cv::Mat& current_descriptors) {
        if (!has_last_keyframe_) {
            has_last_keyframe_ = true;
            return true;
        }

        bool tracking_criterion = false;
        if (!last_keyframe_descriptors_.empty() && !current_descriptors.empty()) {
            std::vector<cv::DMatch> keyframe_matches;
            matcher_.match(current_descriptors, last_keyframe_descriptors_, keyframe_matches);
            
            std::vector<cv::DMatch> good_keyframe_matches;
            float max_distance = 50.0f;
            for (const auto& match : keyframe_matches) {
                if (match.distance < max_distance) {
                    good_keyframe_matches.push_back(match);
                }
            }
            
            tracking_criterion = (good_keyframe_matches.size() < 50);
        }

        if (frames_since_last_keyframe_ > 30 || tracking_criterion) {
            RCLCPP_INFO(this->get_logger(), "Found keyframe!");
            frames_since_last_keyframe_ = 0;
            return true;
        }

        frames_since_last_keyframe_++;
        return false;
    }

    void publishKeyframe(const std::vector<cv::KeyPoint>& current_keypoints, const cv::Mat& current_descriptors, const cv::Mat& current_depth_frame, const rclcpp::Time& stamp) {
        dynamic_visual_slam_interfaces::msg::Keyframe kf;

        geometry_msgs::msg::Transform transform;
    
        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R_.at<double>(i, j);
            }
        }
        Eigen::Quaterniond q(R_eigen);
        
        transform.translation.x = t_.at<double>(0);
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
            cv::Point3f pt_3d((pt.x - rgb_cx_) * pt_depth / rgb_fx_, (pt.y - rgb_cy_) *pt_depth / rgb_fy_, pt_depth);
            
            if (pt_3d.z > 0.3 && pt_3d.z < 3.0) {
                dynamic_visual_slam_interfaces::msg::Landmark landmark;
                landmark.landmark_id = static_cast<uint64_t>(i);
                landmark.position.x = pt_3d.x;
                landmark.position.y = pt_3d.y;
                landmark.position.z = pt_3d.z;
                landmark.is_new = true;
                
                dynamic_visual_slam_interfaces::msg::Observation obs;
                obs.landmark_id = static_cast<uint64_t>(i);
                obs.pixel_x = current_keypoints[i].pt.x;
                obs.pixel_y = current_keypoints[i].pt.y;
                
                kf.landmarks.push_back(landmark);
                kf.observations.push_back(obs);
            }
        }

        last_keyframe_keypoints_ = current_keypoints;
        last_keyframe_descriptors_ = current_descriptors.clone();

        keyframe_pub_->publish(kf);

        // RCLCPP_INFO(this->get_logger(), "Published KeyFrame!");
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
                points3d,           // 3D points from previous frame
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

            cv::Mat T_optical_to_ros = cv::Mat::eye(4, 4, CV_64F);
            
            // For RealSense cameras and standard ROS conventions:
            // X_ros = Z_optical
            // Y_ros = -X_optical
            // Z_ros = -Y_optical
            T_optical_to_ros.at<double>(0, 0) = 0.0;  // Optical Z → ROS X
            T_optical_to_ros.at<double>(0, 1) = 0.0;
            T_optical_to_ros.at<double>(0, 2) = 1.0;
            
            T_optical_to_ros.at<double>(1, 0) = -1.0; // Optical -X → ROS Y
            T_optical_to_ros.at<double>(1, 1) = 0.0;
            T_optical_to_ros.at<double>(1, 2) = 0.0;
            
            T_optical_to_ros.at<double>(2, 0) = 0.0;  // Optical -Y → ROS Z
            T_optical_to_ros.at<double>(2, 1) = -1.0;
            T_optical_to_ros.at<double>(2, 2) = 0.0;

            cv::Mat T_optical = cv::Mat::eye(4, 4, CV_64F);
            R_curr_inv.copyTo(T_optical(cv::Rect(0, 0, 3, 3)));
            t_curr_inv.copyTo(T_optical(cv::Rect(3, 0, 1, 3)));
            
            // Transform to ROS frame
            cv::Mat T_ros = T_optical_to_ros * T_optical * T_optical_to_ros.inv();

            cv::Mat R_ros = T_ros(cv::Rect(0, 0, 3, 3));
            cv::Mat t_ros = T_ros(cv::Rect(3, 0, 1, 3));

            if (isMotionOutlier(R_ros, t_ros)) {
                RCLCPP_WARN(this->get_logger(), "Motion outlier rejected - skipping frame");
                return;
            }

            t_ = t_ + R_ * t_ros;
            R_ = R_ * R_ros;

            // broadcastTransform(stamp);

            RCLCPP_DEBUG(this->get_logger(), "Camera position: [%f, %f, %f]", t_.at<double>(0), t_.at<double>(1), t_.at<double>(2));
                        
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
            cv_bridge::CvImagePtr rgb_cv_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
            cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv::Mat current_frame_rgb = rgb_cv_ptr->image;
            cv::Mat current_frame_depth = depth_cv_ptr->image;
            cv::Mat current_frame_gray;

            cv::cvtColor(current_frame_rgb, current_frame_gray, cv::COLOR_BGR2GRAY);

            cv::Mat depth_mask;
            depth_mask = (current_frame_depth > 300) & (current_frame_depth < 3000);

            if (prev_frame_valid_) {
                cv::Mat vis_image = current_frame_rgb.clone();
                
                std::vector<cv::KeyPoint> current_keypoints;
                cv::Mat current_descriptors;
                orb_detector_->detectAndCompute(current_frame_gray, depth_mask, current_keypoints, current_descriptors);

                RCLCPP_DEBUG(this->get_logger(), "Current keypoints: %zu, Prev keypoints: %zu", 
                        current_keypoints.size(), prev_kps_.size());
                RCLCPP_DEBUG(this->get_logger(), "Current desc rows: %d, Prev desc rows: %d", 
                        current_descriptors.rows, prev_descriptors_.rows);

                if (current_keypoints.empty() || prev_kps_.empty() || current_descriptors.empty() || prev_descriptors_.empty()) {
                    RCLCPP_WARN(this->get_logger(), "No featured detected in one of the frames.");
                    current_frame_gray.copyTo(prev_frame_gray_);
                    current_frame_depth.copyTo(prev_frame_depth_);
                    prev_kps_ = current_keypoints;
                    prev_descriptors_ = current_descriptors;
                    return;
                }

                std::vector<cv::DMatch> all_matches;
                matcher_.match(current_descriptors, prev_descriptors_, all_matches);

                RCLCPP_DEBUG(this->get_logger(), "Number of initial matches: %zu", all_matches.size());

                std::vector<cv::DMatch> distance_filtered_matches;
                float max_distance = 50.0f; 

                for (const auto& match : all_matches) {
                    if (match.distance < max_distance) {
                        distance_filtered_matches.push_back(match);
                    }
                }

                std::vector<cv::DMatch> geometrically_consistent_matches;

                if (distance_filtered_matches.size() >= 8) {
                    std::vector<cv::Point2f> prev_pts, curr_pts;
                    for (const auto& match : distance_filtered_matches) {
                        prev_pts.push_back(prev_kps_[match.trainIdx].pt);
                        curr_pts.push_back(current_keypoints[match.queryIdx].pt);
                    }
                    
                    std::vector<uchar> inliers_mask;
                    cv::Mat fundamental_matrix = cv::findFundamentalMat(prev_pts, curr_pts, inliers_mask, cv::FM_RANSAC, 2.0, 0.99);

                    for (size_t i = 0; i < inliers_mask.size(); i++) {
                        if (inliers_mask[i]) {
                            geometrically_consistent_matches.push_back(distance_filtered_matches[i]);
                        }
                    }
                    
                    RCLCPP_DEBUG(this->get_logger(), "Fundamental matrix RANSAC kept %zu/%zu matches", 
                                geometrically_consistent_matches.size(), distance_filtered_matches.size());
                } else {
                    geometrically_consistent_matches = distance_filtered_matches;
                    RCLCPP_WARN(this->get_logger(), "Insufficient matches for RANSAC: %zu", distance_filtered_matches.size());
                }

                std::vector<cv::DMatch> good_matches = geometrically_consistent_matches;

                std::vector<cv::Point2f> prev_matched_pts, curr_matched_pts;
                for (const auto& match : good_matches) {
                    prev_matched_pts.push_back(prev_kps_[match.trainIdx].pt);
                    curr_matched_pts.push_back(current_keypoints[match.queryIdx].pt);
                    cv::Point2f curr_pt = current_keypoints[match.queryIdx].pt;
                    cv::circle(vis_image, curr_pt, 3, cv::Scalar(0, 255, 0), 2);
                }

                std::vector<uchar> status(good_matches.size(), 1);

                if (good_matches.size() >= 5) {
                    estimateCameraPose(prev_kps_, current_keypoints, good_matches, prev_frame_depth_, rgb_msg->header.stamp);
                }

                if (isKeyframe(current_descriptors)) {
                    publishKeyframe(current_keypoints, current_descriptors, current_frame_depth, rgb_msg->header.stamp);
                }

                prev_points_.clear();
                for (const auto& kp : current_keypoints) {
                    prev_points_.push_back(kp.pt);
                }

                prev_descriptors_ = current_descriptors;
                prev_kps_ = current_keypoints;
                
                sensor_msgs::msg::Image::SharedPtr out_msg = cv_bridge::CvImage(rgb_msg->header, "bgr8", vis_image).toImageMsg();
                image_pub_->publish(*out_msg);

                // Publish camera info if available
                if (latest_rgb_camera_info_) {
                    auto camera_info = *latest_rgb_camera_info_;
                    camera_info.header = rgb_msg->header;
                    camera_info_pub_->publish(camera_info);
                }
                
                current_frame_gray.copyTo(prev_frame_gray_);
                current_frame_depth.copyTo(prev_frame_depth_);
            } 
            else {
                std::vector<cv::KeyPoint> current_keypoints;
                cv::Mat current_descriptors;
                orb_detector_->detectAndCompute(current_frame_gray, depth_mask, current_keypoints, current_descriptors);

                prev_points_.clear();
                for (const auto& kp : current_keypoints) {
                    prev_points_.push_back(kp.pt);
                }

                prev_descriptors_ = current_descriptors;
                prev_kps_ = current_keypoints;

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