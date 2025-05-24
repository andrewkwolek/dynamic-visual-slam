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
#include "dynamic-visual-slam/bundle_adjustment.hpp"

class FeatureDetector : public rclcpp::Node
{
public:
    FeatureDetector() : Node("feature_detector")
    {
        rclcpp::QoS qos = rclcpp::QoS(30);

        // Create message filter subscribers
        rgb_sub_.subscribe(this, "/camera/camera/color/image_raw", qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, "/camera/camera/aligned_depth_to_color/image_raw", qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>(10), rgb_sub_, depth_sub_);
        sync_->registerCallback(std::bind(&FeatureDetector::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        // Create subscriptions
        rgb_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera/camera/color/camera_info", qos, std::bind(&FeatureDetector::rgbInfoCallback, this, std::placeholders::_1));
        depth_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera/camera/aligned_depth_to_color/camera_info", qos, std::bind(&FeatureDetector::depthInfoCallback, this, std::placeholders::_1));

        // Create publishers
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/feature_detector/features_image", qos);
        camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/feature_detector/camera_info", qos);
        
        // Initialize ORB feature detector
        // orb_detector_ = cv::ORB::create(800);

        orb_detector_ = cv::ORB::create(
            1200,        // More features for wider FOV
            1.2f,        // Scale factor
            8,           // Pyramid levels  
            31,          // Edge threshold
            0,           // First level
            2,           // WTA_K
            cv::ORB::HARRIS_SCORE,
            31,          // Patch size
            10           // Fast threshold - adjust for D455 noise
        );

        // bundle_adjuster_ = std::make_unique<SlidingWindowBA>(30, 0.0, 0.0, 0.0, 0.0);

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

    // Camera info
    sensor_msgs::msg::CameraInfo::SharedPtr latest_rgb_camera_info_;
    sensor_msgs::msg::CameraInfo::SharedPtr latest_depth_camera_info_;
    cv::Mat rgb_camera_matrix_;
    cv::Mat rgb_dist_coeffs_;
    cv::Mat depth_camera_matrix_;
    cv::Mat depth_dist_coeffs_;

    // ORB Feature detector
    cv::Ptr<cv::ORB> orb_detector_;

    // Bundle adjustment
    std::unique_ptr<SlidingWindowBA> bundle_adjuster_;

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
        // Maximum allowed translation between frames (meters)
        const double MAX_TRANSLATION = 0.5;
        // Maximum allowed rotation between frames (radians)
        const double MAX_ROTATION = 0.2;
        
        // Check translation magnitude
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

    void estimateCameraPose(const std::vector<cv::Point2f>& prev_pts, const std::vector<cv::Point2f>& curr_pts, const std::vector<uchar>& status, const cv::Mat& prev_depth, const rclcpp::Time& stamp, const std::vector<cv::DMatch>& good_matches) {
        // Only use points that were successfully tracked
        std::vector<cv::Point3f> points3d;
        std::vector<cv::Point2f> points2d;

        if (rgb_camera_matrix_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "RGB camera matrix is empty!");
            return;
        }

        float fx = rgb_camera_matrix_.at<double>(0, 0);
        float fy = rgb_camera_matrix_.at<double>(1, 1);
        float cx = rgb_camera_matrix_.at<double>(0, 2);
        float cy = rgb_camera_matrix_.at<double>(1, 2);

        try {
            for (size_t i = 0; i < status.size(); i++) {
                if (!status[i]) continue;

                float x_prev = prev_pts[i].x;
                float y_prev = prev_pts[i].y;
                
                // Get the depth value at the previous point
                float d_prev = prev_depth.at<uint16_t>(y_prev, x_prev) * 0.001f;
                
                // Skip points with invalid depth
                if (d_prev <= 0.6f || d_prev > 6.0f) {
                    RCLCPP_WARN(this->get_logger(), "Depth: %f", d_prev);
                    continue;
                }
                
                // Back-project to 3D point
                cv::Point3f pt3d_prev((x_prev - cx) * d_prev / fx, (y_prev - cy) * d_prev / fy, d_prev);
                
                // Use current 2D points for PnP
                points3d.push_back(pt3d_prev);
                points2d.push_back(curr_pts[i]);
            }
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during point processing: %s", e.what());
            return;
        }

        // Need at least 4 matching points for PnP
        if (points3d.size() < 4) {
            RCLCPP_WARN(this->get_logger(), "Not enough matching points for pose estimation: %zu", points3d.size());
            return;
        }

        try {
            // Use solvePnPRANSAC to estimate pose
            cv::Mat rvec, tvec, inliers;
            bool use_initial_guess = false;
            
            cv::Mat T_ros_to_optical = cv::Mat::eye(4, 4, CV_64F);
            
            // For RealSense cameras and standard ROS conventions:
            // X_optical = -Y_ros
            // Y_optical = -Z_ros
            // Z_optical = X_ros
            T_ros_to_optical.at<double>(0, 0) = 0.0;   // ROS -Y → Optical X
            T_ros_to_optical.at<double>(0, 1) = -1.0;
            T_ros_to_optical.at<double>(0, 2) = 0.0;
                        
            T_ros_to_optical.at<double>(1, 0) = 0.0;   // ROS -Z → Optical Y
            T_ros_to_optical.at<double>(1, 1) = 0.0;
            T_ros_to_optical.at<double>(1, 2) = -1.0;
                        
            T_ros_to_optical.at<double>(2, 0) = 1.0;   // ROS X → Optical Z
            T_ros_to_optical.at<double>(2, 1) = 0.0;
            T_ros_to_optical.at<double>(2, 2) = 0.0;

            // Use previous pose as initial guess if available
            if (!R_.empty() && !t_.empty()) {
                // Create transformation matrix in ROS frame
                cv::Mat T_ros = cv::Mat::eye(4, 4, CV_64F);
                R_.copyTo(T_ros(cv::Rect(0, 0, 3, 3)));
                t_.copyTo(T_ros(cv::Rect(3, 0, 1, 3)));
                
                // Transform to optical frame
                cv::Mat T_optical = T_ros_to_optical.inv() * T_ros * T_ros_to_optical;
                
                // Extract the rotation and translation in optical frame
                cv::Mat R_optical = T_optical(cv::Rect(0, 0, 3, 3));
                cv::Mat t_optical = T_optical(cv::Rect(3, 0, 1, 3));
                
                // Convert rotation matrix to rotation vector for PnP
                cv::Rodrigues(R_optical, rvec);
                tvec = t_optical.clone();
                use_initial_guess = true;
            }

            bool success = cv::solvePnP(
                points3d,           // 3D points from previous frame
                points2d,           // 2D points in current frame
                rgb_camera_matrix_, // Camera matrix
                rgb_dist_coeffs_,   // Distortion coefficients
                rvec,               // Output rotation vector
                tvec,               // Output translation vector
                use_initial_guess  // Use provided R,t as initial guess?
            );

            if (!success) {
                RCLCPP_WARN(this->get_logger(), "PnP estimation failed.");
                return;
            }

            // Convert rotation vector to rotation matrix
            cv::Mat R_curr;
            cv::Rodrigues(rvec, R_curr);

            // Update the global pose (camera_link in world frame)
            // First convert current camera-to-previous transformation to previous-to-current
            cv::Mat R_curr_inv = R_curr.t();
            cv::Mat t_curr_inv = -R_curr_inv * tvec;

            // Define the transformation matrix from camera optical frame to ROS frame
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

            // Create transformation matrix in optical frame
            cv::Mat T_optical = cv::Mat::eye(4, 4, CV_64F);
            R_curr_inv.copyTo(T_optical(cv::Rect(0, 0, 3, 3)));
            t_curr_inv.copyTo(T_optical(cv::Rect(3, 0, 1, 3)));
            
            // Transform to ROS frame
            // This formula converts a transform expressed in optical frame to one expressed in ROS frame
            cv::Mat T_ros = T_optical_to_ros * T_optical * T_optical_to_ros.inv();
            
            // Extract the new rotation and translation in ROS frame
            cv::Mat R_ros = T_ros(cv::Rect(0, 0, 3, 3));
            cv::Mat t_ros = T_ros(cv::Rect(3, 0, 1, 3));
            
            // Update the cumulative pose using the ROS frame transforms
            t_ = t_ + R_ * t_ros;
            R_ = R_ * R_ros;
            
            // // Add the current frame to the bundle adjustment
            // int frame_id = bundle_adjuster_->addFrame(R_, t_);
            
            // // Add the observed points to bundle adjustment
            // for (int i = 0; i < inliers.rows; i++) {
            //     int idx = inliers.at<int>(i, 0);
                
            //     // Add the 3D point and its 2D observation to bundle adjustment
            //     bundle_adjuster_->addObservation(
            //         frame_id,                  // Current frame ID
            //         points2d[idx].x,           // 2D observation x
            //         points2d[idx].y,           // 2D observation y
            //         points3d[idx].x,           // 3D point X
            //         points3d[idx].y,           // 3D point Y
            //         points3d[idx].z            // 3D point Z
            //     );
            // }
            
            // // Run bundle adjustment optimization
            // bundle_adjuster_->optimize(30);
            
            // // Get the optimized pose for the latest frame
            // auto optimized_pose = bundle_adjuster_->getLatestPose();
            // R_ = optimized_pose.first;
            // t_ = optimized_pose.second;

            if (isMotionOutlier(R_ros, t_ros)) {
                RCLCPP_ERROR(this->get_logger(), "Good matches: %ld", good_matches.size());
            }

            // Broadcast the transform
            broadcastTransform(stamp);

            // Print the current pose for debugging
            RCLCPP_DEBUG(this->get_logger(), "Camera position: [%f, %f, %f]", t_.at<double>(0), t_.at<double>(1), t_.at<double>(2));
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during PnP estimation: %s", e.what());
            return;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during bundle adjustment: %s", e.what());
            return;
        }
    }

    void rgbInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_rgb_camera_info_ = msg;

        rgb_camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        rgb_camera_matrix_.at<double>(0, 0) = msg->k[0];
        rgb_camera_matrix_.at<double>(0, 2) = msg->k[2];
        rgb_camera_matrix_.at<double>(1, 1) = msg->k[4];
        rgb_camera_matrix_.at<double>(1, 2) = msg->k[5];
        rgb_camera_matrix_.at<double>(2, 2) = 1.0;

        rgb_dist_coeffs_ = cv::Mat(1, 5, CV_64F);
        for (int i = 0; i < 5; i++) {
            rgb_dist_coeffs_.at<double>(0, i) = msg->d[i];
        }

        // if (!bundle_adjuster_) {
        //     bundle_adjuster_ = std::make_unique<SlidingWindowBA>(
        //         10,  // Window size (increased from 5 to 10 for stability)
        //         rgb_camera_matrix_.at<double>(0, 0),  // fx
        //         rgb_camera_matrix_.at<double>(1, 1),  // fy
        //         rgb_camera_matrix_.at<double>(0, 2),  // cx
        //         rgb_camera_matrix_.at<double>(1, 2)   // cy
        //     );
        //     RCLCPP_INFO(this->get_logger(), "Initialized Bundle Adjuster with camera parameters");
        // }
    }

    void depthInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_depth_camera_info_ = msg;

        depth_camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        depth_camera_matrix_.at<double>(0, 0) = msg->k[0];
        depth_camera_matrix_.at<double>(0, 2) = msg->k[2];
        depth_camera_matrix_.at<double>(1, 1) = msg->k[4];
        depth_camera_matrix_.at<double>(1, 2) = msg->k[5];
        depth_camera_matrix_.at<double>(2, 2) = 1.0;

        depth_dist_coeffs_ = cv::Mat(1, 5, CV_64F);
        for (int i = 0; i < 5; i++) {
            depth_dist_coeffs_.at<double>(0, i) = msg->d[i];
        }
    }
    
    // Callback function for processing incoming images
    void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
    {
        try {
            // Convert ROS image message to OpenCV image
            cv_bridge::CvImagePtr rgb_cv_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
            cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv::Mat current_frame_rgb = rgb_cv_ptr->image;
            cv::Mat current_frame_depth = depth_cv_ptr->image;
            cv::Mat current_frame_gray;

            cv::cvtColor(current_frame_rgb, current_frame_gray, cv::COLOR_BGR2GRAY);

            cv::Mat depth_mask;
            depth_mask = (current_frame_depth > 600) & (current_frame_depth < 6000);

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

                // Optional: Filter by descriptor distance
                std::vector<cv::DMatch> distance_filtered_matches;
                float max_distance = 50.0f; 

                for (const auto& match : all_matches) {
                    if (match.distance < max_distance) {
                        distance_filtered_matches.push_back(match);
                    }
                }

                // Apply RANSAC with fundamental matrix for geometric consistency
                std::vector<cv::DMatch> geometrically_consistent_matches;

                if (distance_filtered_matches.size() >= 8) {
                    // Extract point correspondences
                    std::vector<cv::Point2f> prev_pts, curr_pts;
                    for (const auto& match : distance_filtered_matches) {
                        prev_pts.push_back(prev_kps_[match.trainIdx].pt);
                        curr_pts.push_back(current_keypoints[match.queryIdx].pt);
                    }
                    
                    // Use fundamental matrix RANSAC for outlier detection
                    std::vector<uchar> inliers_mask;
                    cv::Mat fundamental_matrix = cv::findFundamentalMat(prev_pts, curr_pts, inliers_mask, cv::FM_RANSAC, 2.0, 0.99);
                    
                    // Extract geometrically consistent matches (these are your "good_matches")
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

                // Now use these inliers for pose estimation
                std::vector<cv::DMatch> good_matches = geometrically_consistent_matches;

                // Convert to the format your existing pose estimation expects
                std::vector<cv::Point2f> prev_matched_pts, curr_matched_pts;
                for (const auto& match : good_matches) {
                    prev_matched_pts.push_back(prev_kps_[match.trainIdx].pt);
                    curr_matched_pts.push_back(current_keypoints[match.queryIdx].pt);
                    cv::Point2f curr_pt = current_keypoints[match.queryIdx].pt;
                    cv::circle(vis_image, curr_pt, 3, cv::Scalar(0, 255, 0), 2);
                }

                // Create status vector (all 1s since these are already filtered inliers)
                std::vector<uchar> status(good_matches.size(), 1);

                // Call your existing pose estimation function
                if (good_matches.size() >= 5) {
                    estimateCameraPose(prev_matched_pts, curr_matched_pts, status, prev_frame_depth_, rgb_msg->header.stamp, good_matches);
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

                // Publish camera info if available
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
    rclcpp::spin(std::make_shared<FeatureDetector>());
    rclcpp::shutdown();
    return 0;
}