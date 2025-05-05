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

class FeatureDetector : public rclcpp::Node
{
public:
    FeatureDetector() : Node("feature_detector")
    {
        rclcpp::QoS qos = rclcpp::QoS(10);

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
        orb_detector_ = cv::ORB::create();

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

    void estimateCameraPose(const std::vector<cv::Point2f>& prev_pts, const std::vector<cv::Point2f>& curr_pts, const std::vector<uchar>& status, const cv::Mat& prev_depth, const cv::Mat& curr_depth, const rclcpp::Time& stamp) {
        // Only use points that were successfully tracked
        std::vector<cv::Point3f> points1, points2;

        float fx = rgb_camera_matrix_.at<double>(0, 0);
        float fy = rgb_camera_matrix_.at<double>(1, 1);
        float cx = rgb_camera_matrix_.at<double>(0, 2);
        float cy = rgb_camera_matrix_.at<double>(1, 2);

        for (size_t i = 0; i < status.size(); i++) {
            if (!status[i]) continue;

            int x_prev = std::round(prev_pts[i].x);
            int y_prev = std::round(prev_pts[i].y);
            int x_curr = std::round(curr_pts[i].x);
            int y_curr = std::round(curr_pts[i].y);

            float d_prev = prev_depth.at<uint16_t>(y_prev, x_prev) * 0.001f;
            float d_curr = curr_depth.at<uint16_t>(y_curr, x_curr) * 0.001f;

            cv::Point3f pt3d_prev((x_prev - cx) * d_prev / fx, (y_prev - cy) * d_prev / fy, d_prev);
            cv::Point3f pt3d_curr((x_curr - cx) * d_curr / fx, (y_curr - cy) * d_curr / fy, d_curr);

            points1.push_back(pt3d_prev);
            points2.push_back(pt3d_curr);
        }

        // Need at least 5 matching points
        if (points1.size() < 4) {
            RCLCPP_WARN(this->get_logger(), "Not enough matching points or no camera info for pose estimation");
            return;
        }

        // Recover R and t from the essential matrix
        cv::Mat T, R, t, inliers;
        if (!cv::estimateAffine3D(points1, points2, T, inliers, 0.03)) {
            RCLCPP_WARN(this->get_logger(), "Failed to estimate rigid transformation");
            return;
        }

        R = T(cv::Range(0, 2), cv::Range(0, 2));
        t = T(cv::Range(0, 2), cv::Range(3, 3));

        cv::SVD svd(R);
        R = svd.u * svd.vt;
        // Update the cumulative pose
        t_ = t_ + R_ * t;
        R_ = R * R_;

        // Broadcast the transform
        broadcastTransform(stamp);

        // Print the current pose for debugging
        RCLCPP_DEBUG(this->get_logger(), "Camera position: [%f, %f, %f]", t_.at<double>(0), t_.at<double>(1), t_.at<double>(2));
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
            depth_mask = (current_frame_depth > 0) & (current_frame_depth < 10000);

            if (prev_frame_valid_) {
                cv::Mat vis_image = current_frame_rgb.clone();
                
                std::vector<cv::KeyPoint> current_keypoints;
                cv::Mat current_descriptors;
                orb_detector_->detectAndCompute(current_frame_gray, depth_mask, current_keypoints, current_descriptors);

                if (current_keypoints.empty() || prev_kps_.empty() || current_descriptors.empty() || prev_descriptors_.empty()) {
                    RCLCPP_WARN(this->get_logger(), "No featured detected in one of the frames.");
                    current_frame_gray.copyTo(prev_frame_gray_);
                    current_frame_depth.copyTo(prev_frame_depth_);
                    prev_kps_ = current_keypoints;
                    prev_descriptors_ = current_descriptors;
                    return;
                }

                std::vector<std::vector<cv::DMatch>> matches;
                matcher_.knnMatch(current_descriptors, prev_descriptors_, matches, 2);

                std::vector<cv::DMatch> good_matches;
                for (size_t i = 0; i < matches.size(); i++) {
                    if (matches[i].size() < 2) continue;
                    
                    if (matches[i][0].distance < 0.7 * matches[i][1].distance) {
                        good_matches.push_back(matches[i][0]);
                    }
                }

                for (const auto& match : good_matches) {
                    // Get the matched points
                    cv::Point2f curr_pt = current_keypoints[match.queryIdx].pt;
                    
                    // Draw a green circle
                    cv::circle(vis_image, curr_pt, 5, cv::Scalar(0, 255, 0), 2);
                }

                if (good_matches.size() >= 5) {
                    // Convert keypoints to points for pose estimation
                    std::vector<cv::Point2f> prev_matched_pts, curr_matched_pts;
                    for (const auto& match : good_matches) {
                        prev_matched_pts.push_back(prev_kps_[match.trainIdx].pt);
                        curr_matched_pts.push_back(current_keypoints[match.queryIdx].pt);
                    }
                    
                    // Create status vector (all 1s since these are good matches)
                    std::vector<uchar> status(good_matches.size(), 1);
                    
                    // Estimate camera pose
                    estimateCameraPose(prev_matched_pts, curr_matched_pts, status, prev_frame_depth_, current_frame_depth, rgb_msg->header.stamp);
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
                    camera_info.header = rgb_msg->header;  // Use the same header as the image
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
                    camera_info.header = rgb_msg->header;  // Use the same header as the image
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