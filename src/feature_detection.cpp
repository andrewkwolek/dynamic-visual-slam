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

class FeatureDetector : public rclcpp::Node
{
public:
    FeatureDetector() : Node("feature_detector")
    {
        // Create subscribers
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw",
            10,
            std::bind(&FeatureDetector::imageCallback, this, std::placeholders::_1));

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/color/camera_info",
            10,
            std::bind(&FeatureDetector::cameraInfoCallback, this, std::placeholders::_1));
        
        // Create publishers
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/feature_detector/features_image",
            10
        );

        camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            "/feature_detector/camera_info",
            10
        );

        orb_detector_ = cv::ORB::create();

        prev_frame_valid_ = false;

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        flann_ = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(5), cv::makePtr<cv::flann::SearchParams>(50));

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

        camera_matrix_ = cv::Mat();
        dist_coeffs_ = cv::Mat();
            
        RCLCPP_INFO(this->get_logger(), "Image processor node initialized");
    }

private:
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    //Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;

    // Camera info
    sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    // ORB Feature detector
    cv::Ptr<cv::ORB> orb_detector_;

    // FLANN feature matcher
    cv::FlannBasedMatcher flann_;

    // Previous frame info
    cv::Mat prev_frame_;
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
        transform_stamped.child_frame_id = "camera";
        
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

    void estimateCameraPose(const std::vector<cv::Point2f>& prev_pts, const std::vector<cv::Point2f>& curr_pts, const std::vector<uchar>& status, const rclcpp::Time& stamp) {
        // Only use points that were successfully tracked
        std::vector<cv::Point2f> points1, points2;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                points1.push_back(prev_pts[i]);
                points2.push_back(curr_pts[i]);
            }
        }

        // Need at least 5 matching points
        if (points1.size() < 5 || camera_matrix_.empty()) {
            RCLCPP_WARN(this->get_logger(), "Not enough matching points or no camera info for pose estimation");
            return;
        }

        // Find the essential matrix
        cv::Mat E = cv::findEssentialMat(points1, points2, camera_matrix_, cv::RANSAC, 0.999, 1.0);

        // Recover R and t from the essential matrix
        cv::Mat R, t;
        cv::recoverPose(E, points1, points2, camera_matrix_, R, t);

        // Update the cumulative pose
        t_ = t_ + R_ * t;
        R_ = R * R_;

        // Broadcast the transform
        broadcastTransform(stamp);

        // Print the current pose for debugging
        RCLCPP_DEBUG(this->get_logger(), "Camera position: [%f, %f, %f]", t_.at<double>(0), t_.at<double>(1), t_.at<double>(2));
    }

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_camera_info_ = msg;

        camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        camera_matrix_.at<double>(0, 0) = msg->k[0];
        camera_matrix_.at<double>(0, 2) = msg->k[2];
        camera_matrix_.at<double>(1, 1) = msg->k[4];
        camera_matrix_.at<double>(1, 2) = msg->k[5];
        camera_matrix_.at<double>(2, 2) = 1.0;

        dist_coeffs_ = cv::Mat(1, 5, CV_64F);
        for (int i = 0; i < 5; i++) {
            dist_coeffs_.at<double>(0, i) = msg->d[i];
        }
    }
    
    // Callback function for processing incoming images
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // Convert ROS image message to OpenCV image
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            cv::Mat current_frame = cv_ptr->image;
            cv::Mat current_frame_gray;

            cv::cvtColor(current_frame, current_frame_gray, cv::COLOR_BGR2GRAY);

            if (prev_frame_valid_) {
                cv::Mat vis_image = current_frame.clone();
                
                std::vector<cv::KeyPoint> current_keypoints;
                cv::Mat current_descriptors;
                orb_detector_->detectAndCompute(current_frame_gray, cv::noArray(), current_keypoints, current_descriptors);

                std::vector<std::vector<cv::DMatch>> matches;
                flann_.knnMatch(current_descriptors, prev_descriptors_, matches, 2);

                std::vector<char> matchesMask(matches.size(), 0);

                for (size_t i = 0; i < matches.size(); i++) {
                    if (matches[i].size() < 2) continue;
                    
                    if (matches[i][0].distance < 0.7 * matches[i][1].distance) {
                        matchesMask[i] = 1;
                    }
                }

                std::vector<cv::DMatch> good_matches;
                for (size_t i = 0; i < matches.size(); i++) {
                    if (matchesMask[i]) {
                        good_matches.push_back(matches[i][0]);
                    }
                }

                cv::drawMatches(current_frame, current_keypoints, prev_frame_, prev_kps_, good_matches, vis_image, cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);

                prev_points_.clear();
                for (const auto& kp : current_keypoints) {
                    prev_points_.push_back(kp.pt);
                }

                prev_descriptors_ = current_descriptors;
                prev_kps_ = current_keypoints;
                
                sensor_msgs::msg::Image::SharedPtr out_msg = cv_bridge::CvImage(msg->header, "bgr8", vis_image).toImageMsg();
                image_pub_->publish(*out_msg);

                geometry_msgs::msg::TransformStamped camera_transform;
                camera_transform.header.stamp = this->get_clock()->now();
                camera_transform.header.frame_id = "odom";
                camera_transform.child_frame_id = "camera_link";
                camera_transform.transform.translation.x = 0.0;
                camera_transform.transform.translation.y = 0.0;
                camera_transform.transform.translation.z = 0.0;
                camera_transform.transform.rotation.x = 0.0;
                camera_transform.transform.rotation.y = 0.0;
                camera_transform.transform.rotation.z = 0.0;
                camera_transform.transform.rotation.w = 1.0;

                tf_broadcaster_->sendTransform(camera_transform);

                // Publish camera info if available
                if (latest_camera_info_) {
                    auto camera_info = *latest_camera_info_;
                    camera_info.header = msg->header;  // Use the same header as the image
                    camera_info_pub_->publish(camera_info);
                }
                
                current_frame_gray.copyTo(prev_frame_);
            } 
            else {
                std::vector<cv::KeyPoint> current_keypoints;
                cv::Mat current_descriptors;
                orb_detector_->detectAndCompute(current_frame_gray, cv::noArray(), current_keypoints, current_descriptors);

                prev_points_.clear();
                for (const auto& kp : current_keypoints) {
                    prev_points_.push_back(kp.pt);
                }

                prev_descriptors_ = current_descriptors;
                prev_kps_ = current_keypoints;

                // Publish camera info if available
                if (latest_camera_info_) {
                    auto camera_info = *latest_camera_info_;
                    camera_info.header = msg->header;  // Use the same header as the image
                    camera_info_pub_->publish(camera_info);
                }
                
                current_frame_gray.copyTo(prev_frame_);
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