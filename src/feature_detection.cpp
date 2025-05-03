#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

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
            
        RCLCPP_INFO(this->get_logger(), "Image processor node initialized");
    }

private:
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    //Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;

    sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;

    cv::Ptr<cv::ORB> orb_detector_;

    cv::Mat prev_frame_;
    std::vector<cv::Point2f> prev_points_;
    bool prev_frame_valid_;

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_camera_info_ = msg;
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
                
                if (!prev_points_.empty()) {
                    std::vector<cv::Point2f> current_points;
                    std::vector<uchar> status;
                    std::vector<float> err;
                    
                    cv::calcOpticalFlowPyrLK(
                        prev_frame_, current_frame_gray, 
                        prev_points_, current_points,
                        status, err);
                    
                    for (size_t i = 0; i < current_points.size(); i++) {
                        if (status[i]) {
                            cv::line(vis_image, prev_points_[i], current_points[i], 
                                   cv::Scalar(0, 255, 0), 2);
                            cv::circle(vis_image, current_points[i], 3, 
                                     cv::Scalar(0, 0, 255), -1);
                        }
                    }
                }
                
                std::vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                orb_detector_->detectAndCompute(current_frame_gray, cv::noArray(), 
                                              keypoints, descriptors);
                
                prev_points_.clear();
                for (const auto& kp : keypoints) {
                    prev_points_.push_back(kp.pt);
                }
                
                cv::drawKeypoints(vis_image, keypoints, vis_image, 
                                cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
                
                sensor_msgs::msg::Image::SharedPtr out_msg = 
                    cv_bridge::CvImage(msg->header, "bgr8", vis_image).toImageMsg();
                image_pub_->publish(*out_msg);

                // Publish camera info if available
                if (latest_camera_info_) {
                    auto camera_info = *latest_camera_info_;
                    camera_info.header = msg->header;  // Use the same header as the image
                    camera_info_pub_->publish(camera_info);
                }
                
                current_frame_gray.copyTo(prev_frame_);
            } 
            else {
                std::vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                orb_detector_->detectAndCompute(current_frame_gray, cv::noArray(), 
                                             keypoints, descriptors);
                
                prev_points_.clear();
                for (const auto& kp : keypoints) {
                    prev_points_.push_back(kp.pt);
                }
                
                cv::Mat vis_image = current_frame.clone();
                cv::drawKeypoints(vis_image, keypoints, vis_image, 
                                cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
                
                sensor_msgs::msg::Image::SharedPtr out_msg = 
                    cv_bridge::CvImage(msg->header, "bgr8", vis_image).toImageMsg();
                image_pub_->publish(*out_msg);

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