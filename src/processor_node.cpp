#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>

class Processor : public rclcpp::Node
{
public:
    Processor() : Node("processor")
    {
        // Create a subscriber for the camera image topic
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw",
            10,
            std::bind(&Processor::imageCallback, this, std::placeholders::_1));
            
        RCLCPP_INFO(this->get_logger(), "Image processor node initialized");
    }

private:
    // Subscriber to the camera image topic
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    
    // Callback function for processing incoming images
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        RCLCPP_DEBUG(this->get_logger(), "Received image, size: %dx%d", 
                    msg->width, msg->height);
        
        try {
            // Convert ROS image message to OpenCV image
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            cv::Mat image = cv_ptr->image;
            
            // Process the image (placeholder for your SLAM algorithm)
            // For now, just display basic image info
            RCLCPP_DEBUG(this->get_logger(), "Processing image with dimensions: %dx%d",
                       image.cols, image.rows);
            
            // You'll add your SLAM processing code here
            
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Processor>());
    rclcpp::shutdown();
    return 0;
}