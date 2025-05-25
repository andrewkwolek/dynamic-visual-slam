#include <opencv2/opencv.hpp>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "dynamic_visual_slam/bundle_adjustment.hpp"
#include "dynamic_visual_slam_interfaces/msg/keyframe.hpp"

class Backend : public rclcpp::Node 
{
public:
    Backend() : Node("backend") {
        // subscribe to keyframe topic
        rclcpp::QoS qos = rclcpp::QoS(30);


        bundle_adjuster_ = std::make_unique<SlidingWindowBA>(10, 0.0, 0.0, 0.0, 0.0);

        keyframe_sub_ = this->create_subscription<dynamic_visual_slam_interfaces::msg::Keyframe>("/frontend/keyframe", qos, std::bind(&Backend::keyframeCallback, this, std::placeholders::_1));
    }

private:
    std::unique_ptr<SlidingWindowBA> bundle_adjuster_;

    rclcpp::Subscription<dynamic_visual_slam_interfaces::msg::Keyframe>::SharedPtr keyframe_sub_;

    void keyframeCallback(const dynamic_visual_slam_interfaces::msg::Keyframe::ConstSharedPtr& msg) {
        RCLCPP_INFO(this->get_logger(), "Received keyframe!");
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Backend>());
    rclcpp::shutdown();
    return 0;
}