#include <opencv2/opencv.hpp>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "dynamic_visual_slam/bundle_adjustment.hpp"

class Backend : public rclcpp::Node 
{
public:
    Backend() : Node("backend") {
        // subscribe to keyframe topic
        rclcpp::QoS qos = rclcpp::QoS(30);


        bundle_adjuster_ = std::make_unique<SlidingWindowBA>(10, 0.0, 0.0, 0.0, 0.0);
    }

private:
    std::unique_ptr<SlidingWindowBA> bundle_adjuster_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Backend>());
    rclcpp::shutdown();
    return 0;
}