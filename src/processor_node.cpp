#include "rclcpp/rclcpp.hpp"

class Processor : public rclcpp::Node
{
public:
    Processor() : Node("processor")
    {
        
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Processor>());
    rclcpp::shutdown();
    return 0;
}