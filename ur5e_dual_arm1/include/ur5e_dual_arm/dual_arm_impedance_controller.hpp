// include/ur5e_dual_arm/dual_arm_impedance_controller.hpp
#pragma once

#include <controller_interface/controller_interface.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace dual_arm_controller
{

class DualArmImpedanceController : public controller_interface::ControllerInterface
{
public:
  DualArmImpedanceController();

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  // Joints
  std::vector<std::string> left_joints_;
  std::vector<std::string> right_joints_;

  // Impedance gains
  Eigen::MatrixXd Kp_left_, Kd_left_;
  Eigen::MatrixXd Kp_right_, Kd_right_;

  // Coordination gain
  double k_coord_;

  // Desired joint positions (set by task node via topic)
  Eigen::VectorXd q_des_left_, q_des_right_;
  Eigen::VectorXd qd_des_left_, qd_des_right_;

  // State
  Eigen::VectorXd q_left_, qd_left_;
  Eigen::VectorXd q_right_, qd_right_;

  // Command output
  Eigen::VectorXd cmd_left_, cmd_right_;

  // Coordination: desired relative joint offset between arms
  Eigen::VectorXd coord_offset_;  // q_left - q_right desired difference

  // ROS interfaces
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr left_target_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr right_target_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr data_pub_;

  std::mutex target_mutex_;
};

}  // namespace dual_arm_controller