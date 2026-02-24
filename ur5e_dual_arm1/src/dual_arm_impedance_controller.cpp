// src/dual_arm_impedance_controller.cpp
#include "ur5e_dual_arm/dual_arm_impedance_controller.hpp"
#include <pluginlib/class_list_macros.hpp>

namespace dual_arm_controller
{

DualArmImpedanceController::DualArmImpedanceController()
: controller_interface::ControllerInterface() {}

controller_interface::CallbackReturn
DualArmImpedanceController::on_init()
{
  left_joints_ = {
    "left_shoulder_pan_joint", "left_shoulder_lift_joint", "left_elbow_joint",
    "left_wrist_1_joint", "left_wrist_2_joint", "left_wrist_3_joint"};
  right_joints_ = {
    "right_shoulder_pan_joint", "right_shoulder_lift_joint", "right_elbow_joint",
    "right_wrist_1_joint", "right_wrist_2_joint", "right_wrist_3_joint"};

  int n = 6;
  // Stiffness and damping (tune these — key deliverable!)
  Kp_left_  = Eigen::MatrixXd::Identity(n, n) * 200.0;
  Kd_left_  = Eigen::MatrixXd::Identity(n, n) * 20.0;
  Kp_right_ = Eigen::MatrixXd::Identity(n, n) * 200.0;
  Kd_right_ = Eigen::MatrixXd::Identity(n, n) * 20.0;

  k_coord_ = 50.0;  // Coordination coupling strength

  q_des_left_  = Eigen::VectorXd::Zero(n);
  q_des_right_ = Eigen::VectorXd::Zero(n);
  qd_des_left_ = Eigen::VectorXd::Zero(n);
  qd_des_right_= Eigen::VectorXd::Zero(n);
  q_left_  = Eigen::VectorXd::Zero(n);
  qd_left_ = Eigen::VectorXd::Zero(n);
  q_right_  = Eigen::VectorXd::Zero(n);
  qd_right_ = Eigen::VectorXd::Zero(n);
  cmd_left_  = Eigen::VectorXd::Zero(n);
  cmd_right_ = Eigen::VectorXd::Zero(n);

  // Desired symmetric coordination: left mirrors right on pan joint
  coord_offset_ = Eigen::VectorXd::Zero(n);
  coord_offset_(0) = 0.0;  // adjust per task

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
DualArmImpedanceController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration cfg;
  cfg.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (auto & j : left_joints_)
    cfg.names.push_back(j + "/position");
  for (auto & j : right_joints_)
    cfg.names.push_back(j + "/position");
  return cfg;
}

controller_interface::InterfaceConfiguration
DualArmImpedanceController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration cfg;
  cfg.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (auto & j : left_joints_) {
    cfg.names.push_back(j + "/position");
    cfg.names.push_back(j + "/velocity");
  }
  for (auto & j : right_joints_) {
    cfg.names.push_back(j + "/position");
    cfg.names.push_back(j + "/velocity");
  }
  return cfg;
}

controller_interface::CallbackReturn
DualArmImpedanceController::on_configure(const rclcpp_lifecycle::State &)
{
  // Subscribe to target joint positions from the task node
  left_target_sub_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/left_arm/joint_target", 10,
    [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(target_mutex_);
      if (msg->data.size() == 6)
        for (int i = 0; i < 6; i++) q_des_left_(i) = msg->data[i];
    });

  right_target_sub_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/right_arm/joint_target", 10,
    [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(target_mutex_);
      if (msg->data.size() == 6)
        for (int i = 0; i < 6; i++) q_des_right_(i) = msg->data[i];
    });

  // Publisher for logging
  data_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    "/dual_arm/controller_data", 10);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
DualArmImpedanceController::on_activate(const rclcpp_lifecycle::State &)
{
  // Seed desired = current position to avoid jumps on startup
  for (int i = 0; i < 6; i++) {
    q_des_left_(i)  = state_interfaces_[i * 2].get_value();
    q_des_right_(i) = state_interfaces_[12 + i * 2].get_value();
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
DualArmImpedanceController::on_deactivate(const rclcpp_lifecycle::State &)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type
DualArmImpedanceController::update(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // --- Read states ---
  for (int i = 0; i < 6; i++) {
    q_left_(i)   = state_interfaces_[i * 2].get_value();
    qd_left_(i)  = state_interfaces_[i * 2 + 1].get_value();
    q_right_(i)  = state_interfaces_[12 + i * 2].get_value();
    qd_right_(i) = state_interfaces_[12 + i * 2 + 1].get_value();
  }

  std::lock_guard<std::mutex> lock(target_mutex_);

  // --- Position errors ---
  Eigen::VectorXd e_left  = q_des_left_  - q_left_;
  Eigen::VectorXd e_right = q_des_right_ - q_right_;

  // --- Velocity errors ---
  Eigen::VectorXd ed_left  = qd_des_left_  - qd_left_;
  Eigen::VectorXd ed_right = qd_des_right_ - qd_right_;

  // --- Coordination error ---
  // Enforces: q_left - q_right == coord_offset_
  Eigen::VectorXd coord_error = (q_left_ - q_right_) - coord_offset_;

  // Coordination correction: push both arms toward constraint
  Eigen::VectorXd f_coord_left  = -k_coord_ * coord_error;
  Eigen::VectorXd f_coord_right =  k_coord_ * coord_error;

  // --- Impedance control law (PD + coordination in joint space) ---
  // Note: for full impedance you'd add inertia shaping; this is compliant PD + coupling
  cmd_left_  = Kp_left_  * e_left  + Kd_left_  * ed_left  + f_coord_left;
  cmd_right_ = Kp_right_ * e_right + Kd_right_ * ed_right + f_coord_right;

  // --- Clamp and write commands ---
  // Since hardware interface is position, we integrate: q_cmd = q + correction * dt
  // (Use effort interface for true impedance; position interface gives compliant PD)
  double dt = 0.01;  // 100Hz
  for (int i = 0; i < 6; i++) {
    double cmd_l = q_left_(i)  + cmd_left_(i)  * dt;
    double cmd_r = q_right_(i) + cmd_right_(i) * dt;

    // Safety clamp
    cmd_l = std::clamp(cmd_l, -6.28, 6.28);
    cmd_r = std::clamp(cmd_r, -6.28, 6.28);

    command_interfaces_[i].set_value(cmd_l);
    command_interfaces_[6 + i].set_value(cmd_r);
  }

  // --- Publish logging data ---
  auto log_msg = std_msgs::msg::Float64MultiArray();
  for (int i = 0; i < 6; i++) log_msg.data.push_back(e_left(i));
  for (int i = 0; i < 6; i++) log_msg.data.push_back(e_right(i));
  for (int i = 0; i < 6; i++) log_msg.data.push_back(coord_error(i));
  data_pub_->publish(log_msg);

  return controller_interface::return_type::OK;
}

}  // namespace dual_arm_controller

PLUGINLIB_EXPORT_CLASS(
  dual_arm_controller::DualArmImpedanceController,
  controller_interface::ControllerInterface)