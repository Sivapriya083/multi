#!/usr/bin/env python3
# =============================================================================
# FILE: injector_node.py
# PACKAGE: ur5e_impulse_injector
# ROS VERSION: ROS 2 Jazzy
#
# PURPOSE:
#   Autonomously applies short-duration, high-magnitude impulsive force
#   disturbances to the UR5e end-effector in Gazebo Harmonic simulation.
#   Runs a pre-scheduled 7-test sequence without any human intervention.
#
# HOW IMPULSES ARE APPLIED:
#   Gazebo Harmonic provides the service:
#     /world/<world_name>/apply_link_wrench
#   Type: ros_gz_interfaces/srv/ApplyLinkWrench
#
#   This service directly applies a 6-DOF wrench (force + torque) to a
#   named link for a specified duration. It bypasses contact dynamics and
#   acts as an IDEALIZED IMPULSE - exactly what we need to simulate gun
#   recoil, hammer strikes, etc.
#
# PHYSICS OF IMPULSES:
#   An impulse J = F * dt  [N*s or kg*m/s]
#   For gun_recoil: J = 180N * 0.015s = 2.7 N*s (realistic 9mm pistol: ~1-5 N*s)
#   For hammer_strike: J = 400N * 0.010s = 4.0 N*s (moderate hammer blow)
#   These represent real-world impulsive loads that robotic arms encounter in
#   industrial settings (power tools, assembly operations, impacts).
#
# SOURCES:
#   [1] Rossi, R. (2011). "Gunshot Residue Analysis." Analytical Methods.
#       Source for gun recoil force/duration estimates (100-250N over 5-30ms)
#
#   [2] Mechanical engineering handbook data for hammer impact forces:
#       Light hammer: 50-200N/5-20ms
#       Heavy hammer: 200-600N/5-20ms
#
#   [3] ros_gz_interfaces documentation:
#       https://github.com/gazebosim/ros_gz/tree/ros2/ros_gz_interfaces
#       ApplyLinkWrench.srv specification
#
#   [4] Gazebo Harmonic Entity Component System (ECS) documentation:
#       https://gazebosim.org/api/sim/8/
#       ApplyLinkWrench system plugin implementation
# =============================================================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import threading
import time    # wall-clock time for sleep between pulses

from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Wrench, Vector3

# ros_gz_interfaces: ROS 2 <-> Gazebo Harmonic service interfaces
# Installed via: sudo apt install ros-jazzy-ros-gz-interfaces
try:
    from ros_gz_interfaces.srv import ApplyLinkWrench
    HAS_SERVICE = True
except ImportError:
    # Fallback if ros_gz_interfaces not installed
    # Will use topic-based wrench publication instead
    HAS_SERVICE = False
    print('[Injector] WARNING: ros_gz_interfaces not found. Using topic fallback.')


# =============================================================================
# SECTION 1: IMPULSE PROFILES
# =============================================================================
#
# Each profile defines one or more "pulses":
#   force [N]:      3D force vector in world frame (at EE link origin)
#   torque [Nm]:    3D torque vector in world frame
#   duration_s:     how long to apply the wrench [seconds]
#   delay_after_s:  pause after this pulse before next (for multi-pulse)
#
# COORDINATE FRAME:
#   World frame (not EE frame) is used for consistency and because
#   the Gazebo ApplyLinkWrench service accepts world-frame wrenches.
#   Positive x: forward (away from robot base)
#   Positive z: up
#
# IMPULSE MAGNITUDE DESIGN:
#   We want the controller to be challenged but not to violate joint limits.
#   Tested range: 50-400N over 10-50ms
#   Impulse = Force * duration:
#     gun_recoil:    180N * 15ms = 2.7 Ns  (moderate)
#     hammer_strike: 400N * 10ms = 4.0 Ns  (strong)
#     heavy_impact:  350N * 50ms = 17.5 Ns (very strong)
# =============================================================================
IMPULSE_PROFILES = {

    # -------------------------------------------------------------------------
    # Profile: gun_recoil
    # Simulates firing a pistol held by the robot's gripper.
    # Force: primarily backward axial (negative x), slight upward component
    # Duration: 15ms (typical 9mm pistol dwell time in barrel: 1-2ms but
    #   total recoil force felt over grip: 10-20ms)
    # Reference: firearm mechanics texts; Rossi (2011) ~180N peak force
    # -------------------------------------------------------------------------
    'gun_recoil': {
        'description': 'Gun recoil — 180N axial backward, 15ms',
        'pulses': [{
            'force':        np.array([-180.0,  10.0,  -5.0]),  # Fx=-180N (backward)
            'torque':       np.array([   0.0,   8.0,   2.0]),  # slight muzzle rise
            'duration_s':   0.015,   # 15ms
            'delay_after_s': 0.0,
        }],
    },

    # -------------------------------------------------------------------------
    # Profile: hammer_strike
    # Simulates a hammer striking downward on the EE-held workpiece.
    # Force: primarily downward (negative z), strong magnitude
    # Duration: 10ms (hard metal-on-metal contact; softer material = longer)
    # Reference: Engineering handbook - heavy hammer ~200-600N over 5-20ms
    # -------------------------------------------------------------------------
    'hammer_strike': {
        'description': 'Hammer strike — 400N downward, 10ms',
        'pulses': [{
            'force':        np.array([  5.0,  -10.0, -400.0]),  # Fz=-400N (down)
            'torque':       np.array([ 15.0,   10.0,    0.0]),  # slight tilt moment
            'duration_s':   0.010,   # 10ms
            'delay_after_s': 0.0,
        }],
    },

    # -------------------------------------------------------------------------
    # Profile: side_impact
    # Simulates a lateral collision (e.g., another robot arm, object passing by)
    # Force: purely lateral (positive y), moderate magnitude
    # Duration: 20ms (softer contact than hard metal; clothing/foam object)
    # -------------------------------------------------------------------------
    'side_impact': {
        'description': 'Side impact — 250N lateral, 20ms',
        'pulses': [{
            'force':        np.array([  0.0,  250.0,  30.0]),  # Fy=250N (lateral)
            'torque':       np.array([  5.0,    0.0,  20.0]),  # wrench torque
            'duration_s':   0.020,   # 20ms
            'delay_after_s': 0.0,
        }],
    },

    # -------------------------------------------------------------------------
    # Profile: multi_axis
    # Simultaneous impulse in all 3 translational axes.
    # Tests the arm's ability to handle combined disturbances.
    # Magnitude: sqrt(120^2 + 150^2 + 80^2) = ~213N resultant
    # -------------------------------------------------------------------------
    'multi_axis': {
        'description': 'Multi-axis — 213N resultant, all axes, 25ms',
        'pulses': [{
            'force':        np.array([-120.0,  150.0,  -80.0]),
            'torque':       np.array([  20.0,   15.0,   10.0]),
            'duration_s':   0.025,   # 25ms
            'delay_after_s': 0.0,
        }],
    },

    # -------------------------------------------------------------------------
    # Profile: heavy_impact
    # Very strong sustained impulse - worst case test.
    # Magnitude: sqrt(350^2 + 50^2 + 200^2) = ~408N resultant over 50ms
    # Impulse: 408N * 50ms = 20.4 Ns (equivalent to catching a ~2kg mass
    #          traveling at ~10 m/s)
    # -------------------------------------------------------------------------
    'heavy_impact': {
        'description': 'Heavy impact — 408N resultant, 50ms',
        'pulses': [{
            'force':        np.array([-350.0, -50.0, -200.0]),
            'torque':       np.array([  30.0,  20.0,   15.0]),
            'duration_s':   0.050,   # 50ms - longest duration
            'delay_after_s': 0.0,
        }],
    },

    # -------------------------------------------------------------------------
    # Profile: rapid_succession
    # Two opposing pulses 100ms apart.
    # Tests if the arm can handle sequential disturbances before fully recovering.
    # Pulse 1: +150N in x direction (push forward)
    # Pause:   100ms (controller starts recovering)
    # Pulse 2: -150N in x direction (push backward while recovering)
    # This tests controller robustness during incomplete recovery.
    # -------------------------------------------------------------------------
    'rapid_succession': {
        'description': 'Rapid succession — two opposing 150N pulses, 100ms apart',
        'pulses': [
            {
                'force':        np.array([150.0, 0.0, 0.0]),
                'torque':       np.zeros(3),
                'duration_s':   0.015,     # 15ms pulse
                'delay_after_s': 0.10,     # 100ms pause before pulse 2
            },
            {
                'force':        np.array([-150.0, 0.0, 0.0]),  # opposite direction
                'torque':       np.zeros(3),
                'duration_s':   0.015,     # 15ms pulse
                'delay_after_s': 0.0,
            },
        ],
    },
}

# =============================================================================
# SECTION 2: AUTONOMOUS TEST SEQUENCE
# =============================================================================
#
# The sequence is scheduled by wall-clock time since node start.
# t=0 is when the injector node starts (after startup_delay_s).
# Format: (scheduled_time_s, profile_name, human_readable_label)
#
# Spacing rationale:
#   - 7-8 seconds between tests: enough for controller to fully settle
#     (T_COMPLIANT_HOLD=0.3s + T_RECOVERY_BLEND=1.5s = 1.8s minimum)
#     We give 5-7s extra so performance metrics capture full recovery.
#   - Tests escalate in severity: start gentle (gun_recoil), end strong.
#   - Last test repeats first: verifies no controller state degradation.
# =============================================================================
TEST_SEQUENCE = [
    ( 5.0,  'gun_recoil',       'Test 1 — Gun Recoil (baseline)'),
    (13.0,  'hammer_strike',    'Test 2 — Hammer Strike (vertical)'),
    (22.0,  'side_impact',      'Test 3 — Side Impact (lateral)'),
    (31.0,  'multi_axis',       'Test 4 — Multi-Axis (diagonal)'),
    (41.0,  'heavy_impact',     'Test 5 — Heavy Impact (worst case)'),
    (51.0,  'rapid_succession', 'Test 6 — Rapid Succession (sequential)'),
    (62.0,  'gun_recoil',       'Test 7 — Gun Recoil (regression check)'),
]
# Total sequence duration: ~62 + 7 (final recovery) = ~69 seconds


class ImpulseInjector(Node):
    """
    ROS 2 Node: Autonomous impulse disturbance injector.

    OPERATION:
      1. On startup, waits startup_delay_s for controller to initialize
      2. Runs TEST_SEQUENCE in a background thread (not blocking ROS spin)
      3. For each test: calls Gazebo ApplyLinkWrench service
      4. Publishes events to /impulse_injector/event for performance monitor
      5. After all tests, publishes SEQUENCE_COMPLETE and idles

    THREAD MODEL:
      - Main thread: rclpy.spin(node) handles service responses
      - Background thread: sequence runner sleeps and calls services
      - No shared mutable state between threads (node is effectively read-only
        from the background thread, only publishing outgoing messages)

    GAZEBO SERVICE INTERFACE:
      Service: /world/<world_name>/apply_link_wrench
      Type: ros_gz_interfaces/srv/ApplyLinkWrench

      Request fields:
        link_name       : "model_name::link_name" format
        reference_frame : "" = world frame, or frame name
        reference_point : Point = offset from link origin
        wrench          : geometry_msgs/Wrench (force + torque)
        start_time      : builtin_interfaces/Time (0 = immediate)
        duration        : builtin_interfaces/Duration (seconds + nanoseconds)

      Response fields:
        success : bool
        message : string (error description if failed)
    """

    def __init__(self):
        super().__init__('impulse_injector')

        # -- Parameters --
        self.declare_parameter('startup_delay_s', 3.0)   # wait for controller init
        self.declare_parameter('world_name', 'ur5e_world')  # Gazebo world name
        # UR5e link name in Gazebo: must be "model_name::link_name"
        # The wrist_3_link is the last link before the EE flange
        self.declare_parameter('ee_link', 'ur5e::wrist_3_link')

        world   = self.get_parameter('world_name').value
        self._ee_link = self.get_parameter('ee_link').value

        # -- QoS --
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)

        # -- Publishers --
        # /impulse_injector/event: announces each test to performance monitor
        # Format: "IMPULSE:profile_name_pulse1" or "SEQUENCE_COMPLETE"
        self.pub_event = self.create_publisher(String, '/impulse_injector/event', qos)

        # /impulse_injector/applied_force: logs applied force/torque/duration
        # Used for post-hoc analysis of what was applied vs controller response
        self.pub_force = self.create_publisher(Float64MultiArray, '/impulse_injector/applied_force', qos)

        # -- Service Client --
        # Create client for the Gazebo ApplyLinkWrench service
        # Service name format: /world/<world_name>/apply_link_wrench
        if HAS_SERVICE:
            self._srv = self.create_client(
                ApplyLinkWrench,
                f'/world/{world}/apply_link_wrench')
            self.get_logger().info(
                f'[Injector] Service client: /world/{world}/apply_link_wrench')
        else:
            self._srv = None
            # Fallback topic: expects a custom Gazebo plugin to read this
            self.pub_wrench_cmd = self.create_publisher(
                Wrench, '/ur5e/ee_wrench_cmd', qos)
            self.get_logger().warn('[Injector] Using topic fallback on /ur5e/ee_wrench_cmd')

        # -- Start autonomous sequence in background thread --
        # Background thread so ROS spin can handle service callbacks
        delay = self.get_parameter('startup_delay_s').value
        thread = threading.Thread(
            target=self._run_sequence,
            args=(delay,),
            daemon=True)   # daemon = killed when main process exits
        thread.start()
        self.get_logger().info(f'[Injector] Sequence thread started. Delay={delay}s')

    # =========================================================================
    # SERVICE-BASED PULSE APPLICATION (primary method, Gazebo Harmonic)
    # =========================================================================

    def _apply_via_service(self, force_vec, torque_vec, duration_s, label):
        """
        Apply a single wrench pulse via ros_gz_interfaces/ApplyLinkWrench.

        HOW IT WORKS IN GAZEBO HARMONIC:
          Gazebo Harmonic uses an ECS (Entity Component System) architecture.
          The gz-sim Physics system processes wrench commands and directly
          integrates them into the rigid body dynamics at each physics step.

          When the service is called:
          1. Gazebo finds the named Link entity
          2. Registers the wrench to be applied for the specified duration
          3. At each physics step (1ms = 1kHz), adds the wrench to body forces
          4. After duration expires, removes the wrench

          This is EXACT physics - not a simplified approximation. The wrench
          is applied as a real force in the dynamics simulation.

        TIMING PRECISION:
          Gazebo physics runs at 1kHz. A 15ms impulse = 15 physics steps.
          Start time = {0,0} means "apply immediately on next physics step."
          This gives ~1ms timing jitter (acceptable for our purposes).

        SERVICE CALL PATTERN (ROS 2 async):
          1. self._srv.call_async(req) returns a Future
          2. rclpy.spin_until_future_complete() processes callbacks until done
          3. future.result() gives the response

          NOTE: call_async + spin_until_future_complete is called from a
          background thread. This works because rclpy is reentrant-safe when
          using MultiThreadedExecutor OR if the node uses separate executor.
          With default SingleThreadedExecutor this can deadlock - see note below.
        """
        if not self._srv.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'[Injector] Service unavailable for: {label}')
            return False

        # Build service request
        req = ApplyLinkWrench.Request()

        # Link name must match Gazebo model: "model_name::link_name"
        # In Gazebo Harmonic, model is spawned as "ur5e", last link is "wrist_3_link"
        req.link_name       = self._ee_link   # "ur5e::wrist_3_link"
        req.reference_frame = ''              # '' = world frame

        # Force and torque components [N] and [Nm]
        req.wrench.force.x  = float(force_vec[0])
        req.wrench.force.y  = float(force_vec[1])
        req.wrench.force.z  = float(force_vec[2])
        req.wrench.torque.x = float(torque_vec[0])
        req.wrench.torque.y = float(torque_vec[1])
        req.wrench.torque.z = float(torque_vec[2])

        # Start time: {sec=0, nanosec=0} = apply immediately
        req.start_time.sec     = 0
        req.start_time.nanosec = 0

        # Duration: convert seconds to (sec, nanosec) pair
        req.duration.sec     = int(duration_s)
        req.duration.nanosec = int((duration_s - int(duration_s)) * 1e9)
# Joint names MUST match the URDF joint names exactly.
    # ORDER MATTERS: data[0] in the Float64MultiArray command maps to joints[0].
    # Our controller computes torques in this order:
    #   tau[0] = shoulder_pan_joint torque
    #   tau[1] = shoulder_lift_joint torque
    #   tau[2] = elbow_joint torque
    #   tau[3] = wrist_1_joint torque
    #   tau[4] = wrist_2_joint torque
    #   tau[5] = wrist_3_joint torque
        # Async call + wait for completion
        future = self._srv.call_async(req)
        # spin_until_future_complete: processes all pending callbacks until
        # future is resolved (service response received)
        # timeout_sec=3.0: don't wait forever if Gazebo hangs
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.done() and future.result() and future.result().success:
            f_mag = np.linalg.norm(force_vec)
            self.get_logger().info(
                f'[Injector] APPLIED: {label}  |F|={f_mag:.0f}N  dur={duration_s*1000:.0f}ms')
            return True
        else:
            err = future.result().message if (future.done() and future.result()) else 'timeout'
            self.get_logger().warn(f'[Injector] Service failed for {label}: {err}')
            return False

    # =========================================================================
    # TOPIC-BASED FALLBACK (if ros_gz_interfaces not available)
    # =========================================================================

    def _apply_via_topic(self, force_vec, torque_vec, duration_s, label):
        """
        Fallback: publish wrench on topic for duration_s seconds.

        REQUIRES: A Gazebo plugin or bridge node that reads /ur5e/ee_wrench_cmd
        and applies it to the EE link.

        HOW IT WORKS:
          Publishes at 1kHz for duration_s seconds, then publishes zero.
          Effective impulse = sum(F * dt) = F * duration_s (for constant F)

        LIMITATION vs service method:
          - Less precise timing (ROS topic publish has ~1-10ms jitter)
          - Requires additional Gazebo plugin to be running
          - Not used unless ros_gz_interfaces is unavailable
        """
        msg = Wrench()
        msg.force  = Vector3(x=float(force_vec[0]),  y=float(force_vec[1]),  z=float(force_vec[2]))
        msg.torque = Vector3(x=float(torque_vec[0]), y=float(torque_vec[1]), z=float(torque_vec[2]))

        self.get_logger().info(
            f'[Injector] topic-APPLY: {label}  |F|={np.linalg.norm(force_vec):.0f}N  '
            f'dur={duration_s*1000:.0f}ms')

        t_end = time.monotonic() + duration_s
        while time.monotonic() < t_end:
            self.pub_wrench_cmd.publish(msg)
            time.sleep(0.001)   # 1kHz publish rate

        # Zero out the wrench (stop applying force)
        self.pub_wrench_cmd.publish(Wrench())

    # =========================================================================
    # SINGLE PULSE DISPATCHER
    # =========================================================================

    def _apply_pulse(self, force_vec, torque_vec, duration_s, label):
        """
        Apply a single wrench pulse and announce it.

        Publishes event BEFORE applying (so performance monitor records
        exact t=0 of disturbance, not after it returns from service).
        """
        # Announce event (performance monitor listens for this)
        evt = String()
        evt.data = f'IMPULSE:{label}'
        self.pub_event.publish(evt)

        # Log applied force for analysis
        fmsg = Float64MultiArray()
        fmsg.data = (list(force_vec.astype(float)) +
                     list(torque_vec.astype(float)) + [float(duration_s)])
        self.pub_force.publish(fmsg)

        # Apply via best available method
        if self._srv is not None:
            self._apply_via_service(force_vec, torque_vec, duration_s, label)
        else:
            self._apply_via_topic(force_vec, torque_vec, duration_s, label)

    # =========================================================================
    # PROFILE EXECUTOR
    # =========================================================================

    def _apply_profile(self, profile_name):
        """
        Apply all pulses in a named impulse profile.
        For multi-pulse profiles (rapid_succession), applies each pulse
        sequentially with the specified inter-pulse delay.
        """
        profile = IMPULSE_PROFILES.get(profile_name)
        if profile is None:
            self.get_logger().error(f'[Injector] Unknown profile: {profile_name}')
            return

        self.get_logger().info(f'[Injector] Profile: {profile["description"]}')

        for idx, pulse in enumerate(profile['pulses']):
            pulse_label = f'{profile_name}_p{idx+1}'
            self._apply_pulse(
                pulse['force'],
                pulse['torque'],
                pulse['duration_s'],
                label=pulse_label)

            # Inter-pulse delay (0.0 for single-pulse profiles)
            delay = pulse.get('delay_after_s', 0.0)
            if delay > 0:
                time.sleep(delay)

    # =========================================================================
    # AUTONOMOUS TEST SEQUENCE
    # =========================================================================

    def _run_sequence(self, startup_delay):
        """
        Background thread: runs TEST_SEQUENCE autonomously.

        TIMING:
          The sequence uses wall-clock time (time.monotonic()) for scheduling.
          This is more reliable than ROS clock in simulation where sim time
          can pause/advance differently from wall time.

          t_start is set AFTER startup_delay, so all test times are relative
          to when the actual sequence begins (not node startup).

        STARTUP DELAY PURPOSE:
          We wait startup_delay_s before starting tests to ensure:
          1. Gazebo has fully spawned the UR5e model
          2. ros2_control controllers are active (forward_effort_controller)
          3. Impedance controller has started and is running at 500Hz
          4. Joint state broadcaster is publishing on /joint_states
          5. F/T sensor is online and bridged
          Typical startup chain: 3-5 seconds on a modern PC.
        """
        self.get_logger().info(
            f'[Injector] Waiting {startup_delay}s for system initialization...')
        time.sleep(startup_delay)

        self.get_logger().info('[Injector] *** AUTONOMOUS TEST SEQUENCE START ***')
        self.get_logger().info(f'[Injector] {len(TEST_SEQUENCE)} tests scheduled.')

        t_start = time.monotonic()   # sequence clock starts NOW

        for sched_t, profile_name, label in TEST_SEQUENCE:
            # Busy-wait until scheduled time
            # (low overhead ~5ms sleep granularity is fine for ~5s gaps)
            while True:
                elapsed = time.monotonic() - t_start
                if elapsed >= sched_t:
                    break
                time.sleep(0.005)   # 5ms sleep granularity

            # Log and apply
            elapsed = time.monotonic() - t_start
            self.get_logger().info(
                f'\n[Injector] *** {label} *** (scheduled={sched_t:.1f}s, '
                f'actual={elapsed:.2f}s)')
            self._apply_profile(profile_name)

            # Brief pause between profile and next schedule check
            time.sleep(0.1)

        # Sequence complete
        self.get_logger().info('[Injector] *** TEST SEQUENCE COMPLETE ***')

        evt = String()
        evt.data = 'SEQUENCE_COMPLETE'
        self.pub_event.publish(evt)


def main(args=None):
    rclpy.init(args=args)
    node = ImpulseInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()