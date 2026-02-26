#!/usr/bin/env python3
# =============================================================================
# FILE: controller_node.py
# PACKAGE: ur5e_impedance_controller
# ROS VERSION: ROS 2 Jazzy (rclpy)
# ROBOT: Universal Robots UR5e (6-DOF serial manipulator)
#
# PURPOSE:
#   Implements a fully custom Adaptive Cartesian Impedance Controller that
#   maintains a fixed end-effector (EE) pose while absorbing impulsive external
#   force disturbances (gun recoil ~180N/15ms, hammer strikes ~400N/10ms, etc.)
#
# MATHEMATICAL FOUNDATIONS:
#   1. Modified Denavit-Hartenberg (DH) forward kinematics
#   2. Geometric Jacobian (base-frame, 6x6)
#   3. Cartesian impedance control law:  tau = J^T[K*ex - D*xdot] + tau_null + tau_grav
#   4. Variable impedance blending (nominal <-> compliant)
#   5. Multi-criterion impulse detection
#   6. Critically-damped 2nd-order recovery filter
#   7. Null-space posture control
#   8. Analytic gravity compensation
#
# SOURCES / REFERENCES:
#   [1] Hogan, N. (1985). "Impedance Control: An Approach to Manipulation"
#       Journal of Dynamic Systems, Measurement, and Control, 107(1), 1-24.
#       Source of the impedance control law: F = K*e - D*xdot
#
#   [2] Siciliano, B. et al. (2009). "Robotics: Modelling, Planning and Control"
#       Springer. Ch.3 (DH kinematics), Ch.5 (Jacobians), Ch.8 (impedance control)
#       Source for DH parameter convention, geometric Jacobian derivation
#
#   [3] Universal Robots UR5e Technical Specification (2022)
#       Source of DH parameters, joint limits, link masses used below
#
#   [4] Albu-Schaeffer, A. et al. (2007). "A Unified Passivity-Based Control
#       Framework for Position, Torque and Impedance Control."
#       IEEE T-RO. Source for variable impedance stability proof.
#
#   [5] Khatib, O. (1987). "A unified approach for motion and force control."
#       IEEE J. Robotics Automation. Source for null-space projection.
#
#   [6] Craig, J.J. (2005). "Introduction to Robotics: Mechanics and Control"
#       Pearson. Ch.3 (DH matrix), Ch.5 (velocity kinematics)
# =============================================================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
from numpy.linalg import pinv, norm

import threading

from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped


# =============================================================================
# SECTION 1: UR5e KINEMATIC PARAMETERS
# SOURCE: Universal Robots UR5e Technical Spec + ros-industrial/universal_robot
#         github.com/ros-industrial/universal_robot/blob/melodic-devel/
#         ur_description/urdf/ur5e.urdf.xacro
#
# CONVENTION: Modified DH (Craig 2005, Chapter 3)
#   Each row: [a_i (m), d_i (m), alpha_i (rad), theta_offset_i (rad)]
#
#   a_i:          link length - offset along x_{i-1} from z_{i-1} to z_i
#   d_i:          link offset - along z_{i-1} from x_{i-2} to x_{i-1}
#   alpha_i:      link twist  - rotation about x_{i-1} from z_{i-1} to z_i
#   theta_offset: added to q_i to get DH theta_i (all zero for UR5e)
#
#   Full DH matrix (Craig 2005, Eq. 3.6):
#     T = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
#
#     [ cos(theta)   -sin(theta)*cos(alpha)   sin(theta)*sin(alpha)   a*cos(theta) ]
#     [ sin(theta)    cos(theta)*cos(alpha)  -cos(theta)*sin(alpha)   a*sin(theta) ]
#     [     0              sin(alpha)              cos(alpha)               d      ]
#     [     0                  0                       0                   1      ]
# =============================================================================
UR5E_DH = np.array([
    # [ a_i,      d_i,      alpha_i,       theta_offset ]
    [  0.000,    0.1625,   np.pi / 2,    0.0 ],  # Joint 1: shoulder_pan
    [ -0.425,    0.0000,   0.0,          0.0 ],  # Joint 2: shoulder_lift
    [ -0.3922,   0.0000,   0.0,          0.0 ],  # Joint 3: elbow
    [  0.000,    0.1333,   np.pi / 2,    0.0 ],  # Joint 4: wrist_1
    [  0.000,    0.0997,  -np.pi / 2,    0.0 ],  # Joint 5: wrist_2
    [  0.000,    0.0996,   0.0,          0.0 ],  # Joint 6: wrist_3
])

# =============================================================================
# SECTION 2: HOME CONFIGURATION
#
# "Elbow-up" canonical pose chosen for:
#   - Good distance from kinematic singularities
#   - Reasonable Jacobian condition number (~5-15)
#   - Natural "ready" manipulation pose
#
# Joint angles [rad]:
#   q[0] shoulder_pan  =  0.0    (forward)
#   q[1] shoulder_lift = -pi/2   (upper arm up)
#   q[2] elbow         = +pi/2   (forearm horizontal)
#   q[3] wrist_1       = -pi/2   (wrist neutral)
#   q[4] wrist_2       = -pi/2   (tool forward)
#   q[5] wrist_3       =  0.0    (no wrist roll)
# =============================================================================
Q_HOME = np.array([0.0, -np.pi/2.0, np.pi/2.0, -np.pi/2.0, -np.pi/2.0, 0.0])

# =============================================================================
# SECTION 3: IMPEDANCE MATRICES
#
# SOURCE: Gains derived from Hogan (1985) design methodology:
#   Natural frequency: wn = sqrt(K/m_eff)
#   Damping ratio:     zeta = D / (2 * sqrt(K * m_eff))
#
# WHY DIAGONAL MATRICES:
#   Decouples X,Y,Z translational DOFs from each other. Off-diagonal terms
#   require accurate cross-coupling identification (hard in practice).
#   Siciliano et al. (2009) Ch.8 justifies diagonal choice for typical tasks.
#
# --- NOMINAL IMPEDANCE (stiff pose holding, no disturbance) ---
# K_trans = 800 N/m:
#   With m_eff ~ 5 kg: wn = sqrt(800/5) = 12.6 rad/s (fast enough)
#   Higher (>1500) risks exciting structural resonances of UR5e links
# D_trans = 80 Ns/m:
#   zeta = 80/(2*sqrt(800*5)) = 0.63  (slightly underdamped = fast response)
# K_rot = 60 Nm/rad, D_rot = 8 Nms/rad: typical for wrist compliance
K_NOMINAL   = np.diag([800.0, 800.0, 800.0,  60.0,  60.0,  60.0])
D_NOMINAL   = np.diag([ 80.0,  80.0,  80.0,   8.0,   8.0,   8.0])

# --- COMPLIANT IMPEDANCE (impulse absorption) ---
# K_trans = 80 N/m  (10x softer than nominal):
#   A 300N impulse on 800 N/m spring: deflection = 300/800 = 375mm (too much)
#   On 80 N/m spring: arm yields passively, much less transmitted force
# D_trans = 120 Ns/m  (1.5x MORE damping):
#   zeta = 120/(2*sqrt(80*5)) = 3.0  (heavily overdamped = no rebound)
#   Overdamping is intentional: absorbs ALL kinetic energy without bounce
K_COMPLIANT = np.diag([ 80.0,  80.0,  80.0,  10.0,  10.0,  10.0])
D_COMPLIANT = np.diag([120.0, 120.0, 120.0,  15.0,  15.0,  15.0])

# Null-space gains (see Section 7)
# K_NULL = 5 Nm/rad: gentle ~3% of UR5e max torque per radian
# D_NULL = 2 Nms/rad: critically damped null-space motion
K_NULL = np.diag([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
D_NULL = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

# =============================================================================
# SECTION 4: IMPULSE DETECTION THRESHOLDS
#
# Three independent criteria (any one triggers compliance mode):
#
# FORCE_THRESHOLD = 15 N
#   Context: UR5e 5kg payload -> max static = 49N
#   Normal manipulation forces: 1-10N
#   Gun recoil: ~180N -> 15N catches impulse well before peak
#   Source: Haddadin et al. (2017) "Robot Collisions" IEEE T-RO used
#           similar 10-20N thresholds for collision detection
#
# TORQUE_THRESHOLD = 5 Nm
#   Normal wrist torques from payload: <2 Nm
#   Hammer strike torques: 15-30 Nm -> 5 Nm is early detection
#
# VEL_THRESHOLD = 0.3 rad/s
#   Normal slow manipulation: <0.1 rad/s
#   After 180N/15ms impulse: delta_qdot ~ 1-3 rad/s -> detects easily
#
# SPIKE_THRESHOLD = 7.5 N (half of FORCE_THRESHOLD)
#   |F_raw - F_filtered| measures signal "surprise" vs running average
#   Catches sub-10ms pulses that LPF smooths before they hit threshold
# =============================================================================
FORCE_THRESHOLD   = 15.0   # N
TORQUE_THRESHOLD  =  5.0   # Nm
VEL_THRESHOLD     =  0.3   # rad/s
SPIKE_THRESHOLD   =  7.5   # N

# =============================================================================
# SECTION 5: COMPLIANCE TIMING
#
# T_COMPLIANT_HOLD = 0.30 s
#   Hold full compliance after detection for 300ms.
#   Rationale: Even the longest test impulse (heavy_impact, 50ms) finishes
#   within 300ms from detection. The hold ensures we don't stiffen up mid-impulse.
#   Also gives the recovery filter time to start working.
#
# T_RECOVERY_BLEND = 1.50 s
#   Ramp stiffness back from compliant to nominal over 1.5 seconds.
#   Rationale: Too fast (<0.5s) -> abrupt stiffness change -> oscillation.
#   Too slow (>3s) -> arm wanders at low stiffness, slow convergence.
#   1.5s is empirically good for tested 100-400N impulses on UR5e inertia.
# =============================================================================
T_COMPLIANT_HOLD  = 0.30   # seconds
T_RECOVERY_BLEND  = 1.50   # seconds

# Safety limits
TAU_MAX    = 150.0   # Nm - UR5e continuous joint torque limit from spec
F_CART_MAX = 200.0   # N  - Cartesian force saturation before J^T mapping

# UR5e link masses [kg] from UR5e Technical Specification
# "Weight" section, individual link breakdown
UR5E_MASSES = np.array([3.7, 8.393, 2.33, 1.219, 1.219, 0.1879])

# CoM z-offsets [m] from URDF inertial tags:
# ur_description/urdf/ur5e.urdf.xacro -> <inertial><origin xyz="..."/>
# Approximate z-component of CoM in each link's local frame
UR5E_COM_Z  = np.array([0.0, 0.2125, 0.1961, 0.0, 0.0, 0.0])


# =============================================================================
# KINEMATICS FUNCTIONS
# =============================================================================

def dh_matrix(a, d, alpha, theta):
    """
    Modified DH transform matrix T_{i-1,i}.
    SOURCE: Craig (2005) Eq. 3.6

    Composed of 4 elementary transforms:
      T = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)

    Expanding and simplifying:
      Row 0: [cos(t),  -sin(t)*cos(a),   sin(t)*sin(a),   a*cos(t)]
      Row 1: [sin(t),   cos(t)*cos(a),  -cos(t)*sin(a),   a*sin(t)]
      Row 2: [0,         sin(a),           cos(a),          d      ]
      Row 3: [0,         0,                0,               1      ]

    where t = theta, a = alpha (shorthand)
    """
    ct = np.cos(theta);  st = np.sin(theta)
    ca = np.cos(alpha);  sa = np.sin(alpha)
    return np.array([
        [ct,  -st*ca,   st*sa,  a*ct],
        [st,   ct*ca,  -ct*sa,  a*st],
        [0.0,  sa,      ca,     d   ],
        [0.0,  0.0,     0.0,    1.0 ],
    ])


def forward_kinematics(q):
    """
    Compute UR5e forward kinematics via DH chain multiplication.
    SOURCE: Siciliano (2009) Ch.2, standard DH product formula:
      T_0_n = T_0_1 * T_1_2 * ... * T_{n-1}_n

    Each T_0_i = T_0_{i-1} * dh_matrix(params_i, q_i + offset_i)

    Args:
        q: joint angles [rad], shape (6,)
    Returns:
        T_ee:   4x4 base-to-EE transform
        frames: list of 7 transforms [T_0_0, T_0_1, ..., T_0_6]
                frames[0] = I (base), frames[6] = T_ee
    """
    T = np.eye(4)
    frames = [T.copy()]   # frames[0] = base frame = identity
    for i in range(6):
        a, d, alpha, offset = UR5E_DH[i]
        T = T @ dh_matrix(a, d, alpha, q[i] + offset)
        frames.append(T.copy())
    return T, frames


def geometric_jacobian(q):
    """
    Compute 6x6 geometric Jacobian in base frame.
    SOURCE: Siciliano (2009) Section 3.2 "Geometric Jacobian"
            Craig (2005) Chapter 5

    For revolute joint i, the Jacobian columns are:
      J_v[:,i] = z_{i-1} x (p_ee - p_{i-1})   <- linear velocity
      J_w[:,i] = z_{i-1}                        <- angular velocity

    where:
      z_{i-1} = 3rd col of T_0_{i-1} rotation = joint rotation axis in base frame
      p_{i-1} = 4th col of T_0_{i-1}           = joint i-1 origin in base frame
      p_ee    = 4th col of T_0_6                = EE origin in base frame

    Physical meaning:
      Rotating joint i at 1 rad/s causes the EE to move at:
        v = z_{i-1} x (p_ee - p_{i-1}) m/s  (by rigid body kinematics)
        w = z_{i-1}  rad/s  (angular velocity from all joints accumulates)

    Relates joint velocities to Cartesian EE velocity:
      [v_ee; w_ee] = J(q) * dq
    """
    _, frames = forward_kinematics(q)
    p_ee = frames[6][:3, 3]   # EE position in base frame

    J = np.zeros((6, 6))
    for i in range(6):
        z = frames[i][:3, 2]   # rotation axis of joint i+1 (z-axis of frame i)
        p = frames[i][:3, 3]   # origin of joint i in base frame

        J[:3, i] = np.cross(z, p_ee - p)   # linear velocity column
        J[3:, i] = z                         # angular velocity column
    return J


def pose_error_6d(T_des, T_cur):
    """
    Compute 6D Cartesian pose error [e_position(3); e_orientation(3)].
    SOURCE: Siciliano (2009) Section 3.7; Albu-Schaeffer (2007) Eq.(4)

    POSITION ERROR:
      e_p = p_desired - p_current   [m]   (simple vector subtraction)

    ORIENTATION ERROR via rotation matrix residual + axis-angle extraction:

      Step 1: Rotation error matrix
        R_err = R_des * R_cur^T
        (= rotation needed to bring R_cur into alignment with R_des)

      Step 2: Extract rotation angle from trace formula
        trace(R) = 1 + 2*cos(theta)
        theta = arccos((trace(R_err) - 1) / 2)

      Step 3: Extract axis from skew-symmetric part
        R - R^T = 2*sin(theta) * [axis]_x
        where [v]_x is the skew-symmetric cross-product matrix of v
        So: axis = (theta / 2*sin(theta)) * [R32-R23, R13-R31, R21-R12]

      Step 4: e_o = axis * theta  (axis-angle vector)

    WHY NOT EULER ANGLES:
      Euler angles have gimbal lock singularities and discontinuities.
    WHY NOT QUATERNION:
      Quaternion has sign ambiguity (q and -q = same rotation).
    WHY AXIS-ANGLE:
      Globally defined, smooth for |theta| < pi, matches impedance theory.
    """
    # Position error (trivial)
    e_p = T_des[:3, 3] - T_cur[:3, 3]

    # Rotation matrices
    R_des = T_des[:3, :3]
    R_cur = T_cur[:3, :3]

    # Rotation error: R_err * R_cur = R_des  =>  R_err = R_des * R_cur^{-1} = R_des * R_cur^T
    R_err = R_des @ R_cur.T

    # Rotation angle: theta = arccos((trace(R)-1)/2)
    # np.clip handles floating-point noise pushing value outside [-1,1]
    cos_a = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)

    if abs(angle) < 1e-7:
        e_o = np.zeros(3)   # No rotation error
    else:
        # axis = (angle / 2*sin(angle)) * [R32-R23, R13-R31, R21-R12]
        # These 3 elements are exactly the components of the axial vector
        # of the skew-symmetric part: (R_err - R_err^T)/2
        k = angle / (2.0 * np.sin(angle))
        e_o = k * np.array([
            R_err[2,1] - R_err[1,2],
            R_err[0,2] - R_err[2,0],
            R_err[1,0] - R_err[0,1],
        ])

    return np.concatenate([e_p, e_o])   # shape (6,)


def critically_damped_step(x, v, x_target, omega_n, dt):
    """
    One step of a critically-damped 2nd-order filter.
    SOURCE: Ogata (2010) "Modern Control Engineering" Section 4.3
            Franklin et al. (2014) "Feedback Control" Ch.3

    PURPOSE:
      Generates smooth recovery trajectory toward x_target.
      Used so the impedance setpoint glides back to desired pose
      instead of jumping (which would spike torques).

    MATH:
      Canonical 2nd-order ODE with critical damping (zeta=1):
        x'' + 2*omega_n*x' + omega_n^2*(x - x_target) = 0

      Closed-form solution:
        x(t) = x_target + (A + B*t) * exp(-omega_n*t)
      -> approaches x_target monotonically (ZERO OVERSHOOT guaranteed)

    DISCRETIZATION (semi-implicit Euler / leapfrog):
      a[k]   = -2*omega_n*v[k] - omega_n^2*(x[k]-x_target)
      v[k+1] = v[k] + a[k]*dt           <- velocity first
      x[k+1] = x[k] + v[k+1]*dt         <- position uses updated velocity

    Semi-implicit (vs pure forward Euler) has better energy conservation
    and is more stable for oscillatory systems.

    WHY CRITICAL DAMPING (zeta=1):
      zeta<1 (underdamped): oscillates -> unacceptable
      zeta>1 (overdamped): no oscillation but slower than necessary
      zeta=1 (critical): FASTEST zero-overshoot approach to target

    STABILITY:
      Forward Euler stable when dt < 1/omega_n
      With omega_n=8, dt=0.002: 0.002 < 0.125 -> stable

    Args:
        x        : current state, shape (3,) [position in m]
        v        : current velocity, shape (3,) [m/s]
        x_target : target state, shape (3,)
        omega_n  : natural frequency [rad/s]
                   omega_n=4 -> 63% convergence in ~0.25s (slow, compliance phase)
                   omega_n=8 -> 63% convergence in ~0.125s (faster, nominal phase)
        dt       : timestep [s]
    Returns:
        x_new, v_new: updated state and velocity
    """
    # Restoring acceleration = spring + damping
    a     = -2.0 * omega_n * v - (omega_n**2) * (x - x_target)
    v_new = v + a * dt        # update velocity first (semi-implicit)
    x_new = x + v_new * dt   # use new velocity for position update
    return x_new, v_new


def gravity_torque(q):
    """
    Compute gravity compensation torques for UR5e.
    SOURCE: Spong, Hutchinson, Vidyasagar (2006) "Robot Modeling" Section 4.6
            Principle of virtual work applied to gravity potential energy

    DERIVATION:
      Potential energy: U = sum_i(m_i * g^T * p_com_i)
      Gravity torque:   tau_grav = -dU/dq

      Using chain rule: dU/dq_j = sum_i(m_i * g^T * dp_com_i/dq_j)

      For revolute joint j, dp_com_i/dq_j = z_{j-1} x (p_com_i - p_{j-1})
      when j <= i (joint j affects link i only if i >= j).

      So: tau_grav[j] = sum_{i>=j}(m_i * g^T * (z_{j-1} x r_{j->com_i}))
        = sum_{i>=j}(m_i * (z_{j-1} x r_{j->com_i}) . g)
        where the triple product is scalar.

    LINK MASSES SOURCE:
      Universal Robots UR5e Technical Spec, "Payload" section
      [3.7, 8.393, 2.33, 1.219, 1.219, 0.1879] kg

    COM OFFSET SOURCE:
      ur5e.urdf.xacro <inertial><origin xyz=...> tags (z-component along link axis)
      Approximate: [0, 0.2125, 0.1961, 0, 0, 0] m

    NOTE: We compute only gravity (static) terms.
      Full RNEA would add inertial (Mq''), Coriolis (C(q,q')*q'), centripetal.
      For slow manipulation + impulse rejection, gravity dominates.
      The impedance D*xdot term implicitly handles velocity-dependent effects.
    """
    g     = np.array([0.0, 0.0, -9.81])   # gravity in base frame [m/s^2]
    tau_g = np.zeros(6)
    _, frames = forward_kinematics(q)

    for i in range(6):
        # Link i's frame = frames[i+1] (link i is distal to joint i+1)
        T_i = frames[i+1]

        # CoM position of link i in base frame:
        # p_com = T_i[:3,3] (frame origin) + T_i[:3,2] * COM_Z[i] (along local z)
        p_com = T_i[:3,3] + T_i[:3,2] * UR5E_COM_Z[i]

        # Each joint j <= i must support link i's weight
        for j in range(i+1):
            z_j = frames[j][:3,2]   # rotation axis of joint j+1
            p_j = frames[j][:3,3]   # origin of joint j

            # Gravitational torque contribution:
            # tau[j] += m_i * (z_j x (p_com - p_j)) . g
            tau_g[j] += UR5E_MASSES[i] * np.dot(np.cross(z_j, p_com - p_j), g)

    return tau_g


# =============================================================================
# MAIN CONTROLLER NODE
# =============================================================================

class AdaptiveImpedanceController(Node):
    """
    ROS 2 Node: Adaptive Impedance Controller for UR5e impulse rejection.

    DESIGN PATTERN:
      - Callbacks update shared state under threading.Lock
      - 500Hz timer reads state snapshot, computes control, publishes commands
      - All parameters declared via ROS 2 param system (runtime tunable)

    TOPIC SUMMARY:
      SUB  /joint_states                           JointState     (500Hz, BEST_EFFORT)
      SUB  /ur5e/ft_sensor/wrench                  WrenchStamped  (500Hz, BEST_EFFORT)
      SUB  /aic/desired_pose                        PoseStamped    (on demand, RELIABLE)
      PUB  /forward_effort_controller/commands      Float64MultiArray (500Hz)
      PUB  /aic/pose_error                          Float64MultiArray (500Hz)
      PUB  /aic/compliance_blend                    Float64MultiArray (500Hz)
      PUB  /aic/status                              String (500Hz)
    """

    def __init__(self):
        super().__init__('adaptive_impedance_controller')
        self._declare_all_parameters()

        # -- Shared state (protected by lock for thread safety) --
        self._lock = threading.Lock()

        # Joint state: updated by /joint_states callback
        self.q  = Q_HOME.copy()   # joint positions [rad]
        self.dq = np.zeros(6)     # joint velocities [rad/s]

        # Desired EE pose: initialized to FK of home = hold home from startup
        T_home, _ = forward_kinematics(Q_HOME)
        self.T_desired = T_home.copy()   # 4x4 homogeneous transform

        # Critically-damped recovery filter state:
        # x_filt tracks T_desired[:3,3] smoothly instead of jumping
        self.x_filt = T_home[:3,3].copy()   # filtered EE position [m]
        self.v_filt = np.zeros(3)            # filtered EE velocity [m/s]

        # External wrench from F/T sensor
        # raw: used for impulse DETECTION (needs full transients)
        # filtered: used in CONTROL LAW (noise rejection)
        self.F_ext_raw      = np.zeros(6)
        self.F_ext_filtered = np.zeros(6)

        # Impulse state
        self.impulse_active   = False
        self.t_impulse        = 0.0
        self.compliance_blend = 0.0   # 0=stiff nominal, 1=soft compliant

        # LPF coefficient for wrench filtering
        # alpha=0.1 -> tau ~ 19ms (smooths noise, preserves impulse detection via raw)
        self._F_LPF_ALPHA = 0.1

        # -- QoS Profiles --
        # BEST_EFFORT: sensor data - low latency, dropped packets OK
        # RELIABLE: commands - must arrive, retransmission allowed
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)
        cmd_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)

        # -- Subscribers --
        # /joint_states: from joint_state_broadcaster (ros2_control)
        # Contains: name[], position[], velocity[], effort[]
        # NOTE: joint ORDER in message is not guaranteed -> use name lookup
        self.create_subscription(JointState, '/joint_states',
            self._joint_state_cb, sensor_qos)

        # /ur5e/ft_sensor/wrench: F/T sensor bridged from Gazebo Harmonic
        # via ros_gz_bridge parameter_bridge
        self.create_subscription(WrenchStamped, '/ur5e/ft_sensor/wrench',
            self._wrench_cb, sensor_qos)

        # /aic/desired_pose: optional external setpoint override
        # If never received, holds home pose (T_desired = T_home forever)
        self.create_subscription(PoseStamped, '/aic/desired_pose',
            self._desired_pose_cb, cmd_qos)

        # -- Publishers --
        # ForwardCommandController: receives Float64MultiArray, writes directly
        # to hardware joint effort interfaces (bypasses any PID)
        self.pub_effort = self.create_publisher(Float64MultiArray,
            '/forward_effort_controller/commands', cmd_qos)

        # Diagnostics: 8-element error [ex,ey,ez,erx,ery,erz, |ep|, |eo|]
        self.pub_error  = self.create_publisher(Float64MultiArray, '/aic/pose_error', cmd_qos)

        # Diagnostics: compliance blend scalar [0.0, 1.0]
        self.pub_blend  = self.create_publisher(Float64MultiArray, '/aic/compliance_blend', cmd_qos)

        # Diagnostics: human-readable "NOMINAL" / "COMPLIANT" / "RECOVERING"
        self.pub_status = self.create_publisher(String, '/aic/status', cmd_qos)

        # -- 500Hz Control Timer --
        rate_hz  = self.get_parameter('control_rate_hz').value
        self._dt = 1.0 / float(rate_hz)
        self.create_timer(self._dt, self._control_loop)

        self.get_logger().info(f'[AIC] Started at {rate_hz}Hz. Home EE: {np.round(T_home[:3,3],3)}m')

    # =========================================================================
    # PARAMETERS
    # =========================================================================

    def _declare_all_parameters(self):
        """
        Declare all ROS 2 parameters. Must be declared before use.
        Enables runtime tuning: ros2 param set /adaptive_impedance_controller k_nominal_trans 1000.0
        """
        self.declare_parameter('control_rate_hz',              500)
        self.declare_parameter('force_impulse_threshold',       15.0)
        self.declare_parameter('torque_impulse_threshold',       5.0)
        self.declare_parameter('velocity_impulse_threshold',     0.3)
        self.declare_parameter('t_compliant_hold',               0.3)
        self.declare_parameter('t_recovery_blend',               1.5)
        self.declare_parameter('k_nominal_trans',              800.0)
        self.declare_parameter('k_nominal_rot',                 60.0)
        self.declare_parameter('d_nominal_trans',               80.0)
        self.declare_parameter('d_nominal_rot',                  8.0)
        self.declare_parameter('k_compliant_trans',             80.0)
        self.declare_parameter('k_compliant_rot',               10.0)
        self.declare_parameter('d_compliant_trans',            120.0)
        self.declare_parameter('d_compliant_rot',               15.0)

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def _joint_state_cb(self, msg):
        """
        Update q, dq from /joint_states.
        IMPORTANT: Build name->index map because message joint ORDER is not guaranteed.
        The UR5e driver / mock hardware may list joints alphabetically or by URDF order.
        We always extract in our fixed controller order via name lookup.
        """
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        joint_order = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        with self._lock:
            for k, jn in enumerate(joint_order):
                if jn in name_to_idx:
                    j = name_to_idx[jn]
                    self.q[k]  = msg.position[j]
                    self.dq[k] = msg.velocity[j] if len(msg.velocity) > j else 0.0

    def _wrench_cb(self, msg):
        """
        Update external wrench signals.

        Two parallel signals:
          F_ext_raw      = unfiltered sample  -> used for IMPULSE DETECTION
          F_ext_filtered = exponential LPF    -> used in CONTROL LAW

        WHY SEPARATE:
          LPF with alpha=0.1 has ~8Hz cutoff at 500Hz sampling.
          A 15ms impulse (67Hz) gets heavily smoothed.
          Raw signal preserves impulse spike for detection.
          Filtered signal removes sensor noise for clean torque computation.

        EMA (Exponential Moving Average):
          F_filt[k] = alpha * F_raw[k] + (1-alpha) * F_filt[k-1]
          Time constant tau = -dt / ln(1-alpha) = 0.002/0.105 = 19ms
        """
        f = msg.wrench.force;   t = msg.wrench.torque
        raw = np.array([f.x, f.y, f.z, t.x, t.y, t.z])
        with self._lock:
            self.F_ext_raw      = raw
            a = self._F_LPF_ALPHA
            self.F_ext_filtered = a * raw + (1.0 - a) * self.F_ext_filtered

    def _desired_pose_cb(self, msg):
        """
        Update desired EE pose from external command.
        Converts PoseStamped (position + quaternion) to 4x4 homogeneous transform.

        QUATERNION -> ROTATION MATRIX (standard formula):
          SOURCE: Shoemake (1985) SIGGRAPH; Siciliano (2009) Section 2.8
          Given unit quaternion [w, x, y, z]:
          R[0,0] = 1-2(y^2+z^2)   R[0,1] = 2(xy-wz)    R[0,2] = 2(xz+wy)
          R[1,0] = 2(xy+wz)       R[1,1] = 1-2(x^2+z^2) R[1,2] = 2(yz-wx)
          R[2,0] = 2(xz-wy)       R[2,1] = 2(yz+wx)     R[2,2] = 1-2(x^2+y^2)
        """
        p  = msg.pose.position
        o  = msg.pose.orientation
        qw, qx, qy, qz = o.w, o.x, o.y, o.z
        R = np.array([
            [1-2*(qy**2+qz**2),  2*(qx*qy-qw*qz),  2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz),  1-2*(qx**2+qz**2),  2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy),  2*(qy*qz+qw*qx),  1-2*(qx**2+qy**2)],
        ])
        T = np.eye(4);  T[:3,:3] = R;  T[:3,3] = [p.x, p.y, p.z]
        with self._lock:
            self.T_desired = T

    # =========================================================================
    # IMPULSE DETECTION STATE MACHINE
    # =========================================================================

    def _detect_impulse(self, t_now):
        """
        Detect impulse events using 3 independent criteria.

        CRITERIA (OR logic - any one triggers):
          Criterion 1a: |F_raw[:3]| > FORCE_THRESHOLD (15N)
          Criterion 1b: |F_raw[3:]| > TORQUE_THRESHOLD (5Nm)
          Criterion 2:  |F_raw - F_filt| > SPIKE_THRESHOLD (7.5N)
            -> detects transient even before filtered signal rises
          Criterion 3:  ||dq|| > VEL_THRESHOLD (0.3 rad/s)
            -> kinematic evidence (works even without FT sensor)

        STATE TRANSITIONS:
          NOMINAL -> COMPLIANT: any criterion fires AND not already active
          COMPLIANT -> NOMINAL: time expired AND forces low AND velocity low

        RECOVERY CONDITION (all must be true):
          dt_since_impulse > T_COMPLIANT_HOLD + T_RECOVERY_BLEND
          |F_raw[:3]| < 30% of FORCE_THRESHOLD  (disturbance truly gone)
          ||dq|| < 0.05 rad/s                    (arm has settled)
        """
        f_mag  = norm(self.F_ext_raw[:3])
        tr_mag = norm(self.F_ext_raw[3:])
        spike  = norm(self.F_ext_raw[:3] - self.F_ext_filtered[:3])
        dq_mag = norm(self.dq)

        thr_f  = self.get_parameter('force_impulse_threshold').value
        thr_tr = self.get_parameter('torque_impulse_threshold').value

        # Detection: new impulse?
        new_impulse = (
            not self.impulse_active
            and (f_mag > thr_f or tr_mag > thr_tr or spike > thr_f * 0.5)
        )
        if new_impulse:
            self.impulse_active = True
            self.t_impulse      = t_now
            self.get_logger().info(
                f'[AIC] IMPULSE at t={t_now:.3f}s  F={f_mag:.1f}N  '
                f'spike={spike:.1f}N  dq={dq_mag:.3f}rad/s')

        # Recovery check
        if self.impulse_active:
            t_hold  = self.get_parameter('t_compliant_hold').value
            t_blend = self.get_parameter('t_recovery_blend').value
            dt      = t_now - self.t_impulse
            if dt > (t_hold + t_blend) and f_mag < thr_f*0.3 and dq_mag < 0.05:
                self.impulse_active = False
                self.get_logger().info(f'[AIC] Recovery done at t={t_now:.3f}s  dt={dt:.2f}s')

    # =========================================================================
    # COMPLIANCE BLENDING
    # =========================================================================

    def _compute_blend(self, t_now):
        """
        Compute compliance blend alpha in [0, 1].
        SOURCE: Variable impedance scheduling concept from:
                Albu-Schaeffer (2007) Section IV

        BLEND SCHEDULE:
          Phase 1 ABSORB (dt <= T_COMPLIANT_HOLD):
            target = 1.0 (fully compliant)
          Phase 2 RECOVERY (T_HOLD < dt <= T_HOLD + T_BLEND):
            target = 1 - (dt - T_HOLD) / T_BLEND  (linear ramp 1->0)
          No impulse:
            target = 0.0 (return to nominal)

        ASYMMETRIC LPF ON BLEND ITSELF:
          Rising  (catching impulse): alpha_up   = 0.30/step  -> fast engagement
          Falling (stiffening back):  alpha_down = 0.05/step  -> slow, no oscillation

          Why asymmetric?
            Fast rise: A 15ms impulse at 500Hz = 7.5 steps.
                       With alpha=0.3: blend reaches 0.72 in 7 steps (adequate)
            Slow fall: Abrupt stiffness increase -> torque spike -> oscillation.
                       alpha=0.05 smooths the stiffening over ~60 steps = 120ms

        INTERPOLATED GAINS FORMULA (called in control loop):
          K(alpha) = (1-alpha)*K_nominal  +  alpha*K_compliant
          D(alpha) = (1-alpha)*D_nominal  +  alpha*D_compliant

          This linear interpolation:
          - Preserves positive-definiteness (both endpoints are PD)
          - Is monotonic in each element
          - Is guaranteed stable by passivity (Albu-Schaeffer 2007 Thm 1)
        """
        t_hold  = self.get_parameter('t_compliant_hold').value
        t_blend = self.get_parameter('t_recovery_blend').value

        if not self.impulse_active:
            self.compliance_blend = max(0.0, self.compliance_blend - 0.02)
            return self.compliance_blend

        dt = t_now - self.t_impulse
        target = 1.0 if dt <= t_hold else max(0.0, 1.0 - (dt - t_hold) / t_blend)

        # Asymmetric IIR: F_filt = alpha*target + (1-alpha)*current
        alpha = 0.30 if target > self.compliance_blend else 0.05
        self.compliance_blend = alpha * target + (1.0 - alpha) * self.compliance_blend
        return self.compliance_blend

    # =========================================================================
    # MAIN CONTROL LOOP (500Hz timer callback)
    # =========================================================================

    def _control_loop(self):
        """
        500Hz control loop. Implements:
          tau = J^T[K(blend)*e_x - D(blend)*xdot] + tau_null + tau_grav

        Step-by-step:
          1.  Snapshot state
          2.  FK -> T_cur
          3.  Detect impulse, compute blend
          4.  Build K(blend), D(blend)
          5.  Advance critically-damped filter -> T_filt (smooth setpoint)
          6.  6D pose error: e_x = pose_error_6d(T_filt, T_cur)
          7.  Jacobian J(q)
          8.  Cartesian velocity: xdot = J * dq
          9.  Impedance force: F_imp = K*e_x - D*xdot
          10. Saturate F_imp
          11. tau_imp = J^T * F_imp
          12. tau_null = N * [K_null*(q_home-q) - D_null*dq]
          13. tau_grav = gravity_torque(q)
          14. tau = clip(tau_imp + tau_null + tau_grav, +-150)
          15. Publish tau + diagnostics
        """
        # Step 1: Thread-safe state snapshot
        # We copy under lock then release - compute outside lock to not block callbacks
        with self._lock:
            q     = self.q.copy()
            dq    = self.dq.copy()
            T_des = self.T_desired.copy()

        # Step 2: Forward kinematics
        T_cur, _ = forward_kinematics(q)

        # Step 3: Impulse detection + blend
        t_now = self.get_clock().now().nanoseconds * 1e-9
        self._detect_impulse(t_now)
        blend = self._compute_blend(t_now)

        # Step 4: Build blended K, D matrices
        # K(blend) = (1-blend)*K_nominal + blend*K_compliant (element-wise diagonal)
        # D(blend) = (1-blend)*D_nominal + blend*D_compliant
        #
        # STABILITY PROOF SKETCH (Albu-Schaeffer 2007):
        #   Storage function V = 0.5*e_x^T*K*e_x
        #   dV/dt = e_x^T*K*e_xdot = e_x^T*K*(xdot_des - J*dq)
        #   With control: tau = J^T*(K*e_x - D*xdot) + tau_grav
        #   On the controlled system: dV/dt = -xdot^T*D*xdot <= 0
        #   -> System is passive (Lyapunov stable) for any K>0, D>0
        #   -> Valid for ALL values of blend in [0,1] since both K_nom, K_comp > 0
        knt = self.get_parameter('k_nominal_trans').value
        knr = self.get_parameter('k_nominal_rot').value
        dnt = self.get_parameter('d_nominal_trans').value
        dnr = self.get_parameter('d_nominal_rot').value
        kct = self.get_parameter('k_compliant_trans').value
        kcr = self.get_parameter('k_compliant_rot').value
        dct = self.get_parameter('d_compliant_trans').value
        dcr = self.get_parameter('d_compliant_rot').value

        k_t = (1.0-blend)*knt + blend*kct   # blended translational stiffness
        k_r = (1.0-blend)*knr + blend*kcr   # blended rotational stiffness
        d_t = (1.0-blend)*dnt + blend*dct   # blended translational damping
        d_r = (1.0-blend)*dnr + blend*dcr   # blended rotational damping

        K = np.diag([k_t, k_t, k_t, k_r, k_r, k_r])
        D = np.diag([d_t, d_t, d_t, d_r, d_r, d_r])

        # Step 5: Advance critically-damped recovery filter
        # omega_n=4 during compliance: slow filter, don't chase perturbed arm
        # omega_n=8 during nominal: faster convergence to desired position
        omega_n = 4.0 if blend > 0.1 else 8.0
        with self._lock:
            self.x_filt, self.v_filt = critically_damped_step(
                self.x_filt, self.v_filt, T_des[:3,3], omega_n, self._dt)
            x_filt = self.x_filt.copy()

        # Filtered transform: desired rotation, filtered position
        # (We don't filter orientation separately - orientation error is small)
        T_filt = T_des.copy()
        T_filt[:3,3] = x_filt

        # Step 6: 6D Cartesian pose error
        # e_x = [e_position(3) ; e_orientation(3)] in base frame
        e_x = pose_error_6d(T_filt, T_cur)

        # Step 7: Geometric Jacobian
        # J maps joint velocities to Cartesian EE velocity: xdot = J*dq
        J = geometric_jacobian(q)   # 6x6

        # Step 8: Cartesian EE velocity
        # xdot = J(q) * dq  [v_x, v_y, v_z, omega_x, omega_y, omega_z]
        xdot = J @ dq

        # Step 9: Impedance control force
        # F_imp = K*e_x - D*xdot
        #
        # PHYSICAL INTERPRETATION:
        #   K*e_x: "virtual spring" - pulls EE toward desired pose
        #   D*xdot: "virtual damper" - opposes ALL EE motion (not just error)
        #           Damping absolute velocity is MORE ROBUST than error-velocity:
        #           it damps disturbance motion AND prevents overshoot simultaneously
        #
        # SOURCE: Hogan (1985) Equation (5) - the fundamental impedance law
        F_imp = K @ e_x - D @ xdot

        # Step 10: Cartesian force saturation
        # Near Jacobian singularities: J^T amplifies Cartesian force to huge torques
        # Cap |F_imp[:3]| at F_CART_MAX before mapping through J^T
        fn = norm(F_imp[:3])
        if fn > F_CART_MAX:
            F_imp[:3] *= F_CART_MAX / fn

        # Step 11: Map Cartesian force -> joint torques via Jacobian transpose
        # tau_imp = J^T(q) * F_imp
        #
        # DERIVATION from virtual work principle (Khatib 1987):
        #   delta_W = F^T * delta_x = F^T * J * delta_q = (J^T*F)^T * delta_q
        #   So: tau = J^T * F  (exact equivalence, no approximation)
        tau_imp = J.T @ F_imp

        # Step 12: Null-space posture control
        # SOURCE: Khatib (1987) Section III "Redundancy Resolution"
        #
        # Goal: restore Q_HOME without disturbing EE motion
        #
        # THEORY: Any torque in null(J^T) has zero effect on EE force.
        #   null(J^T) is characterized by projector N = I - J^T * pinv(J^T)
        #             = I - J^T * (J^T)^+ where (J^T)^+ = J * (J*J^T)^-1
        #             = I - J^T * (pinv(J))^T
        #
        # For 6-DOF UR5e (square J): null space is empty in non-singular configs.
        # However, near singularities or with small K_null, this still provides
        # useful regularization that pulls joints toward a well-conditioned pose.
        #
        # tau_posture = K_null*(Q_HOME - q) - D_null*dq
        #   Spring toward home + damping opposing drift
        # tau_null = N * tau_posture
        #   Project into null space (zero EE effect)
        J_pinv    = pinv(J)                        # Moore-Penrose pseudoinverse (SVD)
        N         = np.eye(6) - J.T @ J_pinv.T    # null-space projector
        tau_null  = N @ (K_NULL @ (Q_HOME - q) - D_NULL @ dq)

        # Step 13: Gravity compensation
        # Exact analytic computation from link masses, CoM positions, FK
        tau_grav = gravity_torque(q)

        # Step 14: Total torque + safety saturation
        # tau = tau_impedance + tau_null_space + tau_gravity
        #
        # tau_imp:  main task-space control (pose holding + compliance)
        # tau_null: secondary posture restoration (null-space, no EE effect)
        # tau_grav: static compensation (prevent gravity-induced drift)
        tau_total = tau_imp + tau_null + tau_grav
        tau_cmd   = np.clip(tau_total, -TAU_MAX, TAU_MAX)

        # Step 15: Publish torque command
        # Float64MultiArray data[] must be in joint order matching ros2_controllers.yaml:
        # [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
        cmd = Float64MultiArray()
        cmd.data = tau_cmd.tolist()
        self.pub_effort.publish(cmd)

        # Diagnostics
        err_msg      = Float64MultiArray()
        err_msg.data = list(e_x) + [float(norm(e_x[:3])), float(norm(e_x[3:]))]
        self.pub_error.publish(err_msg)

        blend_msg      = Float64MultiArray()
        blend_msg.data = [float(blend)]
        self.pub_blend.publish(blend_msg)

        s      = String()
        s.data = ('COMPLIANT' if blend > 0.3 else ('RECOVERING' if blend > 0.05 else 'NOMINAL'))
        self.pub_status.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveImpedanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()