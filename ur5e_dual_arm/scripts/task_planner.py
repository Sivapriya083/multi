#!/usr/bin/env python3
"""
Dual UR5e Task Planner — Day 4 (No Bins Version)
-------------------------------------------------
Each arm picks its detected object, lifts it, carries it to a fixed
DROP ZONE, and releases it there. No bins needed in Gazebo.

Full cycle per arm:
  1. Approach  — hover above object     (object z + 0.12 m)
  2. Descend   — move to grasp point    (object z + 0.05 m)
  3. GRASP     — simulated attach (log message)
  4. Lift      — raise straight up      (object z + 0.20 m)
  5. Carry     — move to drop zone      (joint-space, fast)
  6. Descend   — lower into drop zone   (drop z + 0.05 m)
  7. RELEASE   — simulated detach (log message)
  8. Retreat   — lift back up           (drop z + 0.15 m)
  9. Home      — return to home pose

DROP ZONES — adjust x,y,z to match your table layout:
  LEFT  arm drops at  x=0.3, y=+0.45, z=0.55  (behind the sphere)
  RIGHT arm drops at  x=0.3, y=-0.45, z=0.55  (behind the cylinder)

SRDF constants (verified against ur5e_dual_arm.srdf):
  left_arm  : EE = left_tool0
  right_arm : EE = right_tool0
  home      : [0.0, -1.5707, 1.5707, 0.0, 0.0, 0.0]
"""

import threading
import copy
import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    CollisionObject,
    PlanningScene,
    BoundingVolume,
)
from moveit_msgs.srv import GetCartesianPath, ApplyPlanningScene
from rclpy.action import ActionClient


# ── robot constants (from SRDF) ───────────────────────────────────────────────
MOVE_GROUP_LEFT    = 'left_arm'
MOVE_GROUP_RIGHT   = 'right_arm'
EE_LINK_LEFT       = 'left_tool0'
EE_LINK_RIGHT      = 'right_tool0'
HOME_POSE_NAME     = 'home'
JOINT_STATES_TOPIC = '/joint_states'

# ── drop zone positions (ADJUST TO YOUR TABLE LAYOUT) ────────────────────────
# Left arm drops objects here (y > 0 side, somewhere behind/beside the sphere)
DROP_ZONE_LEFT  = {'x': 0.30, 'y':  0.45, 'z': 0.55}

# Right arm drops objects here (y < 0 side, somewhere behind/beside cylinder)
DROP_ZONE_RIGHT = {'x': 0.30, 'y': -0.45, 'z': 0.55}

# ── motion parameters ─────────────────────────────────────────────────────────
CARTESIAN_STEP        = 0.005
CARTESIAN_JUMP_THRESH = 0.0
CARTESIAN_MIN_FRAC    = 0.85
APPROACH_HEIGHT       = 0.12   # hover above object before descending
REACH_OFFSET          = 0.05   # stop above object centroid (grasp point)
LIFT_HEIGHT           = 0.20   # lift this high after grasping
DROP_DESCENT          = 0.05   # descend to this height above drop zone
DROP_RETREAT          = 0.15   # rise this high after releasing
MAX_VELOCITY          = 0.3
MAX_ACCELERATION      = 0.3

# ── time budget ───────────────────────────────────────────────────────────────
TASK_DURATION_SECONDS = 300.0  # 5 minutes
# ─────────────────────────────────────────────────────────────────────────────


def wait_event(node: Node, future, timeout: float = 15.0):
    """Block current thread until future completes. Safe inside executor."""
    event = threading.Event()
    future.add_done_callback(lambda _: event.set())
    event.wait(timeout=timeout)
    return future.result() if future.done() else None


def make_pose(x, y, z, qw=1.0) -> Pose:
    p = Pose()
    p.position.x    = x
    p.position.y    = y
    p.position.z    = z
    p.orientation.w = qw
    return p


# ── planning scene ────────────────────────────────────────────────────────────

class PlanningSceneManager:
    """Adds the table surface to MoveIt2 so arms avoid it."""

    def __init__(self, node: Node):
        self.node = node
        self.cli  = node.create_client(
            ApplyPlanningScene, '/apply_planning_scene',
            callback_group=node.cb_group)
        self.cli.wait_for_service(timeout_sec=10.0)

    def setup(self):
        # Table surface: 80 x 60 x 2 cm slab at z = 0.49
        table               = CollisionObject()
        table.id            = 'table'
        table.header.frame_id = 'world'
        table.operation     = CollisionObject.ADD
        prim                = SolidPrimitive()
        prim.type           = SolidPrimitive.BOX
        prim.dimensions     = [0.80, 0.80, 0.02]
        table.primitives.append(prim)
        tp                  = Pose()
        tp.position.x       = 0.5
        tp.position.y       = 0.0
        tp.position.z       = 0.49
        tp.orientation.w    = 1.0
        table.primitive_poses.append(tp)

        scene               = PlanningScene()
        scene.is_diff       = True
        scene.world.collision_objects = [table]

        req       = ApplyPlanningScene.Request()
        req.scene = scene
        wait_event(self.node, self.cli.call_async(req), timeout=5.0)
        self.node.get_logger().info('Planning scene: table collision object added')


# ── time budget ───────────────────────────────────────────────────────────────

class TimeBudget:
    def __init__(self, total: float):
        self.total = total
        self.start = time.time()

    def fraction(self) -> float:
        return min((time.time() - self.start) / self.total, 1.0)

    def elapsed(self) -> float:
        return time.time() - self.start

    def should_process(self, category: str) -> bool:
        f = self.fraction()
        if f > 0.90:
            return category == 'large'
        if f > 0.80:
            return category in ('large', 'medium')
        return True

    def status_str(self) -> str:
        f = self.fraction()
        mode = ('LARGE ONLY' if f > 0.90
                else 'SKIPPING SMALL' if f > 0.80
                else 'NORMAL')
        return (f'Time: {self.elapsed():.0f}s / {self.total:.0f}s '
                f'({f*100:.0f}%) [{mode}]')


# ── metrics ───────────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.picked   = 0
        self.placed   = 0
        self.failed   = 0
        self.skipped  = 0
        self._t0      = None
        self.cycles   = []

    def start(self):  self._t0 = time.time()

    def done(self, success: bool):
        if self._t0:
            self.cycles.append(time.time() - self._t0)
        if success:
            self.placed += 1
        else:
            self.failed += 1

    def skip(self): self.skipped += 1

    def report(self) -> str:
        total = self.placed + self.failed + self.skipped
        acc   = self.placed / total * 100 if total else 0.0
        avg   = sum(self.cycles) / len(self.cycles) if self.cycles else 0.0
        return (f'METRICS | Placed: {self.placed} | Failed: {self.failed} | '
                f'Skipped: {self.skipped} | Accuracy: {acc:.1f}% | '
                f'Avg cycle: {avg:.1f}s')


# ── arm controller ────────────────────────────────────────────────────────────

class ArmController:
    """MoveIt2 pick-and-place controller for one UR5e arm."""

    def __init__(self, node: 'TaskPlannerNode', group: str, ee: str):
        self.node  = node
        self.group = group
        self.ee    = ee
        self.log   = node.get_logger()
        self.cbg   = node.cb_group

        self.cart_cli = node.create_client(
            GetCartesianPath, '/compute_cartesian_path',
            callback_group=self.cbg)
        self.exec_cli = ActionClient(
            node, ExecuteTrajectory, '/execute_trajectory',
            callback_group=self.cbg)
        self.move_cli = ActionClient(
            node, MoveGroup, '/move_action',
            callback_group=self.cbg)

        self.log.info(f'[{group}] Connecting to MoveIt2...')
        self.cart_cli.wait_for_service(timeout_sec=20.0)
        self.exec_cli.wait_for_server(timeout_sec=20.0)
        self.move_cli.wait_for_server(timeout_sec=20.0)
        self.log.info(f'[{group}] Connected. EE link: {ee}')

    # ── pick ──────────────────────────────────────────────────────────────────

    def pick(self, obj_pose: Pose, label: str) -> bool:
        """
        Approach → descend to grasp point → log grasp → lift.
        All three phases are one continuous Cartesian path.
        """
        approach = copy.deepcopy(obj_pose)
        approach.position.z = obj_pose.position.z + APPROACH_HEIGHT

        grasp = copy.deepcopy(obj_pose)
        grasp.position.z = obj_pose.position.z + REACH_OFFSET

        lift = copy.deepcopy(obj_pose)
        lift.position.z = obj_pose.position.z + LIFT_HEIGHT

        self.log.info(
            f'[{self.group}] PICK {label}: '
            f'approach z={approach.position.z:.3f} -> '
            f'grasp z={grasp.position.z:.3f} -> '
            f'lift z={lift.position.z:.3f}')

        # Phase 1: approach + descend
        if not self._cartesian([approach, grasp], 'descend'):
            return False

        # Simulated grasp
        self.node.publish_status(f'[{self.group}] *** GRASPED {label} ***')

        # Phase 2: lift straight up
        if not self._cartesian([lift], 'lift'):
            return False

        return True

    # ── place ─────────────────────────────────────────────────────────────────

    def place(self, drop: Pose, label: str) -> bool:
        """
        Carry to drop zone (joint-space) → descend → log release → retreat.
        """
        above_drop = copy.deepcopy(drop)
        above_drop.position.z = drop.position.z + APPROACH_HEIGHT

        descend = copy.deepcopy(drop)
        descend.position.z = drop.position.z + DROP_DESCENT

        retreat = copy.deepcopy(drop)
        retreat.position.z = drop.position.z + DROP_RETREAT

        self.log.info(
            f'[{self.group}] PLACE {label} at drop zone '
            f'({drop.position.x:.3f},{drop.position.y:.3f},{drop.position.z:.3f})')

        # Phase 3: carry to above drop zone (joint-space — fast, path doesn't matter)
        if not self._joint_move_to(above_drop, 'carry to drop zone'):
            # Fallback: Cartesian carry
            self.log.warn(f'[{self.group}] Joint-space carry failed, trying Cartesian')
            if not self._cartesian([above_drop], 'carry to drop zone'):
                return False

        # Phase 4: descend into drop zone
        if not self._cartesian([descend], 'descend to drop zone'):
            return False

        # Simulated release
        self.node.publish_status(f'[{self.group}] *** RELEASED {label} ***')

        # Phase 5: retreat back up
        if not self._cartesian([retreat], 'retreat'):
            return False

        return True

    # ── go home ───────────────────────────────────────────────────────────────

    def go_home(self) -> bool:
        req                                  = MotionPlanRequest()
        req.group_name                       = self.group
        req.num_planning_attempts            = 10
        req.allowed_planning_time            = 5.0
        req.max_velocity_scaling_factor      = MAX_VELOCITY
        req.max_acceleration_scaling_factor  = MAX_ACCELERATION
        req.start_state                      = self.node.get_robot_state()
        c                                    = Constraints()
        c.name                               = HOME_POSE_NAME
        req.goal_constraints                 = [c]

        goal         = MoveGroup.Goal()
        goal.request = req

        self.log.info(f'[{self.group}] Going home...')
        gh = wait_event(self.node, self.move_cli.send_goal_async(goal), 10.0)
        if not gh or not gh.accepted:
            self.log.error(f'[{self.group}] Home goal rejected')
            return False

        result = wait_event(self.node, gh.get_result_async(), 20.0)
        if result and result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.log.info(f'[{self.group}] Home reached')
            return True

        code = result.result.error_code.val if result else 'timeout'
        self.log.error(f'[{self.group}] Home failed (code={code})')
        return False

    # ── internal: Cartesian motion ────────────────────────────────────────────

    def _cartesian(self, waypoints: list, phase: str) -> bool:
        req                  = GetCartesianPath.Request()
        req.header.frame_id  = 'world'
        req.header.stamp     = self.node.get_clock().now().to_msg()
        req.group_name       = self.group
        req.link_name        = self.ee
        req.waypoints        = waypoints
        req.max_step         = CARTESIAN_STEP
        req.jump_threshold   = CARTESIAN_JUMP_THRESH
        req.avoid_collisions = True
        req.start_state      = self.node.get_robot_state()

        resp = wait_event(self.node, self.cart_cli.call_async(req), 10.0)
        if resp is None:
            self.log.error(f'[{self.group}] [{phase}] Cartesian timeout')
            return False

        frac = resp.fraction
        self.log.info(f'[{self.group}] [{phase}] Coverage: {frac*100:.1f}%')

        if frac < CARTESIAN_MIN_FRAC:
            self.log.warn(
                f'[{self.group}] [{phase}] Coverage {frac*100:.1f}% too low, skipping')
            return False

        if resp.error_code.val not in (
            MoveItErrorCodes.SUCCESS,
            MoveItErrorCodes.GOAL_TOLERANCE_VIOLATED,
        ):
            self.log.error(f'[{self.group}] [{phase}] Plan error: {resp.error_code.val}')
            return False

        return self._execute(resp.solution, phase)

    # ── internal: joint-space move to pose ────────────────────────────────────

    def _joint_move_to(self, target: Pose, phase: str) -> bool:
        """Joint-space motion to a Cartesian target. Used for carrying."""
        pos_c                         = PositionConstraint()
        pos_c.header.frame_id         = 'world'
        pos_c.link_name               = self.ee
        region                        = BoundingVolume()
        prim                          = SolidPrimitive()
        prim.type                     = SolidPrimitive.SPHERE
        prim.dimensions               = [0.02]   # 2 cm tolerance
        region.primitives.append(prim)
        ps                            = PoseStamped()
        ps.header.frame_id            = 'world'
        ps.pose                       = target
        region.primitive_poses.append(ps)
        pos_c.constraint_region       = region
        pos_c.weight                  = 1.0

        gc = Constraints()
        gc.position_constraints.append(pos_c)

        req                                  = MotionPlanRequest()
        req.group_name                       = self.group
        req.num_planning_attempts            = 5
        req.allowed_planning_time            = 5.0
        req.max_velocity_scaling_factor      = MAX_VELOCITY
        req.max_acceleration_scaling_factor  = MAX_ACCELERATION
        req.start_state                      = self.node.get_robot_state()
        req.goal_constraints                 = [gc]

        goal         = MoveGroup.Goal()
        goal.request = req

        gh = wait_event(self.node, self.move_cli.send_goal_async(goal), 10.0)
        if not gh or not gh.accepted:
            return False

        result = wait_event(self.node, gh.get_result_async(), 20.0)
        if result and result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.log.info(f'[{self.group}] [{phase}] Done')
            return True
        return False

    # ── internal: execute trajectory ──────────────────────────────────────────

    def _execute(self, traj: RobotTrajectory, phase: str) -> bool:
        goal            = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        gh = wait_event(self.node, self.exec_cli.send_goal_async(goal), 10.0)
        if not gh or not gh.accepted:
            self.log.error(f'[{self.group}] [{phase}] Execute rejected')
            return False

        result = wait_event(self.node, gh.get_result_async(), 30.0)
        if result and result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            return True

        code = result.result.error_code.val if result else 'timeout'
        self.log.error(f'[{self.group}] [{phase}] Execute failed (code={code})')
        return False


# ── main node ─────────────────────────────────────────────────────────────────

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')
        self.cb_group = ReentrantCallbackGroup()

        # Joint state cache
        self._js       = None
        self._js_lock  = threading.Lock()
        self._js_event = threading.Event()

        self.js_sub = self.create_subscription(
            JointState, JOINT_STATES_TOPIC,
            self._js_cb, 10,
            callback_group=self.cb_group)

        self.get_logger().info(
            'Waiting for valid /joint_states (non-zero timestamp)...\n'
            'This can take 10-30s while Gazebo clock initialises.')
        if not self._js_event.wait(timeout=60.0):
            raise RuntimeError(
                'No valid /joint_states after 60s. '
                'Check: ros2 control list_controllers')
        self.get_logger().info('Joint states OK')

        # Planning scene (table only — no bins)
        self.scene = PlanningSceneManager(self)
        self.scene.setup()

        # Arms
        self.left_arm  = ArmController(self, MOVE_GROUP_LEFT,  EE_LINK_LEFT)
        self.right_arm = ArmController(self, MOVE_GROUP_RIGHT, EE_LINK_RIGHT)

        # Subscriptions
        self.obj_sub = self.create_subscription(
            PoseArray, '/detected_objects',
            self.objects_callback, 10,
            callback_group=self.cb_group)

        self.cat_sub = self.create_subscription(
            String, '/detected_objects_categories',
            self._cat_cb, 10,
            callback_group=self.cb_group)

        self.status_pub = self.create_publisher(String, '/task_status', 10)

        self._categories: list = []
        self.busy    = False
        self.metrics = Metrics()
        self.budget  = TimeBudget(TASK_DURATION_SECONDS)

        self.publish_status(
            'Day 4 Task Planner ready\n'
            f'  Left  drop zone : x={DROP_ZONE_LEFT["x"]}  '
            f'y={DROP_ZONE_LEFT["y"]}  z={DROP_ZONE_LEFT["z"]}\n'
            f'  Right drop zone : x={DROP_ZONE_RIGHT["x"]}  '
            f'y={DROP_ZONE_RIGHT["y"]}  z={DROP_ZONE_RIGHT["z"]}')

    # ── joint state ───────────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        # Ignore messages with zero timestamp -- sim not ready yet
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            self.get_logger().info(
                'Joint state received but timestamp is zero -- '
                'waiting for simulation clock...')
            return
        with self._js_lock:
            self._js = msg
        self._js_event.set()

    def get_robot_state(self) -> RobotState:
        with self._js_lock:
            js = copy.deepcopy(self._js)
        rs             = RobotState()
        rs.joint_state = js
        return rs

    # ── category cache ────────────────────────────────────────────────────────

    def _cat_cb(self, msg: String):
        self._categories = [c.strip() for c in msg.data.split(',') if c.strip()]

    # ── object callback ───────────────────────────────────────────────────────

    def objects_callback(self, msg: PoseArray):
        if self.busy or len(msg.poses) == 0:
            return
        self.busy = True
        threading.Thread(
            target=self.execute_task,
            args=(msg.poses, list(self._categories)),
            daemon=True
        ).start()

    # ── task execution ────────────────────────────────────────────────────────

    def execute_task(self, poses, categories):
        try:
            self.publish_status(self.budget.status_str())

            # Pair each pose with its category (default 'medium')
            paired = []
            for i, p in enumerate(poses):
                cat = categories[i] if i < len(categories) else 'medium'
                paired.append((p, cat))

            # Time budget filter
            filtered = [(p, c) for p, c in paired
                        if self.budget.should_process(c)]
            skipped  = len(paired) - len(filtered)
            for _ in range(skipped):
                self.metrics.skip()

            if skipped:
                self.publish_status(
                    f'Time budget: skipping {skipped} object(s)')

            # Split by arm side
            left_targets  = [(p, c) for p, c in filtered
                             if p.position.y >= 0]
            right_targets = [(p, c) for p, c in filtered
                             if p.position.y <  0]

            self.publish_status(
                f'Processing: LEFT={len(left_targets)} '
                f'RIGHT={len(right_targets)} '
                f'Skipped={skipped}')

            threads = []

            if left_targets:
                dz = make_pose(
                    DROP_ZONE_LEFT['x'],
                    DROP_ZONE_LEFT['y'],
                    DROP_ZONE_LEFT['z'])
                threads.append(threading.Thread(
                    target=self._run_arm,
                    args=(self.left_arm, left_targets, dz, 'LEFT'),
                    daemon=True))
            else:
                self.publish_status('No left-side objects -- left arm idle')

            if right_targets:
                dz = make_pose(
                    DROP_ZONE_RIGHT['x'],
                    DROP_ZONE_RIGHT['y'],
                    DROP_ZONE_RIGHT['z'])
                threads.append(threading.Thread(
                    target=self._run_arm,
                    args=(self.right_arm, right_targets, dz, 'RIGHT'),
                    daemon=True))
            else:
                self.publish_status('No right-side objects -- right arm idle')

            # Both arms work simultaneously
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Report metrics
            self.publish_status(self.metrics.report())

            # Both arms go home simultaneously
            self.publish_status('Returning home...')
            home_threads = [
                threading.Thread(target=self.left_arm.go_home,  daemon=True),
                threading.Thread(target=self.right_arm.go_home, daemon=True),
            ]
            for t in home_threads:
                t.start()
            for t in home_threads:
                t.join()

            self.publish_status('Cycle complete')

        except Exception as e:
            self.get_logger().error(f'execute_task error: {e}')
        finally:
            self.busy = False

    def _run_arm(self, arm: ArmController,
                 targets: list, drop_zone: Pose, side: str):
        """Pick and place every object assigned to this arm."""
        for pose, category in targets:
            label = (f'{category}@'
                     f'({pose.position.x:.2f},'
                     f'{pose.position.y:.2f},'
                     f'{pose.position.z:.2f})')

            self.publish_status(
                f'[{side}] Starting pick-place for {label}')

            self.metrics.start()
            success = False

            try:
                # Step 1 — Pick
                if not arm.pick(pose, label):
                    self.publish_status(f'[{side}] Pick FAILED: {label}')
                    self.metrics.done(False)
                    continue

                # Step 2 — Place at drop zone
                if not arm.place(drop_zone, label):
                    self.publish_status(f'[{side}] Place FAILED: {label}')
                    self.metrics.done(False)
                    continue

                success = True
                self.publish_status(
                    f'[{side}] Pick-place COMPLETE: {label}')

            except Exception as e:
                self.get_logger().error(
                    f'[{side}] Error on {label}: {e}')
            finally:
                self.metrics.done(success)

    # ── helper ────────────────────────────────────────────────────────────────

    def publish_status(self, msg: str):
        self.get_logger().info(msg)
        s      = String()
        s.data = msg
        self.status_pub.publish(s)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    executor    = MultiThreadedExecutor(num_threads=10)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node = TaskPlannerNode()
        executor.add_node(node)
        spin_thread.join()
    except RuntimeError as e:
        print(f'[FATAL] {e}')
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()