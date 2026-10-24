"""A passive drawer, right-arm open–look–close script, and physical task tracking."""

import math
from copy import deepcopy

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import ArmTag


def camera_above_opening(camera, center, height=0.10):
    """Allow an approximate overhead view; the separate ray check verifies visibility."""
    return bool(
        np.isfinite(camera).all()
        and np.linalg.norm(camera[:2, 3] - center[:2]) <= 0.05
        and abs(camera[2, 3] - center[2] - height) <= 0.05
        and camera[2, 0] <= -math.cos(math.radians(30))
    )


class look_into_drawer(Base_Task):
    """Inspect an empty drawer once with the actual right wrist camera."""

    travel = 0.18
    open_fraction = 0.70
    hover_height = 0.10
    look_seconds = 0.25
    rim_height = 0.108

    def setup_demo(self, **kwargs):
        self._state = None
        self._elapsed = 0.0
        self._inspection = None
        self._planner_original_world = None
        self._sweep_active = False
        self._sweep_contact_impulse = 0.0
        repeats = kwargs.get("repeat_count", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats != 1:
            raise ValueError("look_into_drawer requires repeat_count=1")
        self.repeat_count = 1
        embodiment = kwargs.get("embodiment_name", str(kwargs.get("embodiment", "")))
        self._is_piper = "piper" in embodiment.lower()
        # Piper's longer fingers need an extra 2 cm of palm-to-handle clearance.
        self._grasp_depth = 0.14 if self._is_piper else 0.12
        for name, default in (
            ("drawer_x_range", [0.18, 0.22]),
            ("drawer_y_range", [0.05, 0.07]),
            ("drawer_yaw_range", [-0.05, 0.05]),
        ):
            values = np.asarray(kwargs.get(name, default), dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
                raise ValueError(f"{name} must be a finite [minimum, maximum] pair")
            setattr(self, name, values)
        super()._init_task_env_(**kwargs)
        self.robot.set_origin_endpose()
        self._ready_left = np.asarray(self.robot.left_original_pose)
        self._ready_right = np.asarray(self.robot.right_original_pose)
        self._fingers = [joint.child_link.entity for joint, _, _ in self.robot.right_gripper]
        self._fingers.extend(
            link.entity
            for link in self.robot.right_entity.get_links()
            if link.get_name() in self.robot.right_fix_gripper_name
        )
        self._robot_links = {link.entity for link in self.robot.right_entity.get_links()}
        self._fixture_links = {link.entity for link in self.drawer.get_links()}
        self._state = LookIntoDrawerState(self.travel, self.open_fraction, self.look_seconds)
        self._state.update(
            position=float(self.drawer.get_qpos()[0]),
            dt=self.scene.get_timestep(),
            held=False,
            released=True,
            view_valid=False,
            arms_ready=True,
            left_ready=True,
            still=True,
        )
        # Keep the fixture and hand in view, in front of Aloha's camera mast.
        for name, camera in zip(self.cameras.static_camera_name, self.cameras.static_camera_list):
            if name == "head_camera":
                position = np.array([0.24, -0.43, 1.55 + self.table_z_bias])
                forward = np.array([0.17, -0.04, 0.83 + self.table_z_bias]) - position
                forward /= np.linalg.norm(forward)
                left = np.cross([0, 0, 1], forward)
                left /= np.linalg.norm(left)
                matrix = np.eye(4)
                matrix[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
                matrix[:3, 3] = position
                camera.entity.set_pose(sapien.Pose(matrix))
        self.set_instruction("Use the right arm to look into the drawer.")
        self.info["info"] = {"{A}": "empty drawer", "{a}": "right", "{N}": "1"}

    def load_actors(self):
        """Build one empty drawer with a passive, limited prismatic joint."""
        x, y, yaw = [
            np.random.uniform(*getattr(self, name))
            for name in ("drawer_x_range", "drawer_y_range", "drawer_yaw_range")
        ]
        self.fixture_pose = sapien.Pose(
            [x, y, 0.74 + self.table_z_bias],
            [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
        )
        builder = self.scene.create_articulation_builder()
        housing = builder.create_link_builder()
        housing.set_name("drawer_housing")
        tray = builder.create_link_builder(housing)
        tray.set_name("drawer_tray")
        tray.set_joint_name("drawer_slide")
        axis_pose = sapien.Pose(q=[math.sqrt(0.5), 0, 0, -math.sqrt(0.5)])
        tray.set_joint_properties(
            "prismatic", [[0, self.travel]], axis_pose, axis_pose, friction=0.05, damping=2.0
        )
        wood = sapien.render.RenderMaterial(base_color=[0.43, 0.26, 0.14, 1], roughness=0.75)
        inside = sapien.render.RenderMaterial(base_color=[0.74, 0.57, 0.35, 1], roughness=0.8)
        metal = sapien.render.RenderMaterial(base_color=[0.17, 0.19, 0.21, 1], metallic=0.6)
        material = self.scene.create_physical_material(1.2, 1.0, 0.0)
        self._fixture_boxes = []

        def box(link, position, half_size, color):
            pose = sapien.Pose(position)
            link.add_box_collision(pose=pose, half_size=half_size, material=material, density=200)
            link.add_box_visual(pose=pose, half_size=half_size, material=color)
            if color is not metal:
                self._fixture_boxes.append((link is tray, position, half_size))

        # Housing encloses the closed tray; 3 mm side clearance prevents binding.
        for position, size in (
            ([0, 0, 0.006], [0.128, 0.125, 0.006]),
            ([-0.122, 0, 0.073], [0.006, 0.125, 0.061]),
            ([0.122, 0, 0.073], [0.006, 0.125, 0.061]),
            ([0, 0.119, 0.073], [0.116, 0.006, 0.061]),
            ([0, 0, 0.128], [0.116, 0.125, 0.006]),
        ):
            box(housing, position, size, wood)
        for position, size in (
            ([0, 0, 0.020], [0.110, 0.108, 0.005]),
            ([-0.106, 0, 0.0665], [0.004, 0.108, 0.0415]),
            ([0.106, 0, 0.0665], [0.004, 0.108, 0.0415]),
            ([0, 0.104, 0.0665], [0.102, 0.004, 0.0415]),
            ([0, -0.112, 0.065], [0.116, 0.006, 0.058]),
        ):
            box(tray, position, size, inside)
        # Upright handle: fingers open laterally, clear of the drawer front.
        box(tray, [0, -0.150, 0.071], [0.006, 0.006, 0.035], metal)
        for side in (-1, 1):
            box(tray, [0, -0.133, 0.071 + side * 0.031], [0.006, 0.011, 0.004], metal)
        builder.set_initial_pose(self.fixture_pose)
        self.drawer = builder.build(fix_root_link=True)
        self.drawer.set_qpos([0])
        self.drawer.set_qvel([0])
        self.drawer_link = self.drawer.get_links()[1]
        self.drawer.get_active_joints()[0].set_drive_properties(stiffness=0, damping=2.0)

    def _handle_pose(self):
        # Approach downwards at 45 degrees to keep the wrist above the table.
        c = math.sqrt(0.5)
        matrix = np.eye(4)
        matrix[:3, :3] = [[0, -1, 0], [c, 0, c], [-c, 0, c]]
        matrix[:3, 3] = np.array([0, -0.150, 0.071]) - self._grasp_depth * matrix[:3, 0]
        return self.drawer_link.get_pose() * sapien.Pose(matrix)

    def _opening_center(self):
        q = float(self.drawer.get_qpos()[0])
        # Center of the exposed opening, excluding the housing overhang/front.
        return self.fixture_pose * sapien.Pose([0, -0.116 - q / 2, self.rim_height])

    def _view_target(self, roll):
        center = self._opening_center().p
        matrix = np.eye(4)
        forward = np.array([0.0, 0.0, -1.0])
        left = np.array([math.cos(roll), math.sin(roll), 0])
        matrix[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
        matrix[:3, 3] = center + [0, 0, self.hover_height]
        ee = self.robot.get_right_ee_pose()
        # Camera mounting differs between embodiments. Use its measured transform.
        return (
            sapien.Pose(matrix)
            * self.robot.right_camera.get_pose().inv()
            * sapien.Pose(ee[:3], ee[3:])
        )

    @staticmethod
    def _ready(current, target):
        current, target = np.asarray(current), np.asarray(target)
        return bool(
            np.linalg.norm(current[:3] - target[:3]) <= 0.05
            and abs(np.dot(current[3:], target[3:])) >= math.cos(math.radians(20) / 2)
        )

    def _contact(self):
        for contact in self.scene.get_contacts():
            entities = [body.entity for body in contact.bodies]
            if self.drawer_link.entity in entities and any(e in self._fingers for e in entities):
                if any(np.linalg.norm(p.impulse) > 1e-7 for p in contact.points):
                    return True
        return False

    def _view_valid(self):
        camera = self.robot.right_camera.get_pose().to_transformation_matrix()
        center = self._opening_center().p
        if not camera_above_opening(camera, center, self.hover_height):
            return False
        # The optical axis must actually see the tray floor, rather than a finger,
        # the front panel, or the cabinet roof. This also catches occluded views.
        # Match rendering's near plane: the optical center lies inside the
        # camera housing collider, which otherwise returns a spurious zero hit.
        near = self.cameras.right_camera.near
        ray_start = camera[:3, 3] + near * camera[:3, 0]
        hit = self.scene.physx_system.raycast(ray_start, camera[:3, 0], 0.30 - near)
        if hit is None or hit.component.entity != self.drawer_link.entity:
            return False
        local = self.drawer_link.get_pose().inv() * sapien.Pose(hit.position)
        return bool(
            abs(local.p[0]) < 0.10 and abs(local.p[1]) < 0.10 and abs(local.p[2] - 0.025) < 0.004
        )

    def _after_simulation_step(self):
        if self._state is None:
            return
        self._elapsed += self.scene.get_timestep()
        if self._sweep_active:
            for contact in self.scene.get_contacts():
                entities = {body.entity for body in contact.bodies}
                if not (entities & self._robot_links and entities & self._fixture_links):
                    continue
                impulse = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
                self._sweep_contact_impulse = max(self._sweep_contact_impulse, impulse)
                if impulse > 1e-4:
                    self._state.failed_reason = "robot_drawer_contact_during_camera_motion"
        contact = self._contact()
        left_ready = self._ready(self.robot.get_left_ee_pose(), self._ready_left)
        self._state.update(
            position=float(self.drawer.get_qpos()[0]),
            dt=self.scene.get_timestep(),
            held=contact and self.is_right_gripper_close(),
            released=not contact and self.is_right_gripper_open(),
            view_valid=self._view_valid(),
            arms_ready=left_ready
            and self._ready(self.robot.get_right_ee_pose(), self._ready_right),
            left_ready=left_ready,
            still=abs(float(self.drawer.get_qvel()[0])) < 0.01,
        )
        if self._state.looked and self._inspection is None:
            camera = self.robot.right_camera.get_pose()
            self._inspection = {
                "completed_at_seconds": self._elapsed,
                "camera_pose": camera.p.tolist() + camera.q.tolist(),
                "height_above_opening": float(camera.p[2] - self._opening_center().p[2]),
                "drawer_position": float(self.drawer.get_qpos()[0]),
                "uninterrupted_seconds": self._state.look_time,
            }

    def _hold(self, seconds):
        for _ in range(math.ceil(seconds / self.scene.get_timestep())):
            self._step_simulation()

    def _move(self, pose):
        self.move(self.move_to_pose(ArmTag("right"), pose))
        return self.plan_success and not self._state.failed_reason

    def _slide(self, target):
        # Correct measured compliance without prescribing the drawer's joint state.
        for _ in range(3):
            error = target - float(self.drawer.get_qpos()[0])
            if abs(error) <= 0.001:
                break
            if not self._contact() or not self.is_right_gripper_close():
                self.info["scripted_failure"] = "handle_grasp_not_observed"
                return False
            ee = self.robot.get_right_ee_pose()
            direction = self.fixture_pose.to_transformation_matrix()[:3, :3] @ [0, -1, 0]
            if not self._move(sapien.Pose(np.asarray(ee[:3]) + error * direction, ee[3:])):
                return False
        reached = abs(target - float(self.drawer.get_qpos()[0])) <= 0.002
        if not reached:
            self.info["scripted_failure"] = "drawer_slide_not_observed"
        return self.plan_success and reached and not self._state.failed_reason

    def _look(self):
        # Each candidate is a complete translation+rotation trajectory. Execute one.
        self._set_planner_fixture(True)
        left = self.robot.right_camera.get_pose().to_transformation_matrix()[:3, 1]
        natural_roll = math.atan2(left[1], left[0])
        for roll in (
            natural_roll,
            natural_roll + math.pi / 2,
            natural_roll - math.pi / 2,
            natural_roll + math.pi,
        ):
            target = self._view_target(roll)
            plan = self.robot.right_plan_path(target.p.tolist() + target.q.tolist())
            if (
                plan["status"] != "Success"
                or self._joint_path_excess(
                    [plan], self._arm_joint_bounds(self.robot.right_arm_joints)
                )
                > 0
            ):
                continue
            self.right_joint_path.append(deepcopy(plan))
            self._sweep_active = True
            try:
                self.take_dense_action(
                    {
                        "left_arm": None,
                        "left_gripper": None,
                        "right_arm": plan,
                        "right_gripper": None,
                    }
                )
                # Demonstrations retain their full pause; evaluation accepts a
                # shorter valid inspection by a learned policy.
                self._hold(max(1.0, self.look_seconds) + 0.15)
            finally:
                self._sweep_active = False
                self._set_planner_fixture(False)
            return self._state.looked and not self._state.failed_reason
        self._set_planner_fixture(False)
        self.plan_success = False
        return False

    def _set_planner_fixture(self, enabled):
        """Check the camera sweep against the housing and the currently open tray."""
        from curobo.geom.types import Cuboid, WorldConfig

        planner = self.robot.right_planner
        if self._planner_original_world is None:
            self._planner_original_world = deepcopy(planner.motion_gen.world_model)
        if not enabled:
            planner.motion_gen.update_world(self._planner_original_world)
            return
        world = deepcopy(self._planner_original_world)
        origin = planner.robot_origion_pose
        bias = sapien.Pose(planner.frame_bias)
        correction = sapien.Pose()
        if "aloha-agilex" in planner.yml_path:
            correction = sapien.Pose(q=[math.cos(-0.01 / 2), 0, 0, math.sin(-0.01 / 2)])
        boxes = list(world.cuboid or [])
        for i, (moving, position, size) in enumerate(self._fixture_boxes):
            root = self.drawer_link.get_pose() if moving else self.fixture_pose
            pose = correction * bias * origin.inv() * root * sapien.Pose(position)
            boxes.append(
                Cuboid(
                    name=f"drawer_{i}",
                    dims=(2 * np.asarray(size)).tolist(),
                    pose=pose.p.tolist() + pose.q.tolist(),
                )
            )
        planner.motion_gen.update_world(WorldConfig(cuboid=boxes))

    def play_once(self):
        right = ArmTag("right")
        grasp = self._handle_pose()
        # Piper cannot fold its wrist back toward its base after the pull.
        approach = (
            np.array([0, 0, 0.10])
            if self._is_piper
            else -0.10 * grasp.to_transformation_matrix()[:3, 0]
        )
        if not self._move(sapien.Pose(grasp.p + approach, grasp.q)) or not self._move(grasp):
            return self._episode_info()
        self.move(self.close_gripper(right))
        if not self._slide(self.travel * 0.92):
            return self._episode_info()
        self.move(self.open_gripper(right))
        # Withdraw the fingers from the bar before the camera sweep. This is
        # handle clearance, followed by one combined move/turn to the view pose.
        released = self.robot.get_right_ee_pose()
        if not self._move(sapien.Pose(np.asarray(released[:3]) + approach, released[3:])):
            return self._episode_info()
        if not self._look():
            self.info["scripted_failure"] = "inspection_not_observed"
            return self._episode_info()
        grasp = self._handle_pose()
        self._set_planner_fixture(True)
        self._sweep_active = True
        try:
            if not self._move(sapien.Pose(grasp.p + approach, grasp.q)):
                return self._episode_info()
        finally:
            self._sweep_active = False
            self._set_planner_fixture(False)
        if not self._move(grasp):
            return self._episode_info()
        self.move(self.close_gripper(right))
        if not self._slide(0.0):
            return self._episode_info()
        self.move(self.open_gripper(right))
        self._move(sapien.Pose(self._handle_pose().p + approach, grasp.q))
        self.move(self.back_to_origin(right))
        self._hold(0.20)
        return self._episode_info()

    def _episode_info(self):
        self.info["task_progress"] = self.get_task_progress()
        return self.info

    def get_task_progress(self):
        return self._state.snapshot() if self._state else {}

    def get_task_metadata(self):
        return {
            "fixture_pose": self.fixture_pose.p.tolist() + self.fixture_pose.q.tolist(),
            "drawer_travel": self.travel,
            "open_fraction": self.open_fraction,
            "hover_height": self.hover_height,
            "look_seconds": self.look_seconds,
            "inspection": self._inspection,
            "camera_motion_contact_impulse": self._sweep_contact_impulse,
            "drawer_position": float(self.drawer.get_qpos()[0]),
        }

    def check_success(self):
        return bool(self._state and self._state.success)


class LookIntoDrawerState:
    """Completion is measured from physics, never advanced by scripted commands."""

    def __init__(self, travel=0.18, open_fraction=0.70, look_seconds=0.25):
        self.open_position = travel * open_fraction
        self.look_seconds = look_seconds
        self.initial_closed = False
        self.opened = self.looked = self.closed = False
        self.look_time = self.settle_time = 0.0
        self.max_opening = 0.0
        self.failed_reason = None
        self.phase = "ready"

    @property
    def success(self):
        return self.phase == "success" and self.failed_reason is None

    def update(self, *, position, dt, held, released, view_valid, arms_ready, left_ready, still):
        if self.failed_reason or self.success:
            return
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not math.isfinite(position):
            self.failed_reason = "nonfinite_drawer_position"
            return
        if not left_ready:
            self.failed_reason = "left_arm_left_ready_pose"
            return
        if not self.initial_closed:
            if abs(position) > 0.003:
                self.failed_reason = "drawer_not_initially_closed"
                return
            self.initial_closed = True
        self.max_opening = max(self.max_opening, position)
        if held and position >= self.open_position:
            self.opened = True
        if not self.opened:
            self.phase = "opening" if held else "ready"
            return
        if not self.looked:
            valid = position >= self.open_position and released and view_valid and still
            self.look_time = self.look_time + dt if valid else 0.0
            self.phase = "looking" if valid else "positioning_camera"
            self.looked = self.look_time + 1e-9 >= self.look_seconds
            return
        # A nearly flush drawer is closed even if the policy leaves a small gap.
        self.closed = self.closed or (held and abs(position) <= 0.01)
        self.phase = "returning" if self.closed else "closing"
        finished = self.closed and abs(position) <= 0.01 and released and arms_ready and still
        self.settle_time = self.settle_time + dt if finished else 0.0
        if self.settle_time + 1e-9 >= 0.12:
            self.phase = "success"

    def snapshot(self):
        return {
            "repeat_count": 1,
            "completed_repeats": int(self.looked),
            "phase": self.phase,
            "initial_closed": self.initial_closed,
            "opened": self.opened,
            "looked": self.looked,
            "closed": self.closed,
            "look_time": self.look_time,
            "max_opening": self.max_opening,
            "success": self.success,
            "failed": self.failed_reason is not None,
            "failed_reason": self.failed_reason,
        }
