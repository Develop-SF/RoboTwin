"""Stir an empty pot with a held rod; environment, script and physical success."""

import json
import math
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import Actor, ArmTag


def surface_clearance(utensil_bounds, pot_bounds):
    """Conservative surface distance between horizontal bounds, including handles."""
    gap = np.maximum(
        np.maximum(utensil_bounds[0] - pot_bounds[1], pot_bounds[0] - utensil_bounds[1]), 0
    )
    return float(np.linalg.norm(gap))


def circle_angles(repeats, seconds_per_cycle, dt):
    """Clockwise circles with smooth acceleration only at the two overall ends."""
    duration = repeats * seconds_per_cycle
    u = np.linspace(0, 1, math.ceil(duration / dt) + 1)
    return -2 * math.pi * repeats * (u - np.sin(2 * math.pi * u) / (2 * math.pi))


class stir_pot(Base_Task):
    """Use the right arm to stir N circles, replace the rod and return to ready."""

    def setup_demo(self, **kwargs):
        self._state = None
        self.repeat_count = kwargs.get("repeat_count", 3)
        if (
            isinstance(self.repeat_count, bool)
            or not isinstance(self.repeat_count, int)
            or self.repeat_count < 1
        ):
            raise ValueError("repeat_count must be a positive integer")
        for name, default, low, high in (
            ("stir_radius", 0.025, 0.015, 0.035),
            ("stir_seconds_per_cycle", 4.0, 3.0, 20.0),
            ("min_utensil_pot_clearance", 0.10, 0.10, math.inf),
        ):
            value = kwargs.get(name, default)
            if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")
            setattr(self, name, float(value))
        for name, default in (
            ("utensil_x_range", [0.25, 0.30]),
            ("utensil_y_range", [-0.12, -0.08]),
            ("utensil_yaw_range", [-0.2, 0.2]),
            ("pot_x_range", [0.06, 0.10]),
            ("pot_y_range", [-0.22, -0.20]),
        ):
            values = np.asarray(kwargs.get(name, default), dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
                raise ValueError(f"{name} must be a finite [minimum, maximum] pair")
            setattr(self, name, values)
        self.tip_offset = 0.09
        self._pot_contact_impulse = 0.0
        self._stir_plan_steps = 0
        self._transport_clearance = 0.13
        super()._init_task_env_(**kwargs)
        # Keep the overhead view in front of Aloha's camera mast.
        for name, camera in zip(self.cameras.static_camera_name, self.cameras.static_camera_list):
            if name == "head_camera":
                position = np.array([0.18, -0.38, 1.60 + self.table_z_bias])
                forward = np.array([0.18, -0.14, 0.90 + self.table_z_bias]) - position
                forward /= np.linalg.norm(forward)
                left = np.array([-1.0, 0.0, 0.0])
                matrix = np.eye(4)
                matrix[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
                matrix[:3, 3] = position
                camera.entity.set_pose(sapien.Pose(matrix))
        self.initial_utensil_pose = self.utensil.get_pose()
        self.initial_clearance = surface_clearance(
            self._utensil_bounds(self.initial_utensil_pose), self._pot_bounds
        )
        if self.initial_clearance + 1e-9 < self.min_utensil_pot_clearance:
            raise ValueError("Utensil settled inside min_utensil_pot_clearance")
        self.robot.set_origin_endpose()
        self._ready_left = np.asarray(self.robot.left_original_pose)
        self._ready_right = np.asarray(self.robot.right_original_pose)
        self._fingers = {joint.child_link.entity for joint, _, _ in self.robot.right_gripper}
        self._fingers.update(
            link.entity
            for link in self.robot.right_entity.get_links()
            if link.get_name() in self.robot.right_fix_gripper_name
        )
        self._robot_links = {
            link.entity
            for robot in (self.robot.left_entity, self.robot.right_entity)
            for link in robot.get_links()
        }
        self._body = self.utensil.actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        self.stir_center = np.array([*self.pot_xy, 0.770 + self.table_z_bias])
        self._state = StirPotState(
            self.repeat_count,
            self.initial_utensil_pose.p,
            self.initial_utensil_pose.q,
            self.stir_center,
            self.stir_radius,
        )
        circles = "circle" if self.repeat_count == 1 else "circles"
        self.set_instruction(
            f"Use the right arm to stir the pot clockwise {self.repeat_count} {circles} with the rod."
        )
        self.info["info"] = {
            "{A}": "stirring rod",
            "{B}": "pot",
            "{a}": "right",
            "{N}": str(self.repeat_count),
        }

    @staticmethod
    def _utensil_bounds(pose):
        extent = np.abs(pose.to_transformation_matrix()[:2, :3]) @ [0.012, 0.012, 0.09]
        return np.array([pose.p[:2] - extent, pose.p[:2] + extent])

    def load_actors(self):
        directory = Path("assets/objects/060_kitchenpot/100015")
        model = json.loads((directory / "model_data.json").read_text())
        body = ET.parse(directory / "mobility.urdf").find("./link[@name='link_1']")
        rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        meshes = {
            element.find("./geometry/mesh").attrib["filename"]
            for element in list(body.findall("visual")) + list(body.findall("collision"))
        }
        vertices = [
            [float(v) for v in line.split()[1:4]]
            for filename in meshes
            for line in (directory / filename).read_text().splitlines()
            if line.startswith("v ")
        ]
        vertices = np.asarray(vertices) @ rotation.T * model["scale"]
        local_bounds = np.array([vertices[:, :2].min(0) - 1e-6, vertices[:, :2].max(0) + 1e-6])
        pose, self.pot_xy = self._sample_layout(local_bounds)
        self._pot_bounds = local_bounds + self.pot_xy
        self.utensil = self._create_rod(pose)
        builder = self.scene.create_actor_builder()
        mesh_pose = sapien.Pose(q=[0.5, 0.5, -0.5, -0.5])
        for visual in body.findall("visual"):
            builder.add_visual_from_file(
                filename=str(directory / visual.find("./geometry/mesh").attrib["filename"]),
                pose=mesh_pose,
                scale=[model["scale"]] * 3,
            )
        for collision in body.findall("collision"):
            builder.add_nonconvex_collision_from_file(
                filename=str(directory / collision.find("./geometry/mesh").attrib["filename"]),
                pose=mesh_pose,
                scale=[model["scale"]] * 3,
            )
        builder.set_initial_pose(
            sapien.Pose(
                [*self.pot_xy, 0.741 + self.table_z_bias + model["transform_matrix"][2][3]]
            )
        )
        self.pot = builder.build_static(name="stir_pot")
        self.add_prohibit_area(self.utensil, padding=0.08)
        self.prohibited_area.append([*(self._pot_bounds[0] - 0.04), *(self._pot_bounds[1] + 0.04)])

    def _create_rod(self, pose):
        # A round handle accepts oblique grasps without the fingers forcibly
        # aligning flat sides and twisting the utensil during closure.
        entity = sapien.Entity()
        entity.set_name("stirring_rod")
        entity.set_pose(sapien.Pose(pose.p + [0, 0, self.table_z_bias], pose.q))
        cylinder_pose = sapien.Pose(q=[math.sqrt(0.5), 0, -math.sqrt(0.5), 0])
        body = sapien.physx.PhysxRigidDynamicComponent()
        collision = sapien.physx.PhysxCollisionShapeCylinder(
            radius=0.012,
            half_length=self.tip_offset,
            material=self.scene.default_physical_material,
        )
        collision.local_pose = cylinder_pose
        body.attach(collision)
        render = sapien.render.RenderBodyComponent()
        visual = sapien.render.RenderShapeCylinder(
            radius=0.012,
            half_length=self.tip_offset,
            material=sapien.render.RenderMaterial(base_color=[0.63, 0.34, 0.12, 1]),
        )
        visual.local_pose = cylinder_pose
        render.attach(visual)
        entity.add_component(body)
        entity.add_component(render)
        self.scene.add_entity(entity)
        return Actor(
            entity,
            {
                "scale": [0.012, 0.012, self.tip_offset],
                "contact_points_pose": [
                    [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.65], [0, 0, 0, 1]],
                    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.65], [0, 0, 0, 1]],
                    [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.65], [0, 0, 0, 1]],
                    [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.65], [0, 0, 0, 1]],
                ],
            },
        )

    def _sample_layout(self, pot_bounds, attempts=1000):
        for _ in range(attempts):
            x, y, yaw = (
                np.random.uniform(*getattr(self, f"utensil_{axis}_range"))
                for axis in ("x", "y", "yaw")
            )
            pot_xy = np.array(
                [np.random.uniform(*self.pot_x_range), np.random.uniform(*self.pot_y_range)]
            )
            pose = sapien.Pose(
                [x, y, 0.741 + self.tip_offset], [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            )
            if (
                surface_clearance(self._utensil_bounds(pose), pot_bounds + pot_xy) + 1e-9
                >= self.min_utensil_pot_clearance
            ):
                return pose, pot_xy
        raise ValueError(
            "Object ranges cannot provide min_utensil_pot_clearance; move the utensil farther from the pot"
        )

    @staticmethod
    def _ready(current, target):
        return np.linalg.norm(np.asarray(current[:3]) - target[:3]) <= 0.02 and abs(
            np.dot(current[3:], target[3:])
        ) >= math.cos(0.15 / 2)

    def _contacts(self):
        held = table = False
        for contact in self.scene.get_contacts():
            entities = {body.entity for body in contact.bodies}
            impulse = sum(float(np.linalg.norm(p.impulse)) for p in contact.points)
            if self.utensil.actor in entities and contact.points:
                held |= bool(entities & self._fingers)
                table |= self.table in entities
            if self.pot in entities and (
                entities & self._robot_links or self.utensil.actor in entities
            ):
                self._pot_contact_impulse = max(self._pot_contact_impulse, impulse)
                if impulse > 1e-5:
                    self._state.fail("pot_collision")
        return held, table

    def _after_simulation_step(self):
        if self._state is None:
            return
        pose = self.utensil.get_pose()
        held, table = self._contacts()
        self._state.update(
            pose.p,
            pose.q,
            (pose * sapien.Pose([0, 0, -self.tip_offset])).p,
            dt=self.scene.get_timestep(),
            held=held and self.is_right_gripper_close(),
            closed=self.is_right_gripper_close(),
            opened=self.is_right_gripper_open(),
            table_contact=table,
            left_ready=self._ready(self.robot.get_left_ee_pose(), self._ready_left),
            arms_ready=self._ready(self.robot.get_right_ee_pose(), self._ready_right),
            still=np.linalg.norm(self._body.linear_velocity) < 0.025
            and np.linalg.norm(self._body.angular_velocity) < 0.25,
        )

    def _hold(self, seconds=0.16):
        for _ in range(math.ceil(seconds / self.scene.get_timestep())):
            self._step_simulation()

    def _execute(self, plan):
        bounds = self._arm_joint_bounds(self.robot.right_arm_joints)
        if plan["status"] != "Success" or self._joint_path_excess([plan], bounds) > 0:
            self.info["scripted_failure"] = "invalid_joint_path"
            if plan["status"] == "Success":
                self.info["path_failure"] = {
                    "minimum": np.asarray(plan["position"]).min(0).tolist(),
                    "maximum": np.asarray(plan["position"]).max(0).tolist(),
                    "bounds": np.asarray(bounds).tolist(),
                }
            self.plan_success = False
            return False
        self.right_joint_path.append(deepcopy(plan))
        self.take_dense_action(
            {"left_arm": None, "left_gripper": None, "right_arm": plan, "right_gripper": None}
        )
        return self.plan_success and not self._state.failed_reason

    def _move_object(self, position):
        return self._move_cartesian(position, seconds=2.0)

    def _cartesian_path(self, positions, quaternions=None):
        """Seed each IK solution from its predecessor; execute one continuous path.

        Use the articulation's own kinematics and measured grasp, so no embodiment
        specific camera/tool offset or planner frame correction is assumed.
        """
        from scipy.interpolate import CubicSpline

        if quaternions is None:
            quaternions = np.broadcast_to(self.initial_utensil_pose.q, (len(positions), 4))
        entity = self.robot.right_entity
        link = self.robot.right_ee.child_link
        link_index = entity.get_links().index(link)
        model = entity.create_pinocchio_model()
        active = entity.get_active_joints()
        indices = [active.index(joint) for joint in self.robot.right_arm_joints]
        mask = np.zeros(entity.dof, dtype=np.int32)
        mask[indices] = 1
        qpos = entity.get_qpos().astype(float)
        grasp = self.utensil.get_pose().inv() * link.get_pose()
        base_inv = entity.get_pose().inv()
        # Solve at 25 Hz; interpolate joints to the 250 Hz physics clock. Dense
        # sampling and branch continuity bound interpolation error on the circle.
        samples = np.unique(np.r_[np.arange(0, len(positions), 10), len(positions) - 1])
        joint_path = []
        for index in samples:
            target = base_inv * sapien.Pose(positions[index], quaternions[index]) * grasp
            solved, success, error = model.compute_inverse_kinematics(
                link_index, target, qpos, mask, eps=1e-5, max_iterations=200
            )
            if not success or np.max(np.abs(solved[indices] - qpos[indices])) > 0.12:
                self.info["scripted_failure"] = "continuous_path_ik_failed"
                self.info["path_failure"] = {
                    "sample": int(index),
                    "ik_converged": bool(success),
                    "max_joint_step": float(np.max(np.abs(solved[indices] - qpos[indices]))),
                    "ik_error": float(np.linalg.norm(error)),
                }
                self.plan_success = False
                return False
            qpos = solved
            joint_path.append(qpos[indices].copy())
        dt = self.scene.get_timestep()
        spline = CubicSpline(
            samples * dt,
            joint_path,
            bc_type=((1, np.zeros(len(indices))), (1, np.zeros(len(indices)))),
        )
        times = np.arange(len(positions)) * dt
        plan = {"status": "Success", "position": spline(times), "velocity": spline(times, 1)}
        return self._execute(plan)

    def _move_cartesian(self, position, seconds=1.5, restore_orientation=True):
        from scipy.spatial.transform import Rotation, Slerp

        current = self.utensil.get_pose()
        start = current.p
        u = np.linspace(0, 1, math.ceil(seconds / self.scene.get_timestep()) + 1)
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        # Correct small compliant-grasp drift over the whole move instead of
        # snapping the first IK sample back to the requested orientation.
        target_q = self.initial_utensil_pose.q if restore_orientation else current.q
        endpoints = np.array([current.q, target_q])[:, [1, 2, 3, 0]]
        quaternions = Slerp([0, 1], Rotation.from_quat(endpoints))(blend).as_quat()[
            :, [3, 0, 1, 2]
        ]
        return self._cartesian_path(
            start + blend[:, None] * (np.asarray(position) - start), quaternions
        )

    def _seat_for_release(self):
        """Release only a flat, settled rod, with a small seating correction if needed."""
        for attempt in range(3):
            if attempt and not self._move_cartesian(self.initial_utensil_pose.p - [0, 0, 0.0005]):
                return False
            settled = 0.0
            for _ in range(math.ceil(0.5 / self.scene.get_timestep())):
                pose = self.utensil.get_pose()
                stable = (
                    pose.to_transformation_matrix()[2, 2] >= math.cos(math.radians(1.5))
                    and np.linalg.norm(self._body.linear_velocity) < 0.005
                    and np.linalg.norm(self._body.angular_velocity) < 0.05
                    and self._contacts()[1]
                )
                settled = settled + self.scene.get_timestep() if stable else 0
                if settled >= 0.08 and self._state.replaced:
                    self.info["release_pose"] = pose.p.tolist() + pose.q.tolist()
                    return True
                self._step_simulation()
                if self._state.failed_reason:
                    return False
        self.info["scripted_failure"] = "placement_not_settled"
        return False

    def _return_ready(self):
        """Plan the free hand around the replaced rod and the complete pot."""
        from curobo.geom.types import Cuboid, WorldConfig

        planner = self.robot.right_planner
        original = deepcopy(planner.motion_gen.world_model)
        correction = sapien.Pose()
        if "aloha-agilex" in planner.yml_path:
            correction = sapien.Pose(q=[math.cos(-0.01 / 2), 0, 0, math.sin(-0.01 / 2)])
        to_planner = (
            correction * sapien.Pose(planner.frame_bias) * planner.robot_origion_pose.inv()
        )
        pot_pose = sapien.Pose([*self._pot_bounds.mean(0), 0.781 + self.table_z_bias])
        boxes = list(original.cuboid or [])
        for name, world_pose, dimensions in (
            ("placed_rod", self.utensil.get_pose(), [0.036, 0.036, 0.20]),
            ("stir_pot", pot_pose, [*(self._pot_bounds[1] - self._pot_bounds[0] + 0.01), 0.09]),
        ):
            pose = to_planner * world_pose
            boxes.append(
                Cuboid(name=name, dims=list(dimensions), pose=pose.p.tolist() + pose.q.tolist())
            )
        planner.motion_gen.update_world(WorldConfig(cuboid=boxes))
        try:
            self.move(self.back_to_origin(ArmTag("right")))
        finally:
            planner.motion_gen.update_world(original)

    def choose_best_pose(self, res_pose, center_pose, arm_tag=None):
        if arm_tag != "right":
            return super().choose_best_pose(res_pose, center_pose, arm_tag)
        rotation = self.initial_utensil_pose.to_transformation_matrix()[:3, :3]
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        # The round rod permits oblique grasps. Keep these directions in
        # world coordinates so a small randomized yaw cannot exhaust wrist travel.
        for desired_yaw in (yaw, 0.20, 0.40, 0.60):
            angle = desired_yaw - yaw
            around_contact = (
                sapien.Pose(center_pose[:3])
                * sapien.Pose(q=[math.cos(angle / 2), 0, 0, math.sin(angle / 2)])
                * sapien.Pose(-np.asarray(center_pose[:3]))
            )
            origin = around_contact * sapien.Pose(res_pose[:3], res_pose[3:])
            center = around_contact * sapien.Pose(center_pose[:3], center_pose[3:])
            for clearance in (0.13, 0.09):
                self._transport_clearance = clearance
                pose = self._choose_stir_grasp(
                    origin.p.tolist() + origin.q.tolist(),
                    center.p.tolist() + center.q.tolist(),
                    arm_tag,
                )
                if pose is not None:
                    self.info["grasp_yaw"] = desired_yaw
                    return pose
        return None

    def _choose_stir_grasp(self, res_pose, center_pose, arm_tag):
        """Require room for the insertion and full circle before selecting a grasp."""
        if arm_tag != "right":
            return super().choose_best_pose(res_pose, center_pose, arm_tag)
        candidates = self.robot.create_target_pose_list(res_pose, center_pose, arm_tag)
        plans = self.robot.right_plan_multi_path(candidates)
        entity = self.robot.right_entity
        link = self.robot.right_ee.child_link
        model = entity.create_pinocchio_model()
        link_index = entity.get_links().index(link)
        active = entity.get_active_joints()
        indices = [active.index(joint) for joint in self.robot.right_arm_joints]
        mask = np.zeros(entity.dof, dtype=np.int32)
        mask[indices] = 1
        ee = self.robot.get_right_ee_pose()
        ee_to_link = sapien.Pose(ee[:3], ee[3:]).inv() * link.get_pose()
        base_inv = entity.get_pose().inv()
        bounds = self._arm_joint_bounds(self.robot.right_arm_joints)
        anchor = self.stir_center + [self.stir_radius, 0, self.tip_offset]
        waypoints = [
            self.initial_utensil_pose.p,
            self.initial_utensil_pose.p + [0, 0, self._transport_clearance + 0.04],
            anchor + [0, 0, self._transport_clearance],
            anchor,
        ]
        positions = np.concatenate(
            [np.linspace(a, b, 12) for a, b in zip(waypoints, waypoints[1:])]
        )
        angles = np.linspace(0, -2 * math.pi, 49)
        circle = self.stir_center + np.column_stack(
            [
                self.stir_radius * np.cos(angles),
                self.stir_radius * np.sin(angles),
                np.full(len(angles), self.tip_offset),
            ]
        )
        positions = np.concatenate([positions, circle])
        for index, candidate in enumerate(candidates):
            if (
                plans["status"][index] != "Success"
                # A horizontal grasp leaves the palm below the protruding rod
                # tip. Keep it above the tip so withdrawal cannot scoop it up.
                or candidate[2] < self.initial_utensil_pose.p[2] + self.tip_offset + 0.02
                or self._joint_path_excess([{"position": plans["position"][index]}], bounds) > 0
            ):
                continue
            grasp = (
                self.initial_utensil_pose.inv()
                * sapien.Pose(candidate[:3], candidate[3:])
                * ee_to_link
            )
            qpos = entity.get_qpos().astype(float)
            qpos[indices] = plans["position"][index, -1]
            valid = True
            for position in positions:
                target = base_inv * sapien.Pose(position, self.initial_utensil_pose.q) * grasp
                solved, success, _ = model.compute_inverse_kinematics(
                    link_index, target, qpos, mask, eps=1e-5, max_iterations=200
                )
                joints = solved[indices]
                # Leave a small reserve for physical grasp compliance.
                if (
                    not success
                    or np.any(joints < bounds[:, 0] + 0.04)
                    or np.any(joints > bounds[:, 1] - 0.04)
                ):
                    valid = False
                    break
                qpos = solved
            if valid:
                self.info["grasp_candidate"] = index
                return candidate
        return None

    def play_once(self):
        right = ArmTag("right")
        grasp = None
        for contact_id in range(4):
            grasp = self.get_grasp_pose(
                self.utensil, arm_tag=right, pre_dis=0, contact_point_id=contact_id
            )
            if grasp is not None:
                self.info["grasp_face"] = contact_id
                break
        if grasp is None:
            self.plan_success = False
            self.info["scripted_failure"] = "no_reachable_grasp"
            return self._episode_info()
        approach = np.asarray(grasp).copy()
        approach[2] += 0.08
        self.move(self.move_to_pose(right, approach))
        self.move(self.move_to_pose(right, grasp))
        self.move(self.close_gripper(right))
        if not self.plan_success or not self._contacts()[0]:
            self.info["scripted_failure"] = "grasp_not_observed"
            return self._episode_info()
        lifted = self.initial_utensil_pose.p + [0, 0, self._transport_clearance + 0.04]
        anchor = self.stir_center + [self.stir_radius, 0, self.tip_offset]
        above = anchor + [0, 0, self._transport_clearance]
        if (
            not self._move_cartesian(lifted)
            or not self._move_object(above)
            or not self._move_cartesian(anchor)
        ):
            return self._episode_info()
        self._hold()
        if not self._state.armed:
            self.info["scripted_failure"] = "insertion_not_observed"
            return self._episode_info()
        tip = (self.utensil.get_pose() * sapien.Pose([0, 0, -self.tip_offset])).p
        start_angle = math.atan2(*(tip[:2] - self.stir_center[:2])[::-1])
        angles = start_angle + circle_angles(
            self.repeat_count, self.stir_seconds_per_cycle, self.scene.get_timestep()
        )
        positions = self.stir_center + np.column_stack(
            [
                self.stir_radius * np.cos(angles),
                self.stir_radius * np.sin(angles),
                np.full(len(angles), self.tip_offset),
            ]
        )
        self._stir_plan_steps = len(positions)
        if not self._cartesian_path(positions):
            return self._episode_info()
        self._hold()
        if self._state.completed_repeats != self.repeat_count:
            self.info["scripted_failure"] = "circles_not_observed"
            return self._episode_info()
        if not self._move_cartesian(
            self.utensil.get_pose().p + [0, 0, self._transport_clearance]
        ) or not self._move_object(lifted):
            return self._episode_info()
        if not self._move_cartesian(self.initial_utensil_pose.p):
            return self._episode_info()
        self._hold()
        if not self._seat_for_release():
            self.info.setdefault("scripted_failure", "replacement_not_observed")
            return self._episode_info()
        self.move(self.open_gripper(right))
        ee = self.robot.get_right_ee_pose()
        forward = sapien.Pose(ee[:3], ee[3:]).to_transformation_matrix()[:3, 0]
        # With a shallow grasp the rod protrudes above the palm. Pull the open
        # palm clear along its approach axis before lifting, or it can catch the
        # top of the rod even though both fingertips have already released it.
        if forward[2] > -0.5 and not self._move_cartesian(
            self.utensil.get_pose().p - 0.03 * forward, restore_orientation=False
        ):
            return self._episode_info()
        if not self._move_cartesian(
            self.utensil.get_pose().p + [0, 0, 0.12], restore_orientation=False
        ):
            return self._episode_info()
        self._return_ready()
        self._hold()
        return self._episode_info()

    def _episode_info(self):
        self.info.update(
            task_progress=self.get_task_progress(), instruction=self.get_instruction()
        )
        return self.info

    def get_task_progress(self):
        return {} if self._state is None else self._state.snapshot()

    def get_task_metadata(self):
        final = self.utensil.get_pose()
        return {
            "initial_utensil_pose": self.initial_utensil_pose.p.tolist()
            + self.initial_utensil_pose.q.tolist(),
            "final_utensil_pose": final.p.tolist() + final.q.tolist(),
            "pot_xy": self.pot_xy.tolist(),
            "pot_footprint_bounds": self._pot_bounds.tolist(),
            "min_utensil_pot_clearance": self.min_utensil_pot_clearance,
            "initial_utensil_pot_clearance": self.initial_clearance,
            "pot_contact_impulse": self._pot_contact_impulse,
            "stir_center": self.stir_center.tolist(),
            "stir_radius": self.stir_radius,
            "stir_seconds_per_cycle": self.stir_seconds_per_cycle,
            "stir_plan_steps": self._stir_plan_steps,
            "transport_clearance": self._transport_clearance,
            "grasp_candidate": self.info.get("grasp_candidate"),
            "grasp_face": self.info.get("grasp_face"),
            "grasp_yaw": self.info.get("grasp_yaw"),
            "release_pose": self.info.get("release_pose"),
            "scripted_failure": self.info.get("scripted_failure"),
            "path_failure": self.info.get("path_failure"),
        }

    def check_success(self):
        return self._state is not None and self._state.success


class StirPotState:
    """Count physical clockwise revolutions at fixed depth/orientation, then restore."""

    def __init__(self, repeats, initial_position, initial_quaternion, center, radius):
        self.repeat_count = repeats
        self.initial_position = np.asarray(initial_position, dtype=float)
        self.initial_quaternion = np.asarray(initial_quaternion, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.completed_repeats = 0
        self.clockwise_angle = 0.0
        self.last_angle = None
        self.armed = self.grasped = self.replaced = self.released = False
        self.missing_contact_time = self.settle_time = self.insertion_time = 0.0
        self.phase = "ready"
        self.failed_reason = None

    @property
    def success(self):
        return self.phase == "success"

    def fail(self, reason):
        if self.failed_reason is None:
            self.failed_reason = reason
        self.phase = "failed"

    def update(
        self,
        position,
        quaternion,
        tip,
        *,
        dt,
        held,
        closed,
        opened,
        table_contact,
        left_ready,
        arms_ready,
        still,
    ):
        if self.failed_reason or self.success:
            return
        position, quaternion, tip = (
            np.asarray(v, dtype=float) for v in (position, quaternion, tip)
        )
        if (
            any(
                v.shape != (size,) or not np.isfinite(v).all()
                for v, size in zip((position, quaternion, tip), (3, 4, 3))
            )
            or np.linalg.norm(quaternion) < 1e-9
        ):
            self.fail("invalid_utensil_pose")
            return
        if not left_ready:
            self.fail("left_arm_left_ready_pose")
            return
        orientation_ok = abs(
            np.dot(quaternion / np.linalg.norm(quaternion), self.initial_quaternion)
        ) >= math.cos(0.12 / 2)
        at_start = (
            np.linalg.norm(position - self.initial_position) <= 0.015
            and orientation_ok
            and table_contact
        )
        self.grasped |= held
        if self.grasped and not self.released:
            self.missing_contact_time = 0 if held else self.missing_contact_time + dt
            if not closed:
                if (
                    self.completed_repeats != self.repeat_count
                    or not self.replaced
                    or not at_start
                ):
                    self.fail("released_before_replacement")
                    return
                self.released = True
            elif self.missing_contact_time > 0.12:
                self.fail("lost_right_hand_grasp")
                return
        radial = tip[:2] - self.center[:2]
        in_circle = (
            abs(np.linalg.norm(radial) - self.radius) <= 0.007
            and abs(tip[2] - self.center[2]) <= 0.008
            and orientation_ok
        )
        if not self.armed:
            self.phase = "inserting" if self.grasped else "ready"
            self.insertion_time = self.insertion_time + dt if in_circle and held else 0
            if self.insertion_time >= 0.04:
                self.armed = True
                self.last_angle = math.atan2(radial[1], radial[0])
        elif in_circle and held:
            angle = math.atan2(radial[1], radial[0])
            delta = math.atan2(
                math.sin(self.last_angle - angle), math.cos(self.last_angle - angle)
            )
            if abs(delta) > 0.15:
                self.fail("discontinuous_stir_motion")
                return
            self.clockwise_angle += delta
            self.last_angle = angle
            if self.clockwise_angle < -0.10:
                self.fail("wrong_stir_direction")
                return
            self.completed_repeats = max(
                self.completed_repeats, int((self.clockwise_angle + 0.04) / (2 * math.pi))
            )
            if self.clockwise_angle > 2 * math.pi * self.repeat_count + 0.15:
                self.fail("extra_stir_after_requested_repeats")
                return
            self.phase = "stirring"
        elif self.armed and self.completed_repeats < self.repeat_count:
            self.fail("left_stir_circle_before_completion")
            return
        if self.completed_repeats == self.repeat_count:
            self.phase = "returning"
            self.replaced |= bool(at_start and held)
            finished = self.released and at_start and opened and not held and arms_ready and still
            self.settle_time = self.settle_time + dt if finished else 0
            if self.settle_time >= 0.12:
                self.phase = "success"

    def snapshot(self):
        return {
            "phase": self.phase,
            "repeat_count": self.repeat_count,
            "completed_repeats": self.completed_repeats,
            "clockwise_angle": self.clockwise_angle,
            "armed": self.armed,
            "replaced": self.replaced,
            "released": self.released,
            "success": self.success,
            "failed_reason": self.failed_reason,
        }
