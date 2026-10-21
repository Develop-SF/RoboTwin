"""Salt-shaking environment, right-arm script, and physical completion tracking."""

import json
import math
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import ArmTag, create_box


class shake_salt(Base_Task):
    """Aim a salt shaker into a pot, shake N times, and restore its original pose."""

    def setup_demo(self, **kwargs):
        self._state = None
        self._pickup_complete = False
        self._pickup_pot_contact_impulse = 0.0
        self._turn_in_progress = False
        self._turn_pot_contact_impulse = 0.0
        self.repeat_count = kwargs.get("repeat_count", 3)
        self.shake_distance = float(kwargs.get("shake_distance", 0.05))
        clearance = kwargs.get("min_shaker_pot_clearance", 0.10)
        if isinstance(clearance, bool) or not math.isfinite(clearance) or clearance <= 0:
            raise ValueError("min_shaker_pot_clearance must be positive and finite")
        self.min_shaker_pot_clearance = float(clearance)
        if (
            isinstance(self.repeat_count, bool)
            or not isinstance(self.repeat_count, int)
            or self.repeat_count < 1
        ):
            raise ValueError("repeat_count must be a positive integer")
        if not math.isfinite(self.shake_distance) or not 0 < self.shake_distance <= 0.08:
            raise ValueError("shake_distance must be positive and at most 0.08 meters")
        for name, default in (
            ("shaker_x_range", [0.25, 0.30]),
            ("shaker_y_range", [-0.12, -0.08]),
            ("shaker_yaw_range", [-0.2, 0.2]),
            ("pot_x_range", [0.06, 0.10]),
            ("pot_y_range", [-0.22, -0.20]),
        ):
            values = kwargs.get(name, default)
            if (
                len(values) != 2
                or not all(math.isfinite(v) for v in values)
                or values[0] > values[1]
            ):
                raise ValueError(f"{name} must be a finite [minimum, maximum] pair")
            setattr(self, name, values)
        self.settle_seconds = 0.20
        self.tip_offset = 0.052
        self._tilt_roll_degrees = []
        embodiment = kwargs.get("embodiment_name", str(kwargs.get("embodiment", "")))
        # Piper's larger wrist needs a higher side contact to clear the pot rim;
        # Aloha's slender fingers hold the middle of the body more securely.
        self._grasp_height = 0.8 if "piper" in embodiment.lower() else 0.4
        super()._init_task_env_(**kwargs)
        # Put the camera in front of Aloha's camera mast instead of behind it.
        # The same fixed view includes the pot and pickup area for both embodiments.
        for name, camera in zip(self.cameras.static_camera_name, self.cameras.static_camera_list):
            if name == "head_camera":
                position = np.array([0.18, -0.38, 1.60 + self.table_z_bias])
                target = np.array([0.18, -0.14, 0.90 + self.table_z_bias])
                forward = target - position
                forward /= np.linalg.norm(forward)
                left = np.array([-1.0, 0.0, 0.0])
                pose = np.eye(4)
                pose[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
                pose[:3, 3] = position
                camera.entity.set_pose(sapien.Pose(pose))
        self.initial_shaker_pose = self.shaker.get_pose()
        self.initial_shaker_pot_clearance = self._horizontal_clearance(
            self._shaker_footprint_bounds(
                self.initial_shaker_pose.p,
                self.initial_shaker_pose.to_transformation_matrix()[:3, :3],
            ),
            self._pot_local_bounds + self.pot_center[:2],
        )
        if self.initial_shaker_pot_clearance + 1e-9 < self.min_shaker_pot_clearance:
            raise ValueError("Shaker settled closer to the pot than min_shaker_pot_clearance")
        self.robot.set_origin_endpose()
        self._ready_left_pose = np.asarray(self.robot.left_original_pose, dtype=float)
        self._ready_right_pose = np.asarray(self.robot.right_original_pose, dtype=float)
        self._right_gripper_entities = [
            joint.child_link.entity for joint, _, _ in self.robot.right_gripper
        ]
        self._right_gripper_entities.extend(
            link.entity
            for link in self.robot.right_entity.get_links()
            if link.get_name() in self.robot.right_fix_gripper_name
        )
        self._robot_link_entities = [
            link.entity
            for robot in (self.robot.left_entity, self.robot.right_entity)
            for link in robot.get_links()
        ]
        self._shaker_body = self.shaker.actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        # A fixed approach direction keeps the tilted wrist reachable across
        # randomized object poses and embodiments. Its ray still hits the pot center.
        self.shake_axis = np.array([-0.5, -0.5, -math.sqrt(0.5)])
        self.shake_origin = self.pot_center - 0.20 * self.shake_axis
        # Turning can sweep the wrist below its endpoint. Rotate above the pot,
        # then approach the shaking origin with the final orientation held fixed.
        self.turn_clearance = 0.12
        self.turn_position = self.shake_origin + [0, 0, self.turn_clearance]
        self._rotation_axis = np.array([math.sqrt(0.5), -math.sqrt(0.5), 0.0])
        self.shake_quaternion = self._tilted_quaternion(3 * math.pi / 4)
        self._state = ShakeSaltState(
            self.repeat_count,
            self.initial_shaker_pose.p,
            self.initial_shaker_pose.q,
            self.shake_origin,
            self.shake_axis,
            self.pot_center,
            shake_distance=self.shake_distance,
        )
        times = "time" if self.repeat_count == 1 else "times"
        self.set_instruction(
            f"Use the right arm to shake the salt into the pot {self.repeat_count} {times}."
        )
        self.info["info"] = {
            "{A}": "salt shaker",
            "{B}": "pot",
            "{a}": "right",
            "{N}": str(self.repeat_count),
        }
        self.info["task_progress"] = self.get_task_progress()

    def load_actors(self):
        self._pot_local_bounds = self._pot_footprint_bounds()
        x, y, yaw, pot_x, pot_y = self._sample_object_positions()
        self.shaker = create_box(
            scene=self,
            pose=sapien.Pose([x, y, 0.781], [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]),
            half_size=[0.020, 0.020, 0.040],
            color=(0.94, 0.92, 0.80),
            name="salt_shaker",
            boxtype="side",
        )
        # Apply the embodiment's side-contact height before selecting its grasp.
        for contact in self.shaker.config["contact_points_pose"]:
            contact[2][3] = self._grasp_height
        render = self.shaker.actor.find_component_by_type(sapien.render.RenderBodyComponent)
        # SAPIEN only permits adding shapes while the render body is detached.
        self.shaker.actor.remove_component(render)
        silver = sapien.render.RenderMaterial(
            base_color=[0.72, 0.75, 0.80, 1], metallic=0.7, roughness=0.25
        )
        cap = sapien.render.RenderShapeCylinder(radius=0.023, half_length=0.006, material=silver)
        # SAPIEN cylinders are along local X; rotate the cap to the shaker's +Z axis.
        cap.local_pose = sapien.Pose([0, 0, 0.046], [math.sqrt(0.5), 0, -math.sqrt(0.5), 0])
        render.attach(cap)
        dark = sapien.render.RenderMaterial(base_color=[0.06, 0.06, 0.07, 1])
        for hx, hy in ((0, 0), (0.009, 0), (-0.009, 0), (0, 0.009), (0, -0.009)):
            hole = sapien.render.RenderShapeSphere(radius=0.002, material=dark)
            hole.local_pose = sapien.Pose([hx, hy, 0.0515])
            render.attach(hole)
        # A dark blue band distinguishes the ivory salt-filled body from the lid.
        band = sapien.render.RenderShapeBox(
            [0.0205, 0.0205, 0.004],
            sapien.render.RenderMaterial(base_color=[0.10, 0.25, 0.45, 1]),
        )
        band.local_pose = sapien.Pose([0, 0, -0.024])
        render.attach(band)
        self.shaker.actor.add_component(render)
        self.pot = self._create_open_pot(pot_x, pot_y)
        self.add_prohibit_area(self.shaker, padding=0.08)
        self.prohibited_area.append(
            [
                self.pot_center[0] - 0.15,
                self.pot_center[1] - 0.15,
                self.pot_center[0] + 0.15,
                self.pot_center[1] + 0.15,
            ]
        )

    @staticmethod
    def _pot_footprint_bounds(directory=Path("assets/objects/060_kitchenpot/100015")):
        """Bound the actual body and handle meshes after their fixed rotation/scale."""
        model = json.loads((directory / "model_data.json").read_text())
        body = ET.parse(directory / "mobility.urdf").getroot().find("./link[@name='link_1']")
        # This is the same [0.5, 0.5, -0.5, -0.5] rotation used by _create_open_pot.
        rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        meshes = {
            element.find("./geometry/mesh").attrib["filename"]
            for element in list(body.findall("visual")) + list(body.findall("collision"))
        }
        vertices = []
        for filename in sorted(meshes):
            for line in (directory / filename).read_text().splitlines():
                fields = line.split()
                if fields and fields[0] == "v":
                    vertices.append([float(value) for value in fields[1:4]])
        transformed = np.asarray(vertices) @ rotation.T * model["scale"]
        # Round outward by a micron so mesh import rounding cannot reduce clearance.
        return np.array(
            [transformed[:, :2].min(axis=0) - 1e-6, transformed[:, :2].max(axis=0) + 1e-6]
        )

    @staticmethod
    def _shaker_footprint_bounds(position, rotation):
        """Include the body, band, circular cap and holes in a horizontal AABB."""
        position = np.asarray(position)[:2]
        rotation = np.asarray(rotation)[:2]
        bounds = []
        for center, half_size in (
            ([0, 0, 0], [0.020, 0.020, 0.040]),
            ([0, 0, -0.024], [0.0205, 0.0205, 0.004]),
        ):
            center = position + rotation @ center
            extent = np.abs(rotation) @ half_size
            bounds.extend((center - extent, center + extent))
        center = position + rotation @ [0, 0, 0.046]
        extent = 0.023 * np.linalg.norm(rotation[:, :2], axis=1) + 0.006 * np.abs(rotation[:, 2])
        bounds.extend((center - extent, center + extent))
        for hx, hy in ((0, 0), (0.009, 0), (-0.009, 0), (0, 0.009), (0, -0.009)):
            center = position + rotation @ [hx, hy, 0.0515]
            bounds.extend((center - 0.002, center + 0.002))
        return np.array([np.min(bounds, axis=0), np.max(bounds, axis=0)])

    @staticmethod
    def _horizontal_clearance(shaker_bounds, pot_bounds):
        """Conservative horizontal surface gap, including the pot handles."""
        gap = np.maximum(
            np.maximum(shaker_bounds[0] - pot_bounds[1], pot_bounds[0] - shaker_bounds[1]), 0
        )
        return float(np.linalg.norm(gap))

    def _sample_object_positions(self, max_attempts=1000):
        """Sample both poses jointly; reject crowded layouts before creating actors."""
        # The minimum yaw-dependent shaker extent gives the maximum possible gap.
        low, high = self.shaker_yaw_range
        side = min(abs(math.cos(yaw)) + abs(math.sin(yaw)) for yaw in (low, high))
        if math.ceil(low / (math.pi / 2)) <= math.floor(high / (math.pi / 2)):
            side = 1.0
        extent = max(0.023, 0.0205 * side)
        relative_min = np.array(
            [
                self.shaker_x_range[0] - self.pot_x_range[1],
                self.shaker_y_range[0] - self.pot_y_range[1],
            ]
        )
        relative_max = np.array(
            [
                self.shaker_x_range[1] - self.pot_x_range[0],
                self.shaker_y_range[1] - self.pot_y_range[0],
            ]
        )
        maximum_gap = np.linalg.norm(
            np.maximum(
                np.maximum(
                    relative_max - extent - self._pot_local_bounds[1],
                    self._pot_local_bounds[0] - relative_min - extent,
                ),
                0,
            )
        )
        if maximum_gap + 1e-9 < self.min_shaker_pot_clearance:
            raise ValueError(
                "Object pose ranges cannot satisfy min_shaker_pot_clearance="
                f"{self.min_shaker_pot_clearance:g} m; maximum possible surface gap is "
                f"{maximum_gap:g} m. Move the shaker range farther from the pot."
            )
        for _ in range(max_attempts):
            x = np.random.uniform(*self.shaker_x_range)
            y = np.random.uniform(*self.shaker_y_range)
            yaw = np.random.uniform(*self.shaker_yaw_range)
            pot_x = np.random.uniform(*self.pot_x_range)
            pot_y = np.random.uniform(*self.pot_y_range)
            rotation = np.array(
                [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
            )
            clearance = self._horizontal_clearance(
                self._shaker_footprint_bounds([x, y], rotation),
                self._pot_local_bounds + [pot_x, pot_y],
            )
            if clearance + 1e-9 >= self.min_shaker_pot_clearance:
                return x, y, yaw, pot_x, pot_y
        raise ValueError(
            f"Could not sample min_shaker_pot_clearance={self.min_shaker_pot_clearance:g} m "
            f"within {max_attempts} attempts; widen or move the object pose ranges."
        )

    def _create_open_pot(self, x, y):
        """Reuse the existing kitchen-pot body as a static, lid-free receptacle."""
        directory = Path("assets/objects/060_kitchenpot/100015")
        model = json.loads((directory / "model_data.json").read_text())
        urdf = ET.parse(directory / "mobility.urdf").getroot()
        body = urdf.find("./link[@name='link_1']")
        scale = [model["scale"]] * 3
        # The source URDF rotates the Y-up mesh through its fixed body joint.
        mesh_pose = sapien.Pose(q=[0.5, 0.5, -0.5, -0.5])
        builder = self.scene.create_actor_builder()
        for visual in body.findall("visual"):
            mesh = directory / visual.find("./geometry/mesh").attrib["filename"]
            builder.add_visual_from_file(filename=str(mesh), pose=mesh_pose, scale=scale)
        for collision in body.findall("collision"):
            mesh = directory / collision.find("./geometry/mesh").attrib["filename"]
            builder.add_nonconvex_collision_from_file(
                filename=str(mesh), pose=mesh_pose, scale=scale
            )
        base_z = 0.741 + self.table_z_bias + model["transform_matrix"][2][3]
        builder.set_initial_pose(sapien.Pose([x, y, base_z]))
        self.pot_center = np.array([x, y, 0.80 + self.table_z_bias], dtype=float)
        return builder.build_static(name="salt_pot")

    def _tilted_quaternion(self, angle, roll_degrees=100):
        if angle == 0:
            return self.initial_shaker_pose.q
        rotation = sapien.Pose(
            q=[math.cos(angle / 2), *(math.sin(angle / 2) * self._rotation_axis)]
        )
        # The axial roll changes the wrist posture without changing the cap axis.
        roll = math.radians(roll_degrees)
        return (rotation * sapien.Pose(q=[math.cos(roll / 2), 0, 0, math.sin(roll / 2)])).q

    def _contact_flags(self):
        right_contact = table_contact = False
        for contact in self.scene.get_contacts():
            entities = [body.entity for body in contact.bodies]
            if self.shaker.actor not in entities or not contact.points:
                continue
            other = entities[1] if entities[0] == self.shaker.actor else entities[0]
            right_contact |= other in self._right_gripper_entities
            table_contact |= other == self.table
        return right_contact, table_contact

    @staticmethod
    def _pose_is_ready(current, target):
        current = np.asarray(current, dtype=float)
        cosine = abs(np.dot(current[3:], target[3:])) / (
            np.linalg.norm(current[3:]) * np.linalg.norm(target[3:])
        )
        return (
            np.linalg.norm(current[:3] - target[:3]) <= 0.02
            and 2 * math.acos(np.clip(cosine, 0, 1)) <= 0.15
        )

    def _after_simulation_step(self):
        if self._state is None:
            return
        pose = self.shaker.get_pose()
        axis = pose.to_transformation_matrix()[:3, 2]
        right_contact, table_contact = self._contact_flags()
        self._track_pickup_contacts(pose.p, right_contact, self.is_right_gripper_close())
        self._track_turn_contacts()
        self._state.update(
            pose.p,
            pose.q,
            pose.p + self.tip_offset * axis,
            axis,
            dt=self.scene.get_timestep(),
            right_contact=right_contact,
            table_contact=table_contact,
            gripper_closed=self.is_right_gripper_close(),
            gripper_open=self.is_right_gripper_open(),
            arms_ready=(
                self._pose_is_ready(self.robot.get_left_ee_pose(), self._ready_left_pose)
                and self._pose_is_ready(self.robot.get_right_ee_pose(), self._ready_right_pose)
            ),
            shaker_still=(
                np.linalg.norm(self._shaker_body.linear_velocity) < 0.025
                and np.linalg.norm(self._shaker_body.angular_velocity) < 0.25
            ),
        )

    def _track_pickup_contacts(self, shaker_position, right_contact, gripper_closed):
        """Reject actual robot-pot collisions before the held shaker clears pickup."""
        if self._pickup_complete:
            return
        for contact in self.scene.get_contacts():
            entities = [body.entity for body in contact.bodies]
            if self.pot not in entities:
                continue
            other = entities[1] if entities[0] == self.pot else entities[0]
            if other not in self._robot_link_entities:
                continue
            impulse = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
            self._pickup_pot_contact_impulse = max(self._pickup_pot_contact_impulse, impulse)
            if impulse > 1e-5:
                self._state._fail("robot_pot_contact_during_pickup")
        self._pickup_complete = bool(
            right_contact
            and gripper_closed
            and shaker_position[2] >= self.initial_shaker_pose.p[2] + 0.10
        )

    def _track_turn_contacts(self):
        """Reject pot contacts by either the robot or held shaker during rotation."""
        if not self._turn_in_progress:
            return
        for contact in self.scene.get_contacts():
            entities = [body.entity for body in contact.bodies]
            if self.pot not in entities:
                continue
            other = entities[1] if entities[0] == self.pot else entities[0]
            if other != self.shaker.actor and other not in self._robot_link_entities:
                continue
            impulse = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
            self._turn_pot_contact_impulse = max(self._turn_pot_contact_impulse, impulse)
            if impulse > 1e-5:
                self._state._fail("pot_contact_during_turn")

    def _settle(self, condition=None):
        for _ in range(math.ceil(self.settle_seconds / self.scene.get_timestep())):
            if condition is not None and condition():
                break
            self._step_simulation()

    def _gripper_pose_for_shaker(self, position, quaternion):
        """Transform the measured grasp to the requested object pose."""
        current_ee = self.robot.get_right_ee_pose()
        gripper = sapien.Pose(current_ee[:3], current_ee[3:])
        return sapien.Pose(position, quaternion) * self.shaker.get_pose().inv() * gripper

    def _move_shaker(self, position, quaternion, *, settle=False, time_scale=1.0, turning=False):
        target = self._gripper_pose_for_shaker(position, quaternion)
        if time_scale != 1.0 or turning:
            plan = self.robot.right_plan_path(target.p.tolist() + target.q.tolist())
            if not self._right_plan_is_valid(plan):
                self.plan_success = False
                return False
            return self._execute_right_plan(
                self._retime_right_plan(plan, time_scale), settle=settle, turning=turning
            )
        self.move(self.move_to_pose(ArmTag("right"), target))
        if settle:
            self._settle()
        return self.plan_success and not self._state.failed_reason

    @staticmethod
    def _retime_right_plan(plan, time_scale):
        """Slow a dense trajectory while retaining its geometric path and endpoints."""
        if not math.isfinite(time_scale) or time_scale < 1:
            raise ValueError("time_scale must be finite and at least 1")
        position = np.asarray(plan["position"], dtype=float)
        velocity = np.asarray(plan["velocity"], dtype=float)
        if position.ndim != 2 or len(position) < 2 or velocity.shape != position.shape:
            raise ValueError("A retimed plan must contain matching position/velocity arrays")
        steps = math.ceil((len(position) - 1) * time_scale) + 1
        actual_scale = (steps - 1) / (len(position) - 1)
        original_times = np.arange(len(position), dtype=float)
        sample_times = np.linspace(0, len(position) - 1, steps)
        result = deepcopy(plan)
        result["position"] = np.column_stack(
            [np.interp(sample_times, original_times, joint) for joint in position.T]
        )
        result["velocity"] = np.column_stack(
            [np.interp(sample_times, original_times, joint) / actual_scale for joint in velocity.T]
        )
        return result

    def _right_plan_is_valid(self, plan):
        return (
            plan["status"] == "Success"
            and self._joint_path_excess(
                [plan], self._arm_joint_bounds(self.robot.right_arm_joints)
            )
            <= 0
        )

    def _execute_right_plan(self, plan, *, settle=False, turning=False):
        """Execute and record exactly the trajectory that passed validation."""
        if not self._right_plan_is_valid(plan):
            self.plan_success = False
            return False
        self.right_joint_path.append(deepcopy(plan))
        if turning:
            self._turn_in_progress = True
        try:
            self.take_dense_action(
                {"left_arm": None, "left_gripper": None, "right_arm": plan, "right_gripper": None}
            )
        finally:
            if turning:
                self._turn_in_progress = False
        if settle:
            self._settle()
        return self.plan_success and not self._state.failed_reason

    def _turn_shaker(self, position, angle):
        """Turn continuously to the requested angle using one checked trajectory."""
        for roll_degrees in (100, 90, 110, 80, 120):
            quaternion = self._tilted_quaternion(angle, roll_degrees)
            target = self._gripper_pose_for_shaker(position, quaternion)
            # Failed candidates leave the task usable. Execute the first checked
            # trajectory directly so replanning cannot select another IK branch.
            plan = self.robot.right_plan_path(target.p.tolist() + target.q.tolist())
            if not self._right_plan_is_valid(plan):
                continue
            # Slower rotation reduces inertial slip without adding angular stops.
            plan = self._retime_right_plan(plan, 2.0)
            if not self._execute_right_plan(plan, turning=True):
                return None
            self._tilt_roll_degrees.append(roll_degrees)
            return quaternion.copy()
        self.plan_success = False
        self.info["scripted_failure"] = "no_reachable_tilt"
        return None

    def _replace_shaker(self):
        """Prefer a direct descent, splitting it if trajectory refinement fails."""
        position, quaternion = self.initial_shaker_pose.p, self.initial_shaker_pose.q
        target = self._gripper_pose_for_shaker(position, quaternion)
        plan = self.robot.right_plan_path(target.p.tolist() + target.q.tolist())
        if self._right_plan_is_valid(plan):
            return self._execute_right_plan(plan, settle=True)
        if not self._move_shaker(position + [0, 0, 0.05], quaternion):
            return False
        return self._move_shaker(position, quaternion, settle=True)

    def _replace_shaker_until_observed(self):
        """Correct measured placement error before opening the fingers."""
        for _ in range(3):
            if not self._replace_shaker():
                return False
            pose = self.shaker.get_pose()
            _, table_contact = self._contact_flags()
            if (
                self._state.replaced
                and self._state._original_pose(pose.p, pose.q)
                and table_contact
            ):
                return True
        self.info["scripted_failure"] = "replacement_not_observed"
        self._state._fail("replacement_not_observed")
        return False

    def choose_best_pose(self, res_pose, center_pose, arm_tag=None):
        """Reject low side grasps that press Aloha's fingers into the table."""
        if arm_tag != "right":
            return super().choose_best_pose(res_pose, center_pose, arm_tag)
        candidates = self.robot.create_target_pose_list(res_pose, center_pose, arm_tag)
        plans = self.robot.right_plan_multi_path(candidates)
        bounds = self._arm_joint_bounds(self.robot.right_arm_joints)
        for index, candidate in enumerate(candidates):
            if plans["status"][index] != "Success":
                continue
            if self._grasp_height < 0.8 and candidate[2] < self.shaker.get_pose().p[2] + 0.045:
                continue
            if self._joint_path_excess([{"position": plans["position"][index]}], bounds) > 0:
                continue
            return candidate
        return None

    def _shake_once(self, neutral, expected_count):
        forward = neutral + (self.shake_distance + 0.005) * self.shake_axis
        # Re-evaluate the measured grasp if a compliant finger undershoots
        # the endpoint. Corrections extend the same stroke, not its count.
        for _ in range(3):
            if not self._move_shaker(forward, self.shake_quaternion):
                return False
            self._settle(
                lambda: np.dot(self.shaker.get_pose().p - neutral, self.shake_axis) + 1e-9
                >= self.shake_distance
            )
            stroke = np.dot(self.shaker.get_pose().p - neutral, self.shake_axis)
            if stroke + 1e-9 >= self.shake_distance:
                break
        for _ in range(3):
            if not self._move_shaker(neutral, self.shake_quaternion):
                return False
            self._settle(lambda: self._state.completed_repeats == expected_count)
            if self._state.completed_repeats == expected_count:
                return True
        self.info["scripted_failure"] = "shake_not_observed"
        return False

    def play_once(self):
        right = ArmTag("right")
        grasp_pose = self.get_grasp_pose(
            self.shaker, arm_tag=right, pre_dis=0.0, contact_point_id=0
        )
        if grasp_pose is None:
            self.plan_success = False
            self.info["scripted_failure"] = "no_reachable_grasp"
            return self._episode_info()
        # Approach the side grasp from above. A backward approach can fold the
        # Aloha elbow against its base even when both endpoint IK solutions exist.
        approach = np.array(grasp_pose, dtype=float)
        approach[2] += 0.08
        self.move(self.move_to_pose(right, approach))
        self.move(self.move_to_pose(right, grasp_pose))
        self._settle()
        self.move(self.close_gripper(right))
        if not self.plan_success:
            return self._episode_info()
        lifted = self.initial_shaker_pose.p + [0, 0, 0.17]
        if not self._move_shaker(lifted, self.initial_shaker_pose.q):
            return self._episode_info()
        if not self._move_shaker(self.turn_position, self.initial_shaker_pose.q):
            return self._episode_info()
        quaternion = self._turn_shaker(self.turn_position, 3 * math.pi / 4)
        if quaternion is None:
            return self._episode_info()
        self.shake_quaternion = quaternion
        if not self._move_shaker(self.shake_origin, self.shake_quaternion):
            return self._episode_info()
        self._settle(lambda: self._state.armed)
        if not self._state.armed:
            self.info["scripted_failure"] = "aim_not_observed"
            return self._episode_info()
        # The checker records the physically reached neutral pose. Return to that
        # pose before starting so tracking error cannot shorten a measured stroke.
        neutral = self._state.anchor.copy()
        if not self._move_shaker(neutral, self.shake_quaternion):
            return self._episode_info()
        for repetition in range(self.repeat_count):
            if not self._shake_once(neutral, repetition + 1):
                return self._episode_info()
        return_turn_position = neutral + [0, 0, self.turn_clearance]
        if not self._move_shaker(return_turn_position, self.shake_quaternion):
            return self._episode_info()
        if not self._move_shaker(
            return_turn_position, self.initial_shaker_pose.q, time_scale=2.0, turning=True
        ):
            return self._episode_info()
        if not self._move_shaker(lifted, self.initial_shaker_pose.q):
            return self._episode_info()
        if not self._replace_shaker_until_observed():
            return self._episode_info()
        self.move(self.open_gripper(right))
        self.move(self.move_by_displacement(right, z=0.08))
        self.move(self.back_to_origin(right))
        self._settle()
        return self._episode_info()

    def _episode_info(self):
        self.info["task_progress"] = self.get_task_progress()
        self.info["instruction"] = self.get_instruction()
        return self.info

    def get_task_progress(self):
        return {} if self._state is None else self._state.snapshot()

    def get_task_metadata(self):
        final_pose = self.shaker.get_pose()
        pot_pose = self.pot.get_pose()
        return {
            "initial_shaker_pose": self.initial_shaker_pose.p.tolist()
            + self.initial_shaker_pose.q.tolist(),
            "final_shaker_pose": final_pose.p.tolist() + final_pose.q.tolist(),
            "pot_pose": pot_pose.p.tolist() + pot_pose.q.tolist(),
            "pot_center": self.pot_center.tolist(),
            "shake_origin": self.shake_origin.tolist(),
            "turn_position": self.turn_position.tolist(),
            "turn_clearance": self.turn_clearance,
            "shake_axis": self.shake_axis.tolist(),
            "shake_quaternion": self.shake_quaternion.tolist(),
            "observed_neutral_position": (
                None if self._state.anchor is None else self._state.anchor.tolist()
            ),
            "shake_distance": self.shake_distance,
            "min_shaker_pot_clearance": self.min_shaker_pot_clearance,
            "initial_shaker_pot_clearance": self.initial_shaker_pot_clearance,
            "pot_footprint_bounds": (self._pot_local_bounds + self.pot_center[:2]).tolist(),
            "pickup_complete": self._pickup_complete,
            "pickup_pot_contact_impulse": self._pickup_pot_contact_impulse,
            "turn_pot_contact_impulse": self._turn_pot_contact_impulse,
            "tilt_roll_degrees": list(self._tilt_roll_degrees),
        }

    def check_success(self):
        return self._state is not None and self._state.success


class ShakeSaltState:
    """Count measured, aimed out-and-back strokes and verify the final replacement."""

    def __init__(
        self,
        repeat_count,
        initial_position,
        initial_quaternion,
        shake_origin,
        shake_axis,
        pot_center,
        *,
        shake_distance=0.05,
        pot_radius=0.065,
        position_tolerance=0.02,
        angle_tolerance=0.15,
    ):
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 1:
            raise ValueError("repeat_count must be a positive integer")
        for name, value in (
            ("shake_distance", shake_distance),
            ("pot_radius", pot_radius),
            ("position_tolerance", position_tolerance),
            ("angle_tolerance", angle_tolerance),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name, value, size in (
            ("initial_position", initial_position, 3),
            ("initial_quaternion", initial_quaternion, 4),
            ("shake_origin", shake_origin, 3),
            ("shake_axis", shake_axis, 3),
            ("pot_center", pot_center, 3),
        ):
            vector = np.asarray(value, dtype=float)
            if vector.shape != (size,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} must contain {size} finite coordinates")
            setattr(self, name, vector.copy())
        if np.linalg.norm(self.initial_quaternion) < 1e-9:
            raise ValueError("initial_quaternion must have nonzero norm")
        if np.linalg.norm(self.shake_axis) < 1e-9:
            raise ValueError("shake_axis must have nonzero norm")
        self.initial_quaternion /= np.linalg.norm(self.initial_quaternion)
        self.shake_axis /= np.linalg.norm(self.shake_axis)
        if abs(self.shake_axis[2] + math.sqrt(0.5)) > 0.02:
            raise ValueError("shake_axis must point 45 degrees below horizontal")
        self.repeat_count = repeat_count
        self.shake_distance = shake_distance
        self.pot_radius = pot_radius
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.completed_repeats = 0
        self.phase = "ready"
        self.failed_reason = None
        self.armed = False
        self.grasp_started = False
        self.outward = False
        self.anchor = None
        self.anchor_quaternion = None
        self.aim_time = 0.0
        self.aim_position = None
        self.aim_quaternion = None
        self.needs_anchor = False
        self.settle_time = 0.0
        self.missing_contact_time = 0.0
        self.replaced = False
        self.released = False
        self.max_stroke = 0.0
        self.return_tolerance = min(0.008, self.shake_distance * 0.16)
        self.cross_axis_tolerance = 0.015

    @property
    def success(self):
        return self.phase == "success"

    def _fail(self, reason):
        self.failed_reason = reason
        self.phase = "failed"

    def _original_pose(self, position, quaternion):
        norm = np.linalg.norm(quaternion)
        if norm < 1e-9:
            return False
        cosine = abs(np.dot(quaternion / norm, self.initial_quaternion))
        return (
            np.linalg.norm(position - self.initial_position) <= self.position_tolerance
            and 2 * math.acos(np.clip(cosine, 0, 1)) <= self.angle_tolerance
        )

    def _aimed(self, tip, axis):
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return False
        axis = axis / norm
        if np.dot(axis, self.shake_axis) < math.cos(math.radians(12)):
            return False
        if axis[2] >= -0.1 or tip[2] < self.pot_center[2] + 0.02:
            return False
        travel = (self.pot_center[2] - tip[2]) / axis[2]
        hit = tip + axis * travel
        return travel > 0 and np.linalg.norm(hit[:2] - self.pot_center[:2]) <= self.pot_radius

    def update(
        self,
        position,
        quaternion,
        tip_position,
        axis,
        *,
        dt,
        right_contact,
        table_contact,
        gripper_closed,
        gripper_open,
        arms_ready,
        shaker_still,
    ):
        if self.failed_reason is not None or self.success:
            return
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        values = [np.asarray(v, dtype=float) for v in (position, quaternion, tip_position, axis)]
        if any(
            v.shape != (size,) or not np.isfinite(v).all() for v, size in zip(values, (3, 4, 3, 3))
        ):
            self._fail("nonfinite_shaker_pose")
            return
        position, quaternion, tip_position, axis = values
        if np.linalg.norm(quaternion) < 1e-9 or np.linalg.norm(axis) < 1e-9:
            self._fail("invalid_shaker_orientation")
            return
        if position[2] < self.initial_position[2] - 0.05:
            self._fail("shaker_fell_off_table")
            return
        held = right_contact and gripper_closed
        self.grasp_started = self.grasp_started or held
        aimed = self._aimed(tip_position, axis)
        at_start = self._original_pose(position, quaternion) and table_contact
        if self.armed:
            delta = position - self.anchor
            stroke = float(np.dot(delta, self.shake_axis))
            transverse = float(np.linalg.norm(delta - stroke * self.shake_axis))
        else:
            stroke, transverse = 0.0, 0.0

        if self.completed_repeats == self.repeat_count:
            if (
                aimed
                and held
                and transverse <= self.cross_axis_tolerance
                and stroke + 1e-9 >= self.shake_distance
            ):
                self._fail("extra_shake_after_requested_repeats")
                return
            self.replaced = self.replaced or (at_start and held)
            if not gripper_closed:
                if not self.released and (not self.replaced or not at_start):
                    self._fail("released_before_replacement")
                    return
                self.released = True
            if not self.released:
                self.missing_contact_time = (
                    0.0 if right_contact else self.missing_contact_time + dt
                )
                if self.missing_contact_time > 0.12:
                    self._fail("lost_right_hand_grasp")
                    return
            self.phase = "returning" if gripper_open else "restoring"
            finished = (
                at_start and gripper_open and not right_contact and arms_ready and shaker_still
            )
            self.settle_time = self.settle_time + dt if finished else 0.0
            if self.settle_time + 1e-9 >= 0.12:
                self.phase = "success"
            return

        if self.grasp_started:
            if not gripper_closed:
                self._fail("released_before_last_shake")
                return
            self.missing_contact_time = 0.0 if right_contact else self.missing_contact_time + dt
            if self.missing_contact_time > 0.12:
                self._fail("lost_right_hand_grasp")
                return
        if not self.armed:
            self.phase = "aiming" if held else "ready"
            ready_to_shake = (
                held
                and aimed
                and np.linalg.norm(position - self.shake_origin) <= self.position_tolerance
            )
            if not ready_to_shake:
                self.aim_time = 0.0
                self.aim_position = None
                self.aim_quaternion = None
            elif (
                self.aim_position is None
                or np.linalg.norm(position - self.aim_position) > 0.002
                or abs(np.dot(quaternion / np.linalg.norm(quaternion), self.aim_quaternion))
                < math.cos(math.radians(2) / 2)
            ):
                self.aim_position = position.copy()
                self.aim_quaternion = quaternion / np.linalg.norm(quaternion)
                self.aim_time = dt
            else:
                self.aim_time += dt
            if self.aim_time + 1e-9 >= 0.04:
                self.armed = True
                self.anchor = position.copy()
                self.anchor_quaternion = quaternion / np.linalg.norm(quaternion)
                self.phase = "shaking_out"
            return

        # Once a stroke starts, losing the aimed trajectory invalidates that stroke;
        # moving back to the anchor allows a new, complete stroke to be attempted.
        orientation_cosine = abs(
            np.dot(quaternion / np.linalg.norm(quaternion), self.anchor_quaternion)
        )
        valid = (
            aimed
            and transverse <= self.cross_axis_tolerance
            and orientation_cosine >= math.cos(math.radians(12) / 2)
        )
        if not valid:
            self.outward = False
            self.needs_anchor = True
            self.phase = "shaking_out"
            return
        if not held:
            return
        if self.needs_anchor:
            if np.linalg.norm(position - self.anchor) <= self.return_tolerance:
                self.needs_anchor = False
            return
        self.max_stroke = max(self.max_stroke, stroke)
        if not self.outward:
            if stroke + 1e-9 >= self.shake_distance:
                self.outward = True
                self.phase = "shaking_back"
            return
        if np.linalg.norm(position - self.anchor) <= self.return_tolerance:
            self.completed_repeats += 1
            self.outward = False
            self.phase = (
                "restoring" if self.completed_repeats == self.repeat_count else "shaking_out"
            )

    def snapshot(self):
        return {
            "repeat_count": self.repeat_count,
            "completed_repeats": self.completed_repeats,
            "phase": self.phase,
            "armed": self.armed,
            "success": self.success,
            "failed": self.failed_reason is not None,
            "failed_reason": self.failed_reason,
            "max_stroke": self.max_stroke,
        }
