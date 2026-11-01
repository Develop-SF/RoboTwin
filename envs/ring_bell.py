"""Press the existing static bell N times, observing contact and physical release."""

import math

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import ArmTag, create_actor


class ring_bell(Base_Task):
    """Right-arm bell pressing with the left arm held at its ready pose."""

    def setup_demo(self, **kwargs):
        # Base_Task advances physics during setup, including on reused instances.
        self._state = None
        self.repeat_count = kwargs.get("repeat_count", 3)
        state = RingBellState(self.repeat_count)
        for name, default in (
            ("bell_x_range", [0.12, 0.22]),
            ("bell_y_range", [-0.12, -0.04]),
            ("bell_yaw_range", [-0.2, 0.2]),
        ):
            try:
                values = np.asarray(kwargs.get(name, default), dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite [minimum, maximum] pair") from exc
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
                raise ValueError(f"{name} must be a finite [minimum, maximum] pair")
            setattr(self, name, values)
        super()._init_task_env_(**kwargs)
        self.robot.set_origin_endpose()
        self._ready_left_pose = np.array(self.robot.left_original_pose, dtype=float)
        self._ready_right_pose = np.array(self.robot.right_original_pose, dtype=float)
        self._right_gripper_entities = [
            joint.child_link.entity for joint, _, _ in self.robot.right_gripper
        ]
        self._right_gripper_entities.extend(
            link.entity
            for link in self.robot.right_entity.get_links()
            if link.get_name() in self.robot.right_fix_gripper_name
        )
        self._robot_entities = [
            link.entity
            for robot in (self.robot.left_entity, self.robot.right_entity)
            for link in robot.get_links()
        ]
        self._elapsed_time = 0.0
        self._button_press_events = []
        self._last_button_contact = None
        self._state = state
        times = "time" if self.repeat_count == 1 else "times"
        self.set_instruction(
            f"Use the right arm to press the bell {self.repeat_count} {times}."
        )
        self.info["info"] = {
            "{A}": f"050_bell/base{self.bell_id}",
            "{a}": "right",
            "{N}": str(self.repeat_count),
        }
        self._episode_info()

    def load_actors(self):
        x = np.random.uniform(*self.bell_x_range)
        y = np.random.uniform(*self.bell_y_range)
        yaw = np.random.uniform(*self.bell_yaw_range)
        c, s = math.cos(yaw / 2), math.sin(yaw / 2)
        # World yaw composed with click_bell's upright mesh orientation.
        quaternion = [0.5 * (c - s), 0.5 * (c - s), 0.5 * (c + s), 0.5 * (c + s)]
        self.bell_id = 0
        self.bell = create_actor(
            scene=self,
            pose=sapien.Pose([x, y, 0.741], quaternion),
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )
        self.add_prohibit_area(self.bell, padding=0.07)

    def _contact_flags(self):
        """Require a loaded cap contact, not PhysX's separated contact candidates.

        The base0 button cap is 65--66.7 mm up the mesh's local Y axis,
        with a radius of 3.2 mm. Its native marker is down at 38 mm, inside
        the dome; using that marker and a broad tolerance accepts false presses.
        """
        right_top_contact = right_contact = wrong_contact = False
        self._last_button_contact = None
        transform = self.bell.get_pose().to_transformation_matrix()
        inverse_rotation = transform[:3, :3].T
        for contact in self.scene.get_contacts():
            entities = [body.entity for body in contact.bodies]
            if self.bell.actor not in entities or not contact.points:
                continue
            other = entities[1] if entities[0] == self.bell.actor else entities[0]
            loaded_points = [
                point
                for point in contact.points
                if np.linalg.norm(point.impulse) >= 1e-6 and point.separation <= 0.001
            ]
            if not loaded_points:
                continue
            if other in self._right_gripper_entities:
                right_contact = True
                for point in loaded_points:
                    local = inverse_rotation @ (np.asarray(point.position) - transform[:3, 3])
                    normal = inverse_rotation @ np.asarray(point.normal)
                    impulse = inverse_rotation @ np.asarray(point.impulse)
                    if (
                        math.hypot(local[0], local[2]) <= 0.0035
                        and 0.065 <= local[1] <= 0.0677
                        and abs(normal[1]) >= 0.8
                        and abs(impulse[1]) >= 1e-6
                    ):
                        right_top_contact = True
                        self._last_button_contact = {
                            "position_world": np.asarray(point.position).tolist(),
                            "position_bell_local": local.tolist(),
                            "normal_impulse": float(abs(impulse[1])),
                            "separation": float(point.separation),
                        }
            elif other in self._robot_entities:
                wrong_contact = True
        return bool(right_top_contact), right_contact, wrong_contact

    def _button_top(self):
        transform = self.bell.get_pose().to_transformation_matrix()
        return transform[:3, :3] @ np.array([0.0, 0.0667, 0.0]) + transform[:3, 3]

    def _select_press_tip(self):
        """Center the downward pressing face, even when the wrist is tilted."""
        self._press_link = self.robot.right_gripper[0][0].child_link
        candidates = []
        for shape in self._press_link.get_collision_shapes():
            vertices = np.asarray(shape.vertices) * np.asarray(shape.scale)
            transform = (
                self._press_link.entity.get_pose().to_transformation_matrix()
                @ shape.local_pose.to_transformation_matrix()
            )
            face = press_face_center(vertices, shape.triangles, transform[:3, :3])
            height = (transform[:3, :3] @ face + transform[:3, 3])[2]
            candidates.append((height, shape, face))
        _, self._press_shape, self._press_local_point = min(candidates, key=lambda x: x[0])

    def _press_tip(self):
        transform = (
            self._press_link.entity.get_pose().to_transformation_matrix()
            @ self._press_shape.local_pose.to_transformation_matrix()
        )
        return transform[:3, :3] @ self._press_local_point + transform[:3, 3]

    @staticmethod
    def _pose_is_ready(current, target):
        current = np.asarray(current, dtype=float)
        if np.linalg.norm(current[:3] - target[:3]) > 0.02:
            return False
        cosine = abs(np.dot(current[3:], target[3:])) / (
            np.linalg.norm(current[3:]) * np.linalg.norm(target[3:])
        )
        return 2 * math.acos(np.clip(cosine, 0.0, 1.0)) <= 0.15

    def _after_simulation_step(self):
        if self._state is None:
            return
        right_top_contact, right_contact, wrong_contact = self._contact_flags()
        dt = self.scene.get_timestep()
        self._elapsed_time += dt
        was_pressed = self._state.pressed
        completed = self._state.completed_repeats
        left_ready = self._pose_is_ready(self.robot.get_left_ee_pose(), self._ready_left_pose)
        right_pose = self.robot.get_right_ee_pose()
        self._state.update(
            dt=dt,
            right_top_contact=right_top_contact,
            right_contact=right_contact,
            wrong_contact=wrong_contact,
            gripper_closed=self.is_right_gripper_close(),
            gripper_open=self.is_right_gripper_open(),
            ee_height=right_pose[2],
            left_ready=left_ready,
            arms_ready=left_ready and self._pose_is_ready(right_pose, self._ready_right_pose),
        )
        if self._state.pressed and not was_pressed:
            self._button_press_events.append(
                {
                    "press_index": completed + 1,
                    "pressed_at": self._elapsed_time,
                    "normal_force": self._last_button_contact["normal_impulse"] / dt,
                    **self._last_button_contact,
                }
            )
        if self._state.completed_repeats > completed:
            self._button_press_events[-1].update(
                released_at=self._elapsed_time,
                release_lift=float(right_pose[2] - self._state.contact_height),
            )

    def _settle(self, seconds=0.16):
        for _ in range(math.ceil(seconds / self.scene.get_timestep())):
            self._step_simulation()

    def play_once(self):
        right = ArmTag("right")
        button = self._button_top()
        # Keep a reachable wrist orientation, then compensate its tilt by
        # aligning the actual fingertip with the cap before the vertical press.
        self.move(
            self.grasp_actor(
                self.bell,
                arm_tag=right,
                pre_grasp_dis=0.13,
                grasp_dis=0.13,
                contact_point_id=0,
            )
        )
        self._settle()
        if not self.plan_success or self._state.failed_reason:
            return self._episode_info()
        self._select_press_tip()
        hover = np.array(self.robot.get_right_ee_pose(), dtype=float)
        orientation = hover[3:].copy()
        hover[:3] += button + [0, 0, 0.035] - self._press_tip()
        hover[3:] = orientation
        self.move(self.move_to_pose(arm_tag=right, target_pose=hover))
        self._settle()
        for repetition in range(self.repeat_count):
            for _ in range(3):
                press = np.array(self.robot.get_right_ee_pose(), dtype=float)
                press[:3] += button - self._press_tip() - [0, 0, 0.001]
                press[3:] = orientation
                self.move(self.move_to_pose(arm_tag=right, target_pose=press))
                self._settle()
                if not self.plan_success or self._state.failed_reason:
                    return self._episode_info()
                if self._state.pressed:
                    break
            if not self._state.pressed:
                self.info["scripted_failure"] = "button_press_not_observed"
                return self._episode_info()
            hover = np.array(self.robot.get_right_ee_pose(), dtype=float)
            hover[2] += 0.045
            self.move(self.move_to_pose(arm_tag=right, target_pose=hover))
            self._settle()
            if not self.plan_success or self._state.failed_reason:
                return self._episode_info()
            if self._state.completed_repeats != repetition + 1:
                self.info["scripted_failure"] = "bell_release_not_observed"
                return self._episode_info()

        self.move(self.open_gripper(right))
        self.move(self.back_to_origin(arm_tag=right))
        self._settle()
        return self._episode_info()

    def _episode_info(self):
        self.info["task_progress"] = self.get_task_progress()
        self.info["instruction"] = self.get_instruction()
        return self.info

    def get_task_progress(self):
        return {} if self._state is None else self._state.snapshot()

    def get_task_metadata(self):
        pose = self.bell.get_pose()
        return {
            "bell_model": f"050_bell/base{self.bell_id}",
            "bell_pose": pose.p.tolist() + pose.q.tolist(),
            "button_top_world": self._button_top().tolist(),
            "button_cap_radius": 0.0035,
            "button_press_events": self._button_press_events,
            "release_height": self._state.release_height,
            "contact_duration": self._state.contact_duration,
            "release_duration": self._state.release_duration,
            "scripted_failure": self.info.get("scripted_failure"),
        }

    def check_success(self):
        # Queries never advance the counter; both scripts and policies use physics.
        return self._state is not None and self._state.success


def press_face_center(vertices, triangles, rotation):
    """Area-weighted center of the lowest downward face of a convex finger.

    Averaging globally lowest vertices picks an edge on a tilted finger. On
    Piper this displaced the pad center about 9 mm from the bell button.
    Group coplanar triangles so the target is independent of triangulation.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = vertices[np.asarray(triangles, dtype=int)]
    centers = faces.mean(axis=1)
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    areas = np.linalg.norm(normals, axis=1)
    valid = areas > 1e-12
    faces, centers, normals, areas = faces[valid], centers[valid], normals[valid], areas[valid]
    normals /= areas[:, None]
    # Collision-mesh winding may differ between embodiments.
    inward = np.sum(normals * (centers - vertices.mean(axis=0)), axis=1) < 0
    normals[inward] *= -1
    world_normals = normals @ np.asarray(rotation).T
    downward = np.flatnonzero(world_normals[:, 2] <= -0.8)
    if not len(downward):
        raise ValueError("The closed finger has no downward pressing face")
    heights = (centers @ np.asarray(rotation).T)[:, 2]
    lowest = downward[np.argmin(heights[downward])]
    coplanar = (normals @ normals[lowest] > 1 - 1e-5) & (
        np.abs((centers - centers[lowest]) @ normals[lowest]) < 1e-5
    )
    return np.average(centers[coplanar], axis=0, weights=areas[coplanar])


class RingBellState:
    """Count debounced top-contact/lift-away cycles, followed by a settled return."""

    def __init__(
        self,
        repeat_count,
        *,
        contact_duration=0.02,
        release_duration=0.04,
        release_height=0.02,
        settle_duration=0.12,
    ):
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 1:
            raise ValueError("repeat_count must be a positive integer")
        self.repeat_count = repeat_count
        for name, value in (
            ("contact_duration", contact_duration),
            ("release_duration", release_duration),
            ("release_height", release_height),
            ("settle_duration", settle_duration),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, value)
        self.completed_repeats = 0
        self.pressed = False
        self.phase = "ready"
        self.failed_reason = None
        self.contact_time = self.release_time = self.settled_time = 0.0
        self.contact_height = None

    @property
    def success(self):
        return self.phase == "done" and self.failed_reason is None

    def _fail(self, reason):
        self.failed_reason = reason
        self.phase = "failed"

    def update(
        self,
        *,
        dt,
        right_top_contact,
        right_contact,
        gripper_closed,
        gripper_open,
        ee_height,
        left_ready,
        arms_ready,
        wrong_contact=False,
    ):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not math.isfinite(ee_height):
            raise ValueError("ee_height must be finite")
        if self.failed_reason:
            return
        if not left_ready:
            self._fail("left_arm_left_ready_pose")
            return
        if wrong_contact:
            self._fail("bell_contact_by_other_robot_part")
            return

        qualifying_contact = right_top_contact and right_contact and gripper_closed
        if qualifying_contact:
            self.contact_time += dt
            if self.contact_time + 1e-9 >= self.contact_duration:
                if self.completed_repeats >= self.repeat_count:
                    self._fail("extra_press_after_requested_repeats")
                    return
                if not self.pressed:
                    self.contact_height = ee_height
                    self.pressed = True
                self.phase = "pressed"
        else:
            self.contact_time = 0.0

        if self.pressed:
            released = (
                not right_contact
                and not right_top_contact
                and ee_height - self.contact_height + 1e-9 >= self.release_height
            )
            self.release_time = self.release_time + dt if released else 0.0
            if self.release_time + 1e-9 >= self.release_duration:
                self.completed_repeats += 1
                self.pressed = False
                self.contact_time = self.release_time = 0.0
                self.phase = (
                    "returning" if self.completed_repeats == self.repeat_count else "ready"
                )

        final_ready = (
            self.completed_repeats == self.repeat_count
            and not self.pressed
            and not right_contact
            and not right_top_contact
            and gripper_open
            and arms_ready
        )
        self.settled_time = self.settled_time + dt if final_ready else 0.0
        if self.settled_time + 1e-9 >= self.settle_duration:
            self.phase = "done"
        elif self.completed_repeats == self.repeat_count:
            self.phase = "returning"

    def snapshot(self):
        return {
            "repeat_count": self.repeat_count,
            "completed_repeats": self.completed_repeats,
            "phase": self.phase,
            "pressed": self.pressed,
            "failed_reason": self.failed_reason,
            "success": self.success,
        }
