"""Reverse a two-block stack while keeping start/end scene distributions symmetric."""

import math

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import ArmTag, create_box


class _UnreachableTransferGrasp(RuntimeError):
    pass


def _range(values, name):
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite [minimum, maximum] pair") from exc
    if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
        raise ValueError(f"{name} must be a finite [minimum, maximum] pair")
    return values


def sample_stack_positions(x_range, y_range, rng=np.random):
    """Exchangeable nearby endpoints; neither endpoint is a fixed offset/target.

    Reject the *pair* symmetrically so both positions have exactly the same
    marginal distribution, including near the workspace boundaries.
    """
    x_range = _range(x_range, "block_x_range")
    y_range = _range(y_range, "block_y_range")
    if math.hypot(np.ptp(x_range), np.ptp(y_range)) <= 0.10:
        raise ValueError("block position ranges must span more than 0.10 meters")
    for _ in range(10000):
        positions = rng.uniform([x_range[0], y_range[0]], [x_range[1], y_range[1]], (2, 2))
        if 0.10 <= np.linalg.norm(positions[0] - positions[1]) <= 0.16:
            return positions[0], positions[1]
    raise ValueError("Could not sample neighboring stack positions within the block ranges")


class swap_block_order(Base_Task):
    """Use each arm for one transfer; success depends only on measured block order."""

    def setup_demo(self, **kwargs):
        self._state = None  # Base_Task steps physics during initialization/reset.
        repeats = kwargs.get("repeat_count", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats != 1:
            raise ValueError("swap_block_order requires repeat_count=1")
        self.color_blocks = kwargs.get("color_blocks", False)
        if not isinstance(self.color_blocks, bool):
            raise ValueError("color_blocks must be a boolean")
        embodiment = kwargs.get("embodiment_name", kwargs.get("embodiment", [""])[0])
        # Aloha's long fingers span both cubes at the native center grasp.
        # Raise the pinch so the fingertip pads contact only the upper cube.
        self._grasp_dis = 0.025 if embodiment == "aloha-agilex" else 0.0
        self.block_half_size = 0.025
        self.block_x_range = _range(kwargs.get("block_x_range", [-0.10, 0.10]), "block_x_range")
        self.block_y_range = _range(kwargs.get("block_y_range", [-0.20, -0.10]), "block_y_range")
        self.block_yaw_range = _range(
            kwargs.get("block_yaw_range", [-0.35, 0.35]), "block_yaw_range"
        )
        if math.hypot(np.ptp(self.block_x_range), np.ptp(self.block_y_range)) <= 0.10:
            raise ValueError("block position ranges must span more than 0.10 meters")
        super()._init_task_env_(**kwargs)
        self.initial_bottom_pose = self.original_bottom.get_pose()
        self.initial_top_pose = self.original_top.get_pose()
        self._block_bodies = [
            block.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            for block in (self.original_bottom, self.original_top)
        ]
        self._robot_entities = [
            link.entity
            for entity in (self.robot.left_entity, self.robot.right_entity)
            for link in entity.get_links()
        ]
        self._state = SwapBlockOrderState(self.initial_bottom_pose.p)
        self.set_instruction(
            "Swap the stacked blocks so the bottom block is on top."
        )
        self.info["info"] = {}
        self.info["initial_stack"] = {
            "bottom": self._block_metadata(self.original_bottom, self.bottom_color),
            "top": self._block_metadata(self.original_top, self.top_color),
        }
        self.info["color_blocks"] = self.color_blocks
        self.info["scripted_target_xy"] = self.target_xy.tolist()
        self.info["scripted_target_yaw"] = self.target_yaw
        self.info["initial_yaw"] = self.initial_yaw
        self._episode_info()

    @staticmethod
    def _block_metadata(block, color):
        pose = block.get_pose()
        return {"color": color, "position": pose.p.tolist(), "quaternion": pose.q.tolist()}

    def load_actors(self):
        initial_xy, self.target_xy = sample_stack_positions(self.block_x_range, self.block_y_range)
        self.initial_yaw = float(np.random.uniform(*self.block_yaw_range))
        # Sharing the sampled yaw gives identical before/after yaw distributions
        # without adding an unrelated wrist-rotation challenge to the order swap.
        self.target_yaw = self.initial_yaw
        # Always consume the same color draw, making paired red/red and red/blue
        # experiments share geometry for a given seed.
        blue_on_top = bool(np.random.randint(2))
        self.bottom_color = "blue" if self.color_blocks and not blue_on_top else "red"
        self.top_color = "blue" if self.color_blocks and blue_on_top else "red"
        colors = {"red": (0.90, 0.18, 0.12), "blue": (0.12, 0.25, 0.90)}
        rotation = [math.cos(self.initial_yaw / 2), 0, 0, math.sin(self.initial_yaw / 2)]
        blocks = []
        for level, color in enumerate((self.bottom_color, self.top_color)):
            block = create_box(
                scene=self,
                pose=sapien.Pose(
                    [*initial_xy, 0.741 + self.block_half_size * (1 + 2 * level)], rotation
                ),
                half_size=[self.block_half_size] * 3,
                color=colors[color],
                name=f"swap_block_{level}",
            )
            blocks.append(block)
            self.add_prohibit_area(block, padding=0.10)
        self.original_bottom, self.original_top = blocks
        self.prohibited_area.append(
            [
                self.target_xy[0] - 0.10,
                self.target_xy[1] - 0.10,
                self.target_xy[0] + 0.10,
                self.target_xy[1] + 0.10,
            ]
        )

    def _contact_flags(self):
        bottom, top = self.original_bottom.actor, self.original_top.actor
        bottom_on_table = top_on_table = blocks_touching = robot_contact = False
        for contact in self.scene.get_contacts():
            entities = [body.entity for body in contact.bodies]
            if bottom not in entities and top not in entities:
                continue
            if not any(
                point.separation <= 0.001 and np.linalg.norm(point.impulse) > 1e-8
                for point in contact.points
            ):
                continue
            bottom_on_table |= bottom in entities and self.table in entities
            top_on_table |= top in entities and self.table in entities
            blocks_touching |= bottom in entities and top in entities
            robot_contact |= any(entity in self._robot_entities for entity in entities)
        return bottom_on_table, top_on_table, blocks_touching, robot_contact

    def _after_simulation_step(self):
        if self._state is None:
            return
        bottom_table, top_table, touching, robot_contact = self._contact_flags()
        self._state.update(
            self.original_bottom.get_pose().p,
            self.original_top.get_pose().p,
            dt=self.scene.get_timestep(),
            bottom_on_table=bottom_table,
            top_on_table=top_table,
            blocks_touching=touching,
            robot_contact=robot_contact,
            grippers_open=self.is_left_gripper_open() and self.is_right_gripper_open(),
            blocks_still=all(
                np.linalg.norm(body.linear_velocity) < 0.025
                and np.linalg.norm(body.angular_velocity) < 0.25
                for body in self._block_bodies
            ),
        )

    def _settle(self):
        for _ in range(math.ceil(0.30 / self.scene.get_timestep())):
            self._step_simulation()

    def choose_best_pose(self, res_pose, center_pose, arm_tag=None):
        """Select a grasp that can also lift, transport, place and withdraw.

        The native selector tests only the approach pose. Cross-table transfers
        need to retain enough wrist/arm travel for the sampled destination yaw.
        Planning here is read-only: candidates do not advance the simulation.
        """
        context = getattr(self, "_transfer_context", None)
        if context is None:
            return super().choose_best_pose(res_pose, center_pose, arm_tag)
        block, target = context
        side = str(arm_tag)
        entity = getattr(self.robot, f"{side}_entity")
        joints = getattr(self.robot, f"{side}_arm_joints")
        active = entity.get_active_joints()
        indices = [active.index(joint) for joint in joints]
        bounds = self._arm_joint_bounds(joints)
        plan_path = getattr(self.robot, f"{side}_plan_path")
        candidates = self.robot.create_target_pose_list(res_pose, center_pose, arm_tag)
        approaches = getattr(self.robot, f"{side}_plan_multi_path")(candidates)
        block_pose = block.get_pose()
        bottom_in_block = block_pose.inv() * block.get_functional_point(0, "pose")
        destination = sapien.Pose(target[:3], target[3:]) * bottom_in_block.inv()
        for index, candidate in enumerate(candidates):
            if approaches["status"][index] != "Success":
                continue
            approach_path = {"position": approaches["position"][index]}
            if self._joint_path_excess([approach_path], bounds) > 0:
                continue
            pregrasp = sapien.Pose(candidate[:3], candidate[3:])
            grasp = sapien.Pose(
                pregrasp.p
                + pregrasp.to_transformation_matrix()[:3, 0]
                * (0.09 - getattr(self, "_grasp_dis", 0.0)),
                pregrasp.q,
            )
            placed = destination * block_pose.inv() * grasp
            lifted = sapien.Pose(grasp.p + [0, 0, 0.07], grasp.q)
            above_target = sapien.Pose(placed.p + [0, 0, 0.07], placed.q)
            # Keep SAPIEN's float32 dtype: cuRobo consumes last_qpos directly.
            qpos = entity.get_qpos().copy()
            qpos[indices] = approaches["position"][index][-1]
            valid = True
            for waypoint in (grasp, lifted, above_target, placed, above_target):
                path = plan_path(waypoint.p.tolist() + waypoint.q.tolist(), last_qpos=qpos)
                if path["status"] != "Success" or self._joint_path_excess([path], bounds) > 0:
                    valid = False
                    break
                qpos[indices] = path["position"][-1]
            if valid:
                return candidate
        return None

    def choose_grasp_pose(self, *args, **kwargs):
        poses = super().choose_grasp_pose(*args, **kwargs)
        if getattr(self, "_transfer_context", None) is not None and (
            poses is None or any(pose is None for pose in poses)
        ):
            raise _UnreachableTransferGrasp
        return poses

    def _transfer(self, block, arm, target):
        self._transfer_context = (block, target)
        try:
            grasp_action = self.grasp_actor(
                block,
                arm_tag=arm,
                pre_grasp_dis=0.09,
                grasp_dis=getattr(self, "_grasp_dis", 0.0),
            )
        except _UnreachableTransferGrasp:
            self.plan_success = False
            self.info["scripted_failure"] = "no_reachable_transfer_grasp"
            return
        finally:
            self._transfer_context = None
        self.move(grasp_action)
        if not self.plan_success:
            return
        self.move(self.move_by_displacement(arm_tag=arm, z=0.07))
        if not self.plan_success:
            return
        self.move(
            self.place_actor(
                block,
                arm_tag=arm,
                target_pose=target,
                functional_point_id=0,
                pre_dis=0.07,
                dis=0.0,
                pre_dis_axis="fp",
                constrain="align",
            )
        )
        if not self.plan_success:
            return
        self.move(self.move_by_displacement(arm_tag=arm, z=0.07))
        self.move(self.back_to_origin(arm_tag=arm))
        self._settle()

    def play_once(self):
        first_arm = ArmTag("left" if self.initial_bottom_pose.p[0] < 0 else "right")
        second_arm = first_arm.opposite
        # Bottom functional points face down; preserve the sampled stack yaw.
        target = [
            *self.target_xy,
            0.741 + self.table_z_bias,
            0,
            math.cos(self.target_yaw / 2),
            math.sin(self.target_yaw / 2),
            0,
        ]
        self._transfer(self.original_top, first_arm, target)
        if not self.plan_success or self._state.failed_reason:
            return self._episode_info()
        self._transfer(self.original_bottom, second_arm, self.original_top.get_functional_point(1))
        return self._episode_info()

    def get_task_progress(self):
        return {} if self._state is None else self._state.snapshot()

    def get_task_metadata(self):
        """Identity/pose annotations for reports; never included in policy observations."""
        return {
            "color_blocks": self.color_blocks,
            "initial_stack": self.info["initial_stack"],
            "initial_yaw": self.initial_yaw,
            "scripted_target_xy": self.target_xy.tolist(),
            "scripted_target_yaw": self.target_yaw,
            "final_original_bottom": self._block_metadata(self.original_bottom, self.bottom_color),
            "final_original_top": self._block_metadata(self.original_top, self.top_color),
        }

    def _episode_info(self):
        self.info["task_progress"] = self.get_task_progress()
        self.info["instruction"] = self.get_instruction()
        return self.info

    def check_success(self):
        return self._state is not None and self._state.success


class SwapBlockOrderState:
    """Observe original identities, support, release and settling on physics ticks."""

    def __init__(
        self,
        initial_bottom_position,
        *,
        block_half_size=0.025,
        position_tolerance=0.012,
        height_tolerance=0.008,
        settle_duration=0.20,
    ):
        for name, value in (
            ("block_half_size", block_half_size),
            ("position_tolerance", position_tolerance),
            ("height_tolerance", height_tolerance),
            ("settle_duration", settle_duration),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        initial = np.asarray(initial_bottom_position, dtype=float)
        if initial.shape != (3,) or not np.isfinite(initial).all():
            raise ValueError("initial_bottom_position must contain three finite coordinates")
        self.table_center_height = float(initial[2])
        self.block_size = 2 * block_half_size
        self.position_tolerance = position_tolerance
        self.height_tolerance = height_tolerance
        self.settle_duration = settle_duration
        self.settled_time = 0.0
        self.success = False
        self.phase = "initial_order"
        self.failed_reason = None

    def update(
        self,
        initial_bottom_position,
        initial_top_position,
        *,
        dt,
        bottom_on_table,
        top_on_table,
        blocks_touching,
        robot_contact,
        grippers_open,
        blocks_still,
    ):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if self.failed_reason is not None:
            return
        bottom = np.asarray(initial_bottom_position, dtype=float)
        top = np.asarray(initial_top_position, dtype=float)
        if any(p.shape != (3,) or not np.isfinite(p).all() for p in (bottom, top)):
            self.failed_reason = "nonfinite_block_position"
        elif min(bottom[2], top[2]) < self.table_center_height - self.block_size:
            self.failed_reason = "block_fell_off_table"
        if self.failed_reason:
            self.success = False
            self.phase = "failed"
            self.settled_time = 0.0
            return
        reversed_stack = (
            np.linalg.norm(bottom[:2] - top[:2]) <= self.position_tolerance
            and abs(bottom[2] - top[2] - self.block_size) <= self.height_tolerance
            and abs(top[2] - self.table_center_height) <= self.height_tolerance
            and top_on_table
            and not bottom_on_table
            and blocks_touching
        )
        finished = reversed_stack and grippers_open and not robot_contact and blocks_still
        self.settled_time = self.settled_time + dt if finished else 0.0
        self.success = self.settled_time + 1e-9 >= self.settle_duration
        if self.success:
            self.phase = "success"
        elif reversed_stack:
            self.phase = "settling"
        elif bottom_on_table and blocks_touching and top[2] > bottom[2]:
            self.phase = "initial_order"
        else:
            self.phase = "unstacked"

    def snapshot(self):
        return {
            "phase": self.phase,
            "success": self.success,
            "repeat_count": 1,
            "completed_repeats": int(self.success),
            "settled_seconds": self.settled_time,
            "failed": self.failed_reason is not None,
            "failed_reason": self.failed_reason,
        }
