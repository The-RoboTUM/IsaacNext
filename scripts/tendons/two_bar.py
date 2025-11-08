"""This script demonstrates applying random forces to a two-bar robot and visualizing them with markers using IsaacLab API."""

print("Started")

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Apply random forces to a two-bar robot and visualize them."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import carb
import time
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_inv, quat_apply


def get_two_bar_cfg(usd_path: str) -> ArticulationCfg:
    """Returns an ArticulationCfg for the two-bar robot."""
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={"lower_joint": 1.0, "upper_joint": 2.0},
        ),
        actuators={},
    )


def define_force_markers(link_names: list[str]) -> VisualizationMarkers:
    """Define arrow markers for each link to visualize forces."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ForceMarkers",
        markers={
            name: sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.5, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            )
            for name in link_names
        },
    )
    return VisualizationMarkers(marker_cfg)


class TendonManager:
    def __init__(
        self,
        robot: Articulation,
        tendon_length: float,
        stiffness: float,
        damping: float,
        axis_global: torch.Tensor,
        link_names: list[str],  # shape: N_links
        joint_names: list[str],  # shape: N_links-1
        radii: torch.Tensor,  # shape: N_links+1
        pulley_positions_i: torch.Tensor,  # shape: N_links, 3
        pulley_positions_iplus1: torch.Tensor,  # shape: N_links, 3
        configurations: list[str],  # shape: N_links
        motor_winding_alignments: torch.Tensor,  # shape: N_links-1
    ):
        self.device = robot.device
        N_LINKS = len(link_names)
        assert (
            len(joint_names) == N_LINKS - 1
        ), f"number of joints must be one less than number of links {N_LINKS} got {len(joint_names)}"
        N_JOINTS = N_LINKS - 1
        assert axis_global.shape == (
            3,
        ), f"axis_global must be shape (3,), got {axis_global.shape}"
        assert radii.shape == (
            N_LINKS + 1,
        ), f"radii must one more than links, got {radii.shape}"
        assert pulley_positions_i.shape == (
            N_LINKS,
            3,
        ), f"pulley_positions_i must be shape ({N_LINKS}, 3), got {pulley_positions_i.shape}"
        assert pulley_positions_iplus1.shape == (
            N_LINKS,
            3,
        ), f"pulley_positions_iplus1 must be shape ({N_LINKS}, 3), got {pulley_positions_iplus1.shape}"
        assert (
            len(configurations) == N_LINKS
        ), f"configurations must have length {N_LINKS}, got {len(configurations)}"
        assert (
            len(joint_names) == N_JOINTS
        ), f"joint_names must have length {N_JOINTS}, got {len(joint_names)}"

        # for non-joint pulleys tendon_lengths, tangents and initial_windings have to be overridden
        self.tendon_lengths = torch.zeros(N_LINKS, device=self.device)
        self.tangents = torch.zeros((N_LINKS, 3), device=self.device)
        self.first_torque_axis = torch.clone(axis_global).to(self.device)

        self.radius_vectors_i = torch.zeros((N_LINKS, 3), device=self.device)
        self.radius_vectors_iplus1 = torch.zeros((N_LINKS, 3), device=self.device)
        self.initial_windings = torch.zeros(N_JOINTS, device=self.device)
        self.initial_joint_angles = torch.zeros(N_JOINTS, device=self.device)
        self.joint_angle_to_winding_sign = torch.zeros(
            (N_JOINTS, 3), device=self.device
        )

        self.radii = radii
        self.pulley_positions_iplus1 = pulley_positions_iplus1
        self.desired_length = tendon_length
        self.stiffness = stiffness
        self.damping = damping
        self.robot = robot
        self.n_joints = N_JOINTS
        self.n_links = N_LINKS

        for i, conf in enumerate(configurations):
            assert conf in [
                "up-up",
                "up-down",
                "down-up",
                "down-down",
            ], f"Invalid config {conf}"
            conf_a, conf_b = conf.split("-")

            radius_a = radii[i]
            radius_b = radii[i + 1]
            position_a = pulley_positions_i[i]
            position_b = pulley_positions_iplus1[i]
            v_ab = position_b - position_a
            length_ab = torch.norm(v_ab)

            # store
            self.tendon_lengths[i] = length_ab

            v_ab_unit = v_ab / length_ab
            sign_r_a = -1 if conf == "up-up" or conf == "down-down" else 1
            sign_rotation_angle = -1 if conf == "down-down" or conf == "up-down" else 1
            sin_alpha = (radius_b + sign_r_a * radius_a) / length_ab
            sin_alpha = torch.clamp(sin_alpha, -0.999, 0.999)
            alpha = sign_rotation_angle * torch.asin(sin_alpha)
            # rotate v_ab_unit by alpha around axis_global to get tangent at incoming
            quat = torch.zeros(4, device=self.device)
            quat[0] = torch.cos(alpha / 2)
            quat[1:] = axis_global * torch.sin(alpha / 2)
            tangent_a = quat_apply(quat, v_ab_unit)

            # store
            self.tangents[i] = tangent_a

            radius_vector_a = torch.cross(axis_global, tangent_a)
            radius_vector_a /= radius_vector_a.norm()
            if conf_a == "down":
                radius_vector_a *= -1
            radius_vector_b = radius_vector_a if conf_a == conf_b else -radius_vector_a

            # store
            self.radius_vectors_i[i] = radius_vector_a
            self.radius_vectors_iplus1[i] = radius_vector_b

        self.link_indices, _ = self.robot.find_bodies(link_names, preserve_order=True)
        self.joint_indices, _ = self.robot.find_joints(joint_names, preserve_order=True)

        for i, conf in enumerate(configurations[:-1]):
            i_r_1 = self.radius_vectors_iplus1[i]
            iplus1_r_2 = self.radius_vectors_i[i + 1]
            link_quat_iplus1 = self.robot.data.body_link_quat_w[
                0, self.link_indices[i + 1]
            ]
            link_quat_i = self.robot.data.body_link_quat_w[0, self.link_indices[i]]

            world_r_2 = quat_apply(link_quat_iplus1, iplus1_r_2)
            i_r_2 = quat_apply(quat_inv(link_quat_i), world_r_2)

            conf_a, conf_b = conf.split("-")
            sign_wind_b = 1 if conf_b == "down" else -1
            if i == 0:
                self.first_torque_axis *= sign_wind_b

            # store
            self.initial_windings[i] = torch.acos(torch.dot(i_r_1, i_r_2))
            if sign_wind_b * torch.dot(torch.cross(axis_global, i_r_1), i_r_2) < 0:
                self.initial_windings[i] = 2 * torch.pi - self.initial_windings[i]

            # store
            self.initial_joint_angles[i] = self.robot.data.joint_pos[
                0, self.joint_indices[i]
            ]

            # store
            self.joint_angle_to_winding_sign = motor_winding_alignments[i]

        self.last_length = self.compute_length()
        assert (
            tendon_length > self.last_length
        ), f"tendon_length {tendon_length} must be greater than initial length {self.last_length}"

    def compute_length(self):
        return (
            self.tendon_lengths.sum()
            + (
                (
                    self.initial_windings
                    + self.joint_angle_to_winding_sign
                    * (
                        self.robot.data.joint_pos[0, self.joint_indices]
                        - self.initial_joint_angles
                    )
                )
                * self.radii[1:-1]
            ).sum()
        )

    def apply(self, dt: float):
        current_length = self.compute_length()
        delta_length = current_length - self.desired_length
        length_velocity = (current_length - self.last_length) / dt
        if current_length <= self.desired_length:
            self.last_length = current_length
            print(
                f"[Slack] Current length: {current_length.item():.4f}; Velocity: {length_velocity.item():.4f}"
            )
            return

        tension = self.stiffness * delta_length + self.damping * length_velocity
        print(
            f"[Tense] Excess length: {delta_length.item():.4f}; Velocity: {length_velocity.item():.4f}, Tension: {tension.item():.4f}"
        )

        initial_torque = tension * self.radii[1] * self.first_torque_axis
        final_torque = tension * self.radii[-1] * self.first_torque_axis

        for i in range(self.n_joints):
            # apply force at pulley i
            i_tangent_next = quat_apply(
                quat_inv(self.robot.data.body_link_quat_w[0, self.link_indices[i]]),
                quat_apply(
                    self.robot.data.body_link_quat_w[0, self.link_indices[i + 1]],
                    self.tangents[i + 1],
                ),
            )
            force_vector = (
                i_tangent_next
                if i == 0
                else (
                    -self.tangents[i]
                    if i == self.n_joints - 1
                    else -self.tangents[i] + i_tangent_next
                )
            )
            torque = (
                initial_torque
                if i == 0
                else (
                    final_torque
                    if i == self.n_joints - 1
                    else torch.zeros(3, device=self.device)
                )
            )

            self.robot.set_external_force_and_torque(
                forces=(tension * force_vector).unsqueeze(0).unsqueeze(0),
                torques=(torque).unsqueeze(0).unsqueeze(0),
                positions=self.pulley_positions_iplus1[i].unsqueeze(0).unsqueeze(0),
                body_ids=self.link_indices[i : i + 1],
            )
        self.last_length = current_length


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # IsaacLab simulation setup
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.0016
    sim = SimulationContext(sim_cfg)
    #  sim.set_camera_view([5.0, 0.0, 1.5], [0.0, 0.0, 1.0])  # type: ignore

    # Add ground and light
    sim_utils.GroundPlaneCfg().func(
        "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg()
    )
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg()
    )

    # Robot asset path (update as needed)
    asset_path = "/home/linus/RoboTUM/usds/two_bar.usd"
    two_bar_cfg = get_two_bar_cfg(asset_path)
    two_bar_cfg.prim_path = "/World/Bot"
    robot = Articulation(cfg=two_bar_cfg)
    sim.reset()
    robot.write_joint_state_to_sim(
        position=robot.data.default_joint_pos, velocity=robot.data.default_joint_vel
    )
    robot.write_data_to_sim()
    sim.step()  # step once to load the robot
    robot.update(sim.get_physics_dt())

    carb.log_error("[INFO]: Setup complete 1...")
    time.sleep(3)
    carb.log_error("[INFO]: Continuing...")

    # Link names to apply forces to
    link_names = ["center_link", "top_link"]
    link_indices, _ = robot.find_bodies(link_names, preserve_order=True)

    # Markers for force visualization
    # force_markers = define_force_markers(link_names)

    # Play the simulator

    # robot.write_joint_state_to_sim(
    #     position=torch.tensor([[1.0, 2.0]], device=robot.device),
    #     velocity=torch.tensor([[0.0, 0.0]], device=robot.device),
    #     joint_ids=[0, 1],
    # )

    # Tendon manager setup
    tendon_manager = TendonManager(
        robot=robot,
        tendon_length=3.5,
        stiffness=100000.0,
        damping=10000.0,
        axis_global=torch.tensor([0.0, 1.0, 0.0], device=robot.device),
        link_names=["base_link", "center_link", "top_link"],
        joint_names=["lower_joint", "upper_joint"],
        radii=torch.tensor([0.0, 0.05, 0.05, 0.0], device=robot.device),
        pulley_positions_i=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=robot.device
        ),
        pulley_positions_iplus1=torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=robot.device
        ),
        configurations=["down-down", "down-up", "up-up"],
        motor_winding_alignments=torch.tensor([-1.0, -1.0], device=robot.device),
    )

    while simulation_app.is_running():
        tendon_manager.apply(sim.get_physics_dt())
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    simulation_app.close()

    # while simulation_app.is_running():
    #     # Generate random forces for each link (world frame)
    #     forces = np.zeros((1, len(link_indices), 3)).astype(np.float32)
    #     torques = np.zeros_like(forces)
    #     positions = np.zeros_like(forces)
    #     positions[:, -1, 0] = -1  # Apply force at (1,0,0) in body frame
    #     forces[:, -1, 2] = 100
    #     positions[:, 0, 0] = 1  # Apply force at (1,0,0) in body frame
    #     forces[:, 0, 2] = 500
    #     # Apply forces (in world frame, but position in body frame)
    #     robot.set_external_force_and_torque(
    #         forces=torch.from_numpy(forces).to(device),
    #         torques=torch.from_numpy(torques).to(device),
    #         positions=torch.from_numpy(positions).to(device),
    #         body_ids=link_indices,
    #     )
    #     robot.write_data_to_sim()

    #     # Visualize forces as arrows at the force application point (1,0,0 in body frame, transformed to world)
    #     link_poses = robot.data.body_link_pose_w[0]  # (num_links, 7)
    #     marker_positions = []
    #     marker_orientations = []
    #     marker_indices = []
    #     for i, (idx, force) in enumerate(zip(link_indices, forces[0])):
    #         # Get link pose
    #         link_pos = link_poses[idx, :3].cpu().numpy()
    #         link_quat = link_poses[idx, 3:7].cpu().numpy()  # (w, x, y, z)
    #         carb.log_warn(
    #             f"Link {link_names[i]} pose: pos {link_pos}, quat {link_quat}"
    #         )
    #         # Transform application point (positions[0, i]) from body to world
    #         from scipy.spatial.transform import Rotation as R

    #         r = R.from_quat(
    #             [link_quat[1], link_quat[2], link_quat[3], link_quat[0]]
    #         )  # scipy uses (x,y,z,w)
    #         app_point_local = positions[0, i]
    #         app_point_world = link_pos + r.apply(app_point_local)
    #         # Transform force direction from body to world
    #         force_dir_local = force
    #         force_dir_world = r.apply(force_dir_local)
    #         # Arrow orientation: align +X with force_dir_world
    #         norm = np.linalg.norm(force_dir_world)
    #         if norm < 1e-6:
    #             quat = np.array([1.0, 0.0, 0.0, 0.0])
    #         else:
    #             v0 = np.array([1.0, 0.0, 0.0])
    #             v1 = force_dir_world / norm
    #             axis = np.cross(v0, v1)
    #             dot = np.dot(v0, v1)
    #             if np.linalg.norm(axis) < 1e-6:
    #                 if dot > 0:
    #                     quat = np.array([1.0, 0.0, 0.0, 0.0])
    #                 else:
    #                     quat = np.array([0.0, 0.0, 0.0, 1.0])
    #             else:
    #                 axis = axis / np.linalg.norm(axis)
    #                 angle = np.arccos(np.clip(dot, -1.0, 1.0))
    #                 half_angle = angle / 2.0
    #                 quat = np.concatenate(
    #                     ([np.cos(half_angle)], axis * np.sin(half_angle))
    #                 )
    #         marker_positions.append(app_point_world)
    #         marker_orientations.append(quat)
    #         marker_indices.append(i)

    #     scales = np.ones_like(forces[0][0])
    #     force_markers.visualize(
    #         torch.tensor(marker_positions, dtype=torch.float32),
    #         torch.tensor(marker_orientations, dtype=torch.float32),
    #         marker_indices=torch.tensor(marker_indices, dtype=torch.int64),
    #         scales=torch.tensor(scales, dtype=torch.float32).unsqueeze(1).repeat(1, 3),
    #     )

    #     # Step simulation
    #     sim.step()
    #     robot.update(sim.get_physics_dt())
    #     # time.sleep(sim.get_physics_dt())

    # simulation_app.close()


if __name__ == "__main__":
    main()
