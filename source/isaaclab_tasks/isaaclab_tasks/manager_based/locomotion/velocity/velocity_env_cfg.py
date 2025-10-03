# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 1) 头部导入与预定义
# 引入 Isaac Lab 的各种配置类、管理器与工具（地形、传感器、噪声、材质等）。
# ROUGH_TERRAINS_CFG 是粗糙地形生成器的配置，后面会挂到地形导入器上。

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

##
# Scene definition
##

# 2) 场景配置：MySceneCfg(InteractiveSceneCfg)
# 它定义了一个有粗糙地形、带传感器和天空光的交互式场景，供腿式机器人在里面跑强化学习/控制任务。
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", # prim_path="/World/ground"：USD 场景里这个地形的节点位置（就像绝对路径），所有东西都挂在 /World 下。
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        # terrain_type="generator" + terrain_generator=ROUGH_TERRAINS_CFG：用程序生成的粗糙地形（起伏、高差、坑洼等由生成器决定）。
        max_init_terrain_level=5, # 初始可用的地形难度不超过 5 级（配合课程/关卡制时有用）。
        collision_group=-1, # 碰撞组设置。-1 通常表示默认/与所有交互，不做屏蔽。
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", # 当两个接触物体各自有摩擦时，组合方式是相乘（更保守/更“黏”）。
            restitution_combine_mode="multiply",
            # restitution 这个单词在英文里的原意是 “归还、恢复”。
            #
            # 在物理仿真（碰撞动力学）里，restitution 专门指的是 碰撞恢复系数（coefficient of restitution，简称 COR）。
            # 它描述了两个物体碰撞后 反弹的程度。

            static_friction=1.0,
            dynamic_friction=1.0,
            # 静/动摩擦系数。1.0 属于比较“黏”的地面。
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    # 机器人占位 robot
    robot: ArticulationCfg = MISSING
    # 这里不直接指定机器人（缺省为 MISSING）。真正用的时候要在别处把一个具体的 ArticulationCfg（ANYmal、Unitree 等）赋给它。
    #
    # 场景和下游 MDP 都用 asset_name="robot" 来引用这个实体。


    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    #调参建议：地形更激烈时，把 size 加大、resolution 降低（0.08/0.05）可更细致；性能吃紧时反向调整。ray_alignment="yaw" 能减少观测随姿态抖动，通常更稳。

    # 接触力传感器 contact_forces
    # 作用：记录机器人各刚体与外界的接触力、接触状态等。
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # prim_path="{ENV_REGEX_NS}/Robot/.*"：匹配机器人所有部件（.* 正则）。
    #
    # history_length=3：保留最近 3 帧的历史（可用于去抖、统计）。
    #
    # track_air_time=True：统计腾空时间（常用于足部步态奖励，如“foot air time”）。

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # 穹顶环境光（影响渲染，不影响动力学）。
    #
    # intensity=750.0：亮度（渲染单位依赖于后端；把它当相对亮度调节即可）。
    #
    # texture_file=...hdr：高动态范围天空贴图，给场景自然的环境光与反射。
    #
    # 训练时你可以把渲染关掉以省算力；需要拍 Demo/可视化时再开，这个灯光能让画面更通透。


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        # 表示这条命令是给谁的？这里就是给 机器人。

        resampling_time_range=(10.0, 10.0),
        # 意思是：多久换一次命令。
        #
        # 这里就是 每 10 秒换一条新的命令（比如从“往前 0.5m/s”变成“往左 0.8m/s”）。
        #
        # 如果写 (3.0, 5.0)，就会在 3–5 秒之间随机选一个时刻来换命令。


        rel_standing_envs=0.02, # rel 是 relative（相对比例） 的缩写。
        # 这里表示：在所有的并行环境中，有 2% 的环境 会收到 “站着不动”的命令。
        #
        # 为什么要这样？
        # 因为机器人不仅要会跑，也要会稳稳地站住。这个参数让一部分环境专门用来学“站立”。


        rel_heading_envs=1.0,
        # 同样，rel 表示比例。
        #
        # 这里是 100% 的环境都会收到“朝向命令”。
        #
        # 意思是：所有环境的机器人都会有一个目标朝向（比如“面朝北”），它们需要学会转向并保持这个方向。

        heading_command=True,
        # 表示真的要用“朝向”命令。
        #
        # 当它为 True 时，系统会根据“目标朝向”和“当前朝向”的误差，自动生成一个角速度命令（转动的快慢）。
        heading_control_stiffness=0.5,
        # 就像一个“方向盘灵敏度”。
        #
        # 越大 → 转向更快，但可能抖动。
        #
        # 越小 → 转向更慢，但更平稳。
        #
        # 这里设置成 0.5，就是一个折中。


        debug_vis=True,
        # 开启调试可视化。
        #
        # 在仿真里会画箭头，显示“机器人此刻被要求的目标速度/方向”，方便你观察。

        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
        # 这是命令的取值范围：
        #
        # lin_vel_x=(-1.0, 1.0) → 前后速度范围是 -1 到 1 m/s
        #
        # lin_vel_y=(-1.0, 1.0) → 左右速度范围是 -1 到 1 m/s
        #
        # ang_vel_z=(-1.0, 1.0) → 旋转速度范围是 -1 到 1 rad/s
        #
        # heading=(-π, π) → 朝向角度范围是 -180° 到 +180°
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True)
# 这部分就是定义：强化学习策略网络输出的动作，是如何映射到机器人身上的。
#
# 换句话说：
#
# 策略给出一个动作向量（一般在 [-1, 1] 区间）。
#
# 这里规定：这些动作要被解释为“关节位置目标（Joint Position）”。
#
# 之后环境会把这个目标传给机器人（通常是 PD 控制器），让关节往那个位置运动。
#
# 参数逐个解释
# asset_name="robot"
#
# 指明这个动作是作用到哪个实体（就是我们之前定义的 robot）。
#
# joint_names=[".*"]
#
# ".*" 是正则表达式，意思是“所有关节”。
#
# 所以这个动作会控制机器人身上所有的可动关节。
#
# 如果只想控制腿部，可以写 "leg.*"，只匹配腿的关节。
#
# scale=0.5
#
# 缩放系数，作用是把策略的输出（范围 [-1, 1]）缩小。
#
# 举例：
#
# 策略输出 0.8
#
# 乘上 scale=0.5 → 0.4
#
# 最后这个 0.4 会加到默认关节角度上，作为目标位置。
#
# 这么做的好处是避免动作过大，让训练更稳定。
#
# use_default_offset=True
#
# 表示关节的目标值是基于一个默认偏置来的。
#
# 默认偏置通常是机器人的“站立姿态”。
#
# 举例：
#
# 站立时某个膝关节的角度是 0.8 rad
#
# 策略输出一个动作 0.2（乘上 scale 后可能是 0.1 rad）
#
# 最终目标关节角度 = 0.8 + 0.1 = 0.9 rad
#
# 这样机器人动作是“围绕站立姿势小幅调整”，而不是乱跳。

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # 机体线速度（机体坐标系）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))

        # 注：Unoise(n_min, n_max) 是加性均匀噪声，在给定区间内随机采样后直接加到该观测上，帮助抗噪与泛化。

        # 机体角速度（机体坐标系）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        # 重力投影（projected_gravity）
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )


        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})


        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            # 开启观测“腐蚀/扰动”机制（比如随机丢失/置零/延迟等，视框架实现开启哪些），用于鲁棒性。

            self.concatenate_terms = True
            # 把下面每个观测项按书写顺序拼在一起，形成一个单一的一维向量作为策略输入（顺序就是网络看到的列顺序）。

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
