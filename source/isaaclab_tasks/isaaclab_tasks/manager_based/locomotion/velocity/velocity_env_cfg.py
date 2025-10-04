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
# 这一段就是在给“训练过程里的事件”做配置：什么时候（启动/复位/间隔）对机器人或环境做什么小动作（改摩擦、改质量、改质心、随机初始位姿、推一把……）。这么做的目的主要是两点：
#
# 域随机化：让机器人别只会在一种理想条件下走，换个条件也能稳。
#
# 稳健性训练：学会被推一下也不倒、从各种初始姿态都能站起来走。
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm( # 设定/随机化刚体接触材质
        func=mdp.randomize_rigid_body_material,
        mode="startup", # 启动时做一次（环境创建时）
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
        # 干嘛用：给机器人所有部件设置摩擦和反弹参数（与地面接触相关）。
        #
        # 什么时候：每个并行环境创建时一次。
        #
        # 参数：
        #
        # body_names=".*"：机器人所有刚体。
        #
        # static_friction_range=(0.8,0.8) / dynamic_friction_range=(0.6,0.6)：其实这里没有随机（上下界相同），就是把静摩擦=0.8、动摩擦=0.6 固定住。
        #
        # restitution_range=(0.0, 0.0)：不弹跳。
        #
        # num_buckets=64：如果你给了区间，才会按“分桶”随机；这儿是固定值，用不到。
        #
        # 为什么：给
    )

    add_base_mass = EventTerm( # 给底座质量加个偏差
        func=mdp.randomize_rigid_body_mass,
        mode="startup", # 启动时做一次（环境创建时）
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
        # 干嘛用：在底座质量上加/减一个随机值（-5 到 +5）。
        #
        # 什么时候：启动时一次。
        #
        # 为什么：现实里装了相机、电池、负载重量都会变。提前让策略习惯质量不确定。
        #
        # 提示：如果你的机器人很小，±5 可能过大；可以缩小到 ±1 或按比例设定。
    )

    base_com = EventTerm( # 随机化底座质心位置（COM）
        func=mdp.randomize_rigid_body_com,
        mode="startup", # 启动时做一次（环境创建时）
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
        # 干嘛用：把底座质心小幅偏移（前后/左右 ±5 cm，上下 ±1 cm）。
        #
        # 什么时候：启动时一次。
        #
        # 为什么：现实装配误差、负载摆放不同会导致质心变化；提前适应就更稳。
    )

    # reset
    # 每一回合开始前，随机一下起始条件，防止策略“背答案”。
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque, # 复位时施加外力/外矩（这里其实关了）
        mode="reset", # 每个 episode 复位时做
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    # 干嘛用：在复位瞬间拍一下机器人（加力/矩）。
    #
    # 现在的设置：全是 0，相当于没开。
    #
    # 为什么保留：占位，方便以后打开，比如让它在回合开始就带点扰动，练“起步抗干扰”。

    reset_base = EventTerm( # 根姿态/速度随机化复位
        func=mdp.reset_root_state_uniform,
        mode="reset", # 每个 episode 复位时做
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # → 表示每次复位时，机器人在 x、y 平面上的位置会在 ±0.5 米范围内随机摆放。
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

    reset_robot_joints = EventTerm( # 关节姿态随机到默认姿态的某个比例
        func=mdp.reset_joints_by_scale,
        mode="reset", # 每个 episode 复位时做
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
        # 干嘛用：按比例把关节位置重置到“默认站姿”的 0.5× ~ 1.5× 附近（不同关节按各自默认角度缩放），关节速度设为 0。
        #
        # 为什么：避免每次都从一模一样的站姿开始；让策略习惯姿态有点偏也能恢复。
        #
        # 小贴士：如果某些关节默认角度接近 0，“按比例”就没啥变化；这时可改成“加偏移”的复位函数或适当调默认姿态。
    )

    # interval
    # 就像你训练时隔三差五推它一下，看它会不会摔、能不能稳住。
    push_robot = EventTerm( # 训练过程中随机“推一把”
        func=mdp.push_by_setting_velocity,
        mode="interval", # 回合进行中，每隔一段时间触发
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        # 干嘛用：每隔 10~15 秒，随机给底座一个瞬时的水平速度（x/y 方向 -0.5~0.5 m/s）。
        #
        # 效果：像“侧面撞了一下”或“地面突然滑了一下”；看机器人能不能自我恢复。
        #
        # 为什么：现实世界到处都有扰动（人轻碰一下、地面不平、绊一下），不练这个到了真机就容易出事。
    )
    # 一图流（触发时机）
    #
    # 启动（startup）：创建环境时 → 设材质 / 质量 / 质心
    #
    # 每次复位（reset）：新一回合开始 → 随机根姿态/速度 / 关节初值（可选外力）
    #
    # 间隔（interval）：训练进行中 → 随机时刻推一下


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task

    # 跟踪水平线速度，指数型）
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    ) # 作用：让实际的 v_x, v_y 接近命令速度（来自 base_velocity）。
    # “exp” 提示是指数型相似度：误差小 → 接近 1；误差大 → 快速掉到 0。
    # 走对方向、走对快慢就高分。权重 1.0，最重要的项之一。

    # （跟踪偏航角速度，指数型）
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # 作用：让实际的 w_x 跟上命令（若开启 heading 模式，就是跟上“朝向误差产生的目标角速度”）。
    # 权重 0.5：重要，但不如线速度重要。


    # -- penalties
    # 代价/惩罚（抑制不良行为）
    # 这些是负权重，表示“越大越扣分”。

    # （竖直速度惩罚）
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 不鼓励上下弹跳（𝑣_𝑧 大就扣多点），有助于稳步而不是蹦迪。

    # （滚转/俯仰角速度惩罚）
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 抑制左右/前后方向的晃动(ω_x, ω_y)  ，让上身更稳。

    # （关节力矩惩罚）
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # 鼓励省力、能效更好。权重很小，作为轻微正则。

    # （关节加速度惩罚）
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 抑制突然发力，让动作更顺滑，减少机械冲击。

    # （动作变化率惩罚）
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 连续两步动作差别太大就扣分 → 平滑控制，减小抖动与噪音。

    # 步态/接触相关（教它“走得像话”）
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # 用接触传感器统计脚离地的时间，给“正常摆腿”加分。
    #
    # threshold=0.5 常用于：只有当命令速度超过某阈值时才鼓励更明显的腾空（走/跑起来），站桩时不会鼓励乱抬脚。
    #
    # 直觉：别拖着脚走，该迈步就迈步。

    # （不期望的接触  的惩罚）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # 大腿等不该碰地的部位接触地面就扣分（阈值控制敏感度）。
    #
    # 防止“用大腿/髋部蹭地”这种作弊/不自然的姿态。


    # -- optional penalties
    # （保持机体水平）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # 让机体尽量“水平”。现在关掉了；若地形太颠，可开小权重做姿态保底。

    # （接近关节限位惩罚）
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    # 靠近关节极限时扣分，保护关节、留冗余。现在关掉；若真机应用可逐渐开一点。

# 整体直觉图
#
# 主航向：track_lin_vel_xy_exp（1.0） + track_ang_vel_z_exp（0.5）
# → 走对方向和速度，奖励拉满。
#
# 稳 & 省：lin_vel_z_l2、ang_vel_xy_l2、torques/acc/action_rate
# → 少弹跳、少摇晃、动作平滑、省力。
#
# 像“走路”：feet_air_time（迈步）、undesired_contacts（别用大腿着地）。

@configclass
# 终止条件

# 环境里每一步都会检查这些“DoneTerm”。满足就立刻结束本回合（episode），然后重置进下一回合。这样做有两个目的：
# 不在“坏状态”（比如摔倒）上浪费时间采样；
# 给策略一个清晰的信号：那样做会直接出局。
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 含义：到时间就结束本回合。
    # 时间长短由环境里设置的 episode_length_s 决定（这份配置里是 20 秒, 见下面）。
    # 作用：保证每回合最长跑 20 秒，不会无限长；

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

    # 含义：底座（base）不许碰地。一旦检测到底座与地面有接触（强度超过阈值），立即终止。
    #
    # 用到的是你前面配置好的 contact_forces 传感器，这里只关注 body_names="base"（底座）。
    #
    # threshold=1.0：阈值（通常代表接触力或接触强度的门槛），超过就判“非法接触”。
    #
    # 作用：把“摔倒”快速判出局，让策略知道“倒地=坏事”，别靠趴地取巧。
    #
    # 对比一下奖励里那个 undesired_contacts（惩罚大腿蹭地）：
    #
    # 大腿蹭地：扣分但不断回合（逼它姿态好看）；
    #
    # 底座碰地：直接结束（这是“致命错误”）。


@configclass
# 课程学习 / 动态难度
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # 含义：随着策略变强，自动把地形难度调高；如果表现变差，也可能退回低难度。
    #
    # 你在环境的 __post_init__ 里已经看到：如果配置了这个项，就把地形生成器的 curriculum=True 打开——也就是说粗糙地形会分等级，不同回合或不同并行环境会被分配到不同“坑洼/坡度/台阶”的等级上。
    #
    # 典型逻辑（概念上，具体细节看 mdp 实现）：
    #
    # 先从容易的地形开始（低等级）。
    #
    # 如果在当前等级里，速度跟踪做得好、摔倒少，系统就提升一个等级。
    #
    # 如果在高等级里表现差，可能降级或停在当前等级练一会儿。
    #
    # 为什么要这样？
    #
    # 直接在高难度地形上学，策略可能一开始什么都做不好，收不到有效奖励；
    #
    # 先易后难，像游戏打怪升级，更稳、更快收敛，最终学到“粗糙地形也能稳稳跟踪”的能力。


##
# Environment configuration
##


@configclass
# 这一段是把整个训练环境真正“组装起来”的地方：
# 把场景、观测/动作/命令、奖励、终止、事件、课程学习都挂到一个 ManagerBasedRLEnvCfg 上，
# 并在 __post_init__ 里把仿真频率、时长、传感器刷新、地形课程难度等细节定下来。
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # num_envs=4096：一次并行跑 4096 个环境（采样效率暴增，但显存/算力要顶得住）。
    # env_spacing=2.5：每个环境原点之间相距 2.5 米，避免互相碰撞/视觉重叠。


    # Basic settings
    # 前面你看过——网络要看什么（观测）、怎么发力（动作）、要跟谁（命令）。
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()



    # MDP settings
    # 奖励函数、回合结束规则、扰动/随机化事件、递进式加难度。
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


    # 关键运行参数
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4 # 每 4 个物理步才给一次新动作（俗称 action repeat）。
        # ⇒ 控制频率 = 200 / 4 = 50 Hz（策略每 20 ms 出一次动作）。
        #
        # 好处：省算力、更稳；坏处：响应没那么快。

        self.episode_length_s = 20.0 # 每回合 20 秒 到点就结束（配合 time_out 终止项）。
        # simulation settings
        self.sim.dt = 0.005 # 物理时间步长 = 5 ms → 物理引擎更新频率 200 Hz。
        self.sim.render_interval = self.decimation # 渲染也按“每次动作”刷新一次（不是每个物理步都渲染），更省性能。


        self.sim.physics_material = self.scene.terrain.physics_material
        # 统一物理材质：把仿真默认的接触材质设置为与地形一致，避免“地面材质 vs 全局材质”不一致导致的摩擦奇怪。

        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # 渲染也按“每次动作”刷新一次（不是每个物理步都渲染），更省性能。



        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            # 高度扫描（ray caster）：更新周期 = decimation × dt = 4 × 0.005 = 0.02 s（50 Hz）。
            # 把地形“感知频率”和控制频率对齐，够用了；没必要 200 Hz 打光线，省算力。

            # decimation： 每隔多少个物理仿真步才执行一次控制动作。


        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
            # 接触力传感器：更新周期 = dt = 0.005 s（200 Hz）。
            # 接触事件瞬时性强，每个物理步都要更新，避免漏检测（比如落脚瞬间）。



        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
        # getattr(obj, name, default)：如果对象 obj 有名为 name 的属性，就返回它；否则返回 default（这里是 None）。

            if self.scene.terrain.terrain_generator is not None:
            # 确认确实存在地形生成器。
                self.scene.terrain.terrain_generator.curriculum = True


        else:
            # 如果没有配置 terrain_levels，
            # 则（在有生成器的前提下）关闭课程模式，
            # 地形难度保持固定/不随训练进度动态提升。
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # 逻辑小结（用一句话串起来）
        # 有 terrain_levels → 打开 terrain_generator.curriculum（启用地形课程学习）。
        # 没有 terrain_levels → 关闭 terrain_generator.curriculum（固定难度）。
        # 两个分支都先确认确实有地形生成器，才去改它的 curriculum 开关。