import argparse
from isaaclab.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from ts_cfg import SAW_DIGIT_R_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms
import torch
import numpy as np
import cv2
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SphereCfg
from img_utils import process_image_rgba
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
USD_PATHS = {
    "box": "assets/box.usd",
    "bolt": "assets/bolt.usda",
    "light": "assets/lights.usda",
    "border_indices": "black_indices_expanded.npy"
}

class DigitSimulation:
    """Simulates a digit robot in Isaac Sim with cameras and a differential IK controller."""

    def __init__(self, args):
        """Initializes the DigitSimulation with camera orientation, border indices, and workspace limits.

        Args:
            args: Parsed command-line arguments
        """
        ori = [0, np.pi, -np.pi/2]
        ori = quat_from_euler_xyz(torch.tensor(ori[0]),torch.tensor(ori[1]),torch.tensor(ori[2]))
        ori = ori.detach().cpu().numpy()
        self.ori = (ori[0], ori[1], ori[2], ori[3])
        self.boder_ind = np.load(USD_PATHS["border_indices"])
        self.X_RANGE = (0.45, 0.55)
        self.Y_RANGE = (-0.1, 0.1)
        self.Z_RANGE = (0.2, 0.15)
        self.args = args

    def get_scene_cfg(self):
        """Defines the configuration for the interactive scene with robot, objects, lights, and cameras.

        Returns:
            DigitSceneCfg: Configured scene class
        """
        @configclass
        class DigitSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
            dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.75, 0.75, 0.75)))
            digit: ArticulationCfg = SAW_DIGIT_R_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            box = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Box",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=USD_PATHS["box"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                    scale=(0.5,1,1)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)))
            if not hasattr(box, "has_external_wrench"):
                box.has_external_wrench = False
            cube = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=USD_PATHS["bolt"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False),
                    scale=(30,30,30)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.48, 0, 0.035)))
            camera_l = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/finger_left_tip_body/front_cam",
                update_period=0.01,
                height=320,
                width=240,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=115, focus_distance=0.0015, horizontal_aperture=20.955, clipping_range=(0.1, 1e5)
                ),
                offset=CameraCfg.OffsetCfg(pos=(0.003,0,-0.1),rot=self.ori, convention="opengl"),
            )
            camera_r = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/finger_right_tip_body/front_cam",
                update_period=0.01,
                height=320,
                width=240,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=115, focus_distance=0.0015, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                ),
                offset=CameraCfg.OffsetCfg(pos=(0.003,0,-0.1),rot=self.ori, convention="opengl"),
            )
            camera_light_l = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Robot/finger_left_tip_body/L",
                spawn=sim_utils.UsdFileCfg(usd_path=USD_PATHS["light"]),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.006, 0, -0.003)))
            camera_light_r = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Robot/finger_right_tip_body/L",
                spawn=sim_utils.UsdFileCfg(usd_path=USD_PATHS["light"]),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.006, 0, -0.003)))

        return DigitSceneCfg(num_envs=self.args.num_envs, env_spacing=2.0)

    def process_depth(self, depth):
        """Processes and normalizes the depth image, masking out border pixels.

        Args:
            depth (np.ndarray): Depth image

        Returns:
            np.ndarray: Processed depth image
        """
        depth[self.boder_ind[:, 0], self.boder_ind[:, 1]] = 0
        normalized_depth = cv2.normalize(depth, None, 0, 70000, cv2.NORM_MINMAX)
        normalized_depth = normalized_depth.astype(np.uint8)
        return normalized_depth

    def process_rgb(self, rgb):
        """Processes RGB image by masking the border region.

        Args:
            rgb (np.ndarray): Raw RGB image

        Returns:
            np.ndarray: RGB image with white border
        """
        rgb = rgb[:, :, :3]
        rgb[self.boder_ind[:, 0], self.boder_ind[:, 1]] = [255, 255, 255]
        return rgb

    def run(self):
        """Runs the main simulation loop with the robot, gripper control, IK, and live camera visualization."""
        sim_cfg = sim_utils.SimulationCfg(dt=0.01)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
        sim_dt = sim.get_physics_dt()
        scene_cfg = self.get_scene_cfg()
        scene = InteractiveScene(scene_cfg)
        ENV_REGEX_NS = scene.env_regex_ns
        sim.reset()
        robot = scene["digit"]
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        robot_entity_cfg = SceneEntityCfg("digit", joint_names=["right_j[0-6]"], body_names=["right_hand"])
        gripper_cfg = SceneEntityCfg("digit", joint_names=["base_joint_gripper_left","base_joint_gripper_right"], body_names=["right_hand"])
        gripper_cfg.resolve(scene)
        robot_entity_cfg.resolve(scene)
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        ee_goals = [[0.5, 0, 0.5, 0, 0, 1, 0]]
        ee_goals = torch.tensor(ee_goals, device=sim.device)
        current_goal_idx = 0
        ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
        ik_commands[:] = ee_goals[current_goal_idx]
        i = 0
        count = 0
        while simulation_app.is_running():
            # Get RGB images from left and right finger cameras
            img_r = scene["camera_r"].data.output["rgb"][0].detach().cpu().numpy()
            img_r = cv2.normalize(img_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_l = scene["camera_l"].data.output["rgb"][0].detach().cpu().numpy()
            img_l = cv2.normalize(img_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if img_r.shape != img_l.shape:
                height = min(img_r.shape[0], img_l.shape[0])
                width = min(img_r.shape[1], img_l.shape[1])
                img_r = cv2.resize(img_r, (width, height))
                img_l = cv2.resize(img_l, (width, height))
            img_concat = np.hstack((img_l, img_r))
            cv2.imshow("Sensor Image", img_concat)
            if cv2.waitKey(1) == ord('q'):
                break
            if count % 450 == 0:
                count = 0
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
                ik_commands[:] = ee_goals[current_goal_idx]
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)
                current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            else:
                jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
                ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                root_pose_w = robot.data.root_state_w[:, 0:7]
                joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
                # Gripper sequence
                if count < 80:
                    gripper_close = torch.tensor([-0.05,0.05], device=sim.device).unsqueeze(0).repeat(scene.num_envs, 1)
                    robot.set_joint_position_target(gripper_close, joint_ids=gripper_cfg.joint_ids)
                elif count == 80:
                    diff_ik_controller.reset()
                    diff_ik_controller.set_command(torch.tensor([[0.5, 0, 0.17, 0, 0, 1, 0]]))
                elif 120 < count < 200:
                    gripper_open = torch.tensor([-0.008,0.008], device=sim.device).unsqueeze(0).repeat(scene.num_envs, 1)
                    robot.set_joint_position_target(gripper_open, joint_ids=gripper_cfg.joint_ids)
                elif 200 < count < 240:
                    gripper_close = torch.tensor([-0.05,0.05], device=sim.device).unsqueeze(0).repeat(scene.num_envs, 1)
                    robot.set_joint_position_target(gripper_close, joint_ids=gripper_cfg.joint_ids)
                elif 240 < count < 320:
                    gripper_open = torch.tensor([-0.008,0.008], device=sim.device).unsqueeze(0).repeat(scene.num_envs, 1)
                    robot.set_joint_position_target(gripper_open, joint_ids=gripper_cfg.joint_ids)
                elif 320 < count < 360:
                    gripper_close = torch.tensor([-0.05,0.05], device=sim.device).unsqueeze(0).repeat(scene.num_envs, 1)
                    robot.set_joint_position_target(gripper_close, joint_ids=gripper_cfg.joint_ids)
                elif 360 < count < 400:
                    gripper_open = torch.tensor([-0.008,0.008], device=sim.device).unsqueeze(0).repeat(scene.num_envs, 1)
                    robot.set_joint_position_target(gripper_open, joint_ids=gripper_cfg.joint_ids)
                elif count > 400:
                    diff_ik_controller.reset()
                    diff_ik_controller.set_command(torch.tensor([[0.5, 0, 0.3, 0, 0, 1, 0]]))
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            sim.step()
            count += 1
            scene.update(0.01)
            i += 1

if __name__ == "__main__":
    sim_runner = DigitSimulation(args_cli)
    sim_runner.run()
    simulation_app.close()
