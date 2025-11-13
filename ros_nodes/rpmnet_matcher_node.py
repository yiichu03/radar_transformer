#! /usr/bin/env python3

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
from geometry_msgs.msg import Pose, Quaternion
from radar_deep_matcher.msg import PointcloudInput, RpmnetRelativePose
from sensor_msgs.msg import PointCloud2

try:
    import open3d as o3d  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("open3d is required for RPMNet inference") from exc

RPMNET_ROOT = pathlib.Path(__file__).resolve().parents[2] / "RPM-Net_snail"
if not RPMNET_ROOT.exists():
    raise RuntimeError(f"RPMNet repo not found at {RPMNET_ROOT}")

sys.path.append(str(RPMNET_ROOT / "src"))
sys.path.append(str(RPMNET_ROOT / "snail_test"))

from arguments import rpmnet_eval_arguments  # type: ignore  # noqa: E402
import models.rpmnet as rpmnet_mod  # type: ignore  # noqa: E402
from tools import estimate_normals_for_radar  # type: ignore  # noqa: E402

PC2_TOPIC_NAME = "/mmWaveDataHdl/RScan"
MATCHINGS_TOPIC_NAME = "/rpmnet/pointcloud_input"
POSE_TOPIC_NAME = "/rpmnet/relative_pose"
DEFAULT_TARGET_POINTS = 1024 # 1024
DEFAULT_RANDOM_SEED = 42


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array).float().to(device)


def quaternion_from_matrix(matrix: np.ndarray) -> Quaternion:
    rot = matrix[:3, :3]
    qw = np.sqrt(1 + np.trace(rot)) / 2.0
    qx = (rot[2, 1] - rot[1, 2]) / (4 * qw)
    qy = (rot[0, 2] - rot[2, 0]) / (4 * qw)
    qz = (rot[1, 0] - rot[0, 1]) / (4 * qw)
    q = Quaternion()
    q.x, q.y, q.z, q.w = qx, qy, qz, qw
    return q


def matrix_from_transform(transform: torch.Tensor) -> np.ndarray:
    if isinstance(transform, torch.Tensor):
        transform = transform.detach().cpu().numpy()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = transform[:3, :3]
    T[:3, 3] = transform[:3, 3]
    return T


def flatten_matches(src_points: np.ndarray, ref_points: np.ndarray,
                    assignment: np.ndarray, weights: np.ndarray,
                    min_weight: float, max_count: int) -> np.ndarray:
    valid_idx = np.where((assignment >= 0) & (weights >= min_weight))[0]
    if valid_idx.size == 0:
        return np.empty(0, dtype=np.float64)
    matches = []
    for idx in valid_idx[:max_count]:
        dst_idx = assignment[idx]
        matches.append(np.concatenate([src_points[idx], ref_points[dst_idx]], axis=0))
    if not matches:
        return np.empty(0, dtype=np.float64)
    return np.asarray(matches, dtype=np.float64).reshape(-1)


def read_pointcloud(msg: PointCloud2) -> np.ndarray:
    points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    return points.astype(np.float32)


def downsample(points: np.ndarray, target: int) -> np.ndarray:
    # Return original points without resampling.
    return points


def prepare_points(points: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        return empty_xyz, np.zeros((0, 6), dtype=np.float32)
    sampled = downsample(points, target)
    normals, _ = estimate_normals_for_radar(sampled, k=20)
    valid = np.linalg.norm(normals, axis=1) > 1e-6
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    stacked = np.concatenate([sampled, normals], axis=1)
    return sampled, stacked


@dataclass
class PreparedScan:
    stamp: rospy.Time
    raw_msg: PointCloud2
    xyz: np.ndarray
    xyz_normals: np.ndarray


class RpmnetMatcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self._seed = DEFAULT_RANDOM_SEED
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self._seed)
        parser = rpmnet_eval_arguments()
        self.model_args = parser.parse_args([])
        self.model_args.resume = args.checkpoint
        self.model_args.radius = args.radius
        self.model_args.num_neighbors = args.num_neighbors
        self.model_args.num_reg_iter = args.num_iter
        self.model = self._load_model(self.model_args)
        self.prev_scan: Optional[PreparedScan] = None
        self.target_points = args.target_points

    def _load_model(self, model_args):
        model = rpmnet_mod.get_model(model_args)
        model.to(self.device)
        model.eval()
        state = torch.load(model_args.resume, map_location="cpu")
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=True)
        return model

    def process(self, msg: PointCloud2):
        points = read_pointcloud(msg)
        if points.size == 0:
            return None, None
        xyz, xyz6 = prepare_points(points, self.target_points)
        scan = PreparedScan(msg.header.stamp, msg, xyz, xyz6)
        if self.prev_scan is None:
            self.prev_scan = scan
            return None, None
        result = self._run_rpmnet(self.prev_scan, scan)
        self.prev_scan = scan
        return result

    def _run_rpmnet(self, src_scan: PreparedScan, ref_scan: PreparedScan):
        batch = {
            'points_src': to_tensor(src_scan.xyz_normals[None, ...], self.device),
            'points_ref': to_tensor(ref_scan.xyz_normals[None, ...], self.device)
        }
        with torch.no_grad():
            transforms, endpoints = self.model(batch, self.args.num_iter)
        transform = transforms[-1][0] if isinstance(transforms, list) else transforms[0, -1]
        T = matrix_from_transform(transform)

        matches_msg = None
        pose_msg = None
        if not self.args.pose_only:
            perm = endpoints['perm_matrices'][-1][0]
            perm_np = perm.detach().cpu().numpy()
            weights = perm_np.sum(axis=1)
            weighted_ref = endpoints['weighted_ref'][-1][0].detach().cpu().numpy()
            matches_list = []
            max_count = min(self.args.max_matches, src_scan.xyz.shape[0])
            for idx in range(max_count):
                weight_val = float(weights[idx]) if idx < weights.shape[0] else 0.0
                if weight_val < self.args.match_threshold:
                    continue
                prev_pt = src_scan.xyz[idx]
                curr_pt = weighted_ref[idx]
                if np.isnan(curr_pt).any():
                    continue
                matches_list.append(np.concatenate([prev_pt, curr_pt, [weight_val]], axis=0))

            if matches_list:
                flattened = np.asarray(matches_list, dtype=np.float64).reshape(-1)
                matches_msg = PointcloudInput()
                matches_msg.header = ref_scan.raw_msg.header
                matches_msg.has_matches = True
                matches_msg.matches_flattened = flattened
                matches_msg.current_pointcloud = ref_scan.raw_msg
            else:
                matches_msg = PointcloudInput()
                matches_msg.header = ref_scan.raw_msg.header
                matches_msg.has_matches = False
                matches_msg.current_pointcloud = ref_scan.raw_msg

        if self.args.pose_only or self.args.publish_pose:
            pose_msg = RpmnetRelativePose()
            pose_msg.header = ref_scan.raw_msg.header
            pose_msg.previous_stamp = src_scan.stamp
            pose_msg.pose.position.x = T[0, 3]
            pose_msg.pose.position.y = T[1, 3]
            pose_msg.pose.position.z = T[2, 3]
            quat = quaternion_from_matrix(T)
            pose_msg.pose.orientation = quat
            cov = np.zeros((6, 6), dtype=np.float64)
            cov[:3, :3] = np.eye(3) * (self.args.pose_cov_trans ** 2)
            cov[3:, 3:] = np.eye(3) * (self.args.pose_cov_rot ** 2)
            pose_msg.covariance = cov.reshape(-1)
            pose_msg.has_covariance = True

        return matches_msg, pose_msg


def run_live(args):
    rospy.init_node("rpmnet_matcher", anonymous=False)
    matcher = RpmnetMatcher(args)
    matches_pub = rospy.Publisher(MATCHINGS_TOPIC_NAME, PointcloudInput, queue_size=5)
    pose_pub = rospy.Publisher(POSE_TOPIC_NAME, RpmnetRelativePose, queue_size=5)

    def callback(msg: PointCloud2):
        matches_msg, pose_msg = matcher.process(msg)
        if matches_msg is not None:
            matches_pub.publish(matches_msg)
        if pose_msg is not None:
            pose_pub.publish(pose_msg)

    rospy.Subscriber(args.topic, PointCloud2, callback, queue_size=1)
    rospy.loginfo("RPMNet live matcher running on %s", args.topic)
    rospy.spin()


def parse_args():
    parser = argparse.ArgumentParser(description="RPMNet matcher node")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to RPMNet checkpoint (.pth)")
    parser.add_argument("--topic", type=str, default=PC2_TOPIC_NAME,
                        help="Radar PointCloud2 topic")
    parser.add_argument("--target_points", type=int, default=DEFAULT_TARGET_POINTS)
    parser.add_argument("--max_matches", type=int, default=300)
    parser.add_argument("--match_threshold", type=float, default=0.02)
    parser.add_argument("--num_iter", type=int, default=5)
    parser.add_argument("--num_neighbors", type=int, default=20) # 64
    parser.add_argument("--radius", type=float, default=2) # 
    parser.add_argument("--pose_only", action="store_true",
                        help="Only publish transform measurements")
    parser.add_argument("--publish_pose", action="store_true",
                        help="Always publish transform measurements along with matches")
    parser.add_argument("--pose_cov_trans", type=float, default=0.05)
    parser.add_argument("--pose_cov_rot", type=float, default=np.deg2rad(3.0))
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


if __name__ == "__main__":
    cli_args = parse_args()
    run_live(cli_args)
