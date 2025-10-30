#! /usr/bin/env python3
import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
from geometry_msgs.msg import Pose, Quaternion
from radar_deep_matcher.msg import PointcloudInput, RpmnetRelativePose
from sensor_msgs.msg import PointCloud2
from scipy.optimize import linear_sum_assignment


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
DEFAULT_TARGET_POINTS = 1024

PACKAGE_ROOT_FOLDER = "radar_transformer"

def get_root_folder():
    current_file_path = pathlib.Path(__file__).resolve()
    package_root_index = current_file_path.parts.index(PACKAGE_ROOT_FOLDER)
    return pathlib.Path(*current_file_path.parts[:package_root_index + 1])

root_folder = get_root_folder()
sys.path.append(root_folder.parent.as_posix())
sys.path.append(root_folder.as_posix())
from radar_transformer import utils as rt_utils  # noqa: E402
from radar_transformer import prepare_dataset as rt_prepare  # noqa: E402


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
                    min_weight: float, max_count: int,
                    top_k: bool = False,
                    fov_filter: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None) -> np.ndarray:
    valid_idx = np.where((assignment >= 0) & (weights >= min_weight))[0]
    if valid_idx.size == 0:
        return np.empty(0, dtype=np.float64)
    if top_k:
        order = np.argsort(weights[valid_idx])[::-1]
        valid_idx = valid_idx[order]
    matches = []
    for idx in valid_idx[:max_count]:
        dst_idx = assignment[idx]
        if fov_filter is not None and not fov_filter(src_points[idx], ref_points[dst_idx]):
            continue
        matches.append(np.concatenate([src_points[idx], ref_points[dst_idx]], axis=0))
    if not matches:
        return np.empty(0, dtype=np.float64)
    return np.asarray(matches, dtype=np.float64).reshape(-1)


def read_pointcloud(msg: PointCloud2) -> np.ndarray:
    points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    return points.astype(np.float32)


def downsample(points: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    if n == 0:
        return np.zeros((target, 3), dtype=np.float32), np.zeros(target, dtype=np.int64)
    if n >= target:
        indices = np.random.choice(n, target, replace=False)
    else:
        pad = target - n
        pad_indices = np.random.choice(n, pad, replace=True)
        indices = np.concatenate([np.arange(n), pad_indices])
    return points[indices], indices


def prepare_points(radar_points: np.ndarray, imu_points: np.ndarray,
                   target: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_imu, indices = downsample(imu_points, target)
    if radar_points.shape[0] == 0:
        sampled_radar = np.zeros((target, 3), dtype=np.float32)
    else:
        sampled_radar = radar_points[indices]
    normals, _ = estimate_normals_for_radar(sampled_imu, k=20)
    valid = np.linalg.norm(normals, axis=1) > 1e-6
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    stacked = np.concatenate([sampled_imu, normals], axis=1)
    return sampled_radar.astype(np.float32), sampled_imu.astype(np.float32), stacked


@dataclass
class PreparedScan:
    stamp: rospy.Time
    raw_msg: PointCloud2
    xyz: np.ndarray  # radar frame points fed to matcher output
    imu_xyz: np.ndarray
    xyz_normals: np.ndarray


class RpmnetMatcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.match_passes = max(1, args.match_passes)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
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
        radar_points = read_pointcloud(msg)
        if radar_points.size == 0:
            return None, None
        imu_points = rt_utils.apply_radar_imu_transform_to_single_pointcloud(
            rt_prepare.RADAR_TO_IMU_ROTATION,
            rt_prepare.RADAR_TO_IMU_TRANSLATION,
            radar_points).astype(np.float32)
        sampled_radar, sampled_imu, sampled_with_normals = prepare_points(
            radar_points.astype(np.float32), imu_points, self.target_points)
        scan = PreparedScan(msg.header.stamp, msg,
                            sampled_radar,
                            sampled_imu,
                            sampled_with_normals)
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
            perm_seq = endpoints['perm_matrices']
            perm_init_seq = endpoints.get('perm_matrices_init', perm_seq)
            num_passes = min(self.match_passes, len(perm_seq))
            perm_tensors = [perm_seq[-(i + 1)][0] for i in range(num_passes)]
            perm_init_tensors = [perm_init_seq[-(i + 1)][0] for i in range(num_passes)]
            perm_np = np.mean([p.detach().cpu().numpy() for p in perm_tensors], axis=0)
            perm_pre_np = np.mean([p.detach().cpu().numpy() for p in perm_init_tensors], axis=0)
            if self.args.hungarian_matches:
                cost = -perm_pre_np
                row_ind, col_ind = linear_sum_assignment(cost)
                selected_vals = perm_pre_np[row_ind, col_ind] if row_ind.size else np.array([], dtype=np.float64)
                max_val = selected_vals.max() if selected_vals.size else 0.0
                threshold = max_val / 1.3 if max_val > 0 else 0.0
                keep_mask = selected_vals > threshold
                row_ind = row_ind[keep_mask]
                col_ind = col_ind[keep_mask]
                assignment = -np.ones(perm_np.shape[0], dtype=int)
                assignment[row_ind] = col_ind
                weights = np.zeros(perm_np.shape[0], dtype=np.float64)
                weights[row_ind] = perm_pre_np[row_ind, col_ind]
                flattened = flatten_matches(
                    src_scan.xyz, ref_scan.xyz,
                    assignment, weights,
                    self.args.match_threshold, self.args.max_matches,
                    top_k=self.args.topk_matches,
                    fov_filter=lambda prev_pt, curr_pt: rt_utils.is_in_fov(
                        np.vstack((prev_pt, curr_pt))))
            else:
                assignment = np.argmax(perm_np, axis=1)
                weights = perm_np[np.arange(perm_np.shape[0]), assignment]
                flattened = flatten_matches(
                    src_scan.xyz, ref_scan.xyz, assignment, weights,
                    self.args.match_threshold, self.args.max_matches,
                    top_k=self.args.topk_matches,
                    fov_filter=lambda prev_pt, curr_pt: rt_utils.is_in_fov(
                        np.vstack((prev_pt, curr_pt))))
            if flattened.size > 0:
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
    args.match_passes = rospy.get_param("~match_passes", args.match_passes)
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
    parser.add_argument("--match_passes", type=int, default=1,
                        help="Number of RPMNet permutation iterations to aggregate")
    parser.add_argument("--hungarian_matches", action="store_true",
                        help="Use Hungarian assignment for correspondences")
    parser.add_argument("--topk_matches", action="store_true",
                        help="Select top-K matches by weight before truncation")
    parser.add_argument("--num_iter", type=int, default=5)
    parser.add_argument("--num_neighbors", type=int, default=64)
    parser.add_argument("--radius", type=float, default=0.3)
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
