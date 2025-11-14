#!/usr/bin/env python3

# Copyright (C) 2024
# All rights reserved.
#
# Live Radar Transformer matcher node for ROS1. Subscribes to PointCloud2
# messages, runs the Radar Transformer network, and publishes
# radar_deep_matcher/PointcloudInput messages on /pointcloud_input.

import pathlib
import sys

import rospy
import torch
from radar_deep_matcher.msg import PointcloudInput
from sensor_msgs.msg import PointCloud2

PACKAGE_ROOT_FOLDER = "radar_transformer"


def _get_root_folder() -> pathlib.Path:
    current_file_path = pathlib.Path(__file__).resolve()
    package_root_index = current_file_path.parts.index(PACKAGE_ROOT_FOLDER)
    return pathlib.Path(*current_file_path.parts[:package_root_index + 1])


ROOT_FOLDER = _get_root_folder()
if ROOT_FOLDER.as_posix() not in sys.path:
    sys.path.append(ROOT_FOLDER.as_posix())

from ros_nodes.pc_matcher_node import (  # type: ignore  # noqa: E402
    MATCHINGS_TOPIC_NAME,
    MODEL_DIR,
    MODEL_FILE,
    PC2_TOPIC_NAME,
    PointcloudPreprocessor,
    get_root_folder,
)
import transformer_models  # type: ignore  # noqa: E402


class LiveMatcherNode:
    """Online ROS node that publishes Radar Transformer correspondences."""

    def __init__(self) -> None:
        rospy.init_node("radar_transformer_live_matcher")
        max_pc2_size = rospy.get_param("~max_pc2_size_in_data", 121)

        self.preprocessor = PointcloudPreprocessor(
            input_bag_folders=None, max_pc2_size_in_data=max_pc2_size
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_length = 2 * (3 * self.preprocessor.max_pc2_size_in_data + 3)

        model_filepath = get_root_folder() / MODEL_DIR / MODEL_FILE
        self.model = transformer_models.RadarDeepMatcher(input_length)
        self.model.eval()
        state_dict = torch.load(
            model_filepath, weights_only=True, map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        self.pub_matches = rospy.Publisher(
            MATCHINGS_TOPIC_NAME, PointcloudInput, queue_size=10
        )
        self.sub_pc2 = rospy.Subscriber(
            PC2_TOPIC_NAME, PointCloud2, self.pc2_callback, queue_size=1
        )
        self.frame_index = 0

    def pc2_callback(self, msg: PointCloud2) -> None:
        pointcloud_input_msg = PointcloudInput()
        pointcloud_input_msg.header = msg.header
        pointcloud_input_msg.has_matches = False
        pointcloud_input_msg.current_pointcloud = msg

        matches_used = 0
        model_input, input_non_empty = self.preprocessor.build_input_to_model(msg)
        if input_non_empty and model_input is not None:
            input_tensor = torch.from_numpy(model_input).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_tensor)
            prediction_cpu = prediction.cpu()
            input_tensor_cpu = input_tensor.cpu()
            matches, pc1_size, pc2_size = self.preprocessor.get_matches_from_prediction(
                prediction_cpu[0, :, :], input_tensor_cpu
            )
            self.preprocessor.build_output_msg_from_prediction(
                matches,
                input_tensor_cpu,
                pointcloud_input_msg,
                self.frame_index,
                prediction_cpu[0, :, :],
                pc1_size,
                pc2_size,
            )
            matches_used = len(matches)
        self.frame_index += 1
        self.pub_matches.publish(pointcloud_input_msg)
        if self.frame_index % 50 == 0:
            rospy.loginfo(
                "[transformer_live] processed %d frames, matches in last frame: %d (has_matches=%s)",
                self.frame_index,
                matches_used,
                pointcloud_input_msg.has_matches,
            )


def main() -> None:
    LiveMatcherNode()
    rospy.loginfo("radar_transformer_live_matcher node started")
    rospy.spin()


if __name__ == "__main__":
    main()
