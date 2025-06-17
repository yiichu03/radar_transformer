#! /usr/bin/env python3

# Copyright (C) 2024 Jan Michalczyk, Control of Networked Systems, University
# of Klagenfurt, Austria.
#
# All rights reserved.
#
# This software is licensed under the terms of the BSD-2-Clause-License with
# no commercial use allowed, the full terms of which are made available
# in the LICENSE file. No license in patents is granted.
#
# You can contact the author at <jan.michalczyk@aau.at>

import numpy as np
import rosbag
import utils
import visualization
import copy
from scipy.spatial.transform import Rotation as R

# In this script we implement a functionality to write to file
# the training examples and labels. They will be written into
# `data/{train|test}/pointclouds.hdf5` and `data/{train|test}/labels.hdf5`.
# TODO(jan): externalize parameters into a json file.

RADAR_TO_BASE_ROTATION = R.from_quat(
    [0.0, 0.0, 0.706825181105, 0.707388269167]).as_matrix()
RADAR_TO_BASE_TRANSLATION = np.array([[-0.145, 0.09, -0.025]])
IMU_TO_BASE_ROTATION = R.from_quat(
    [0.707105451466, 0.707108048814, 0.000209535067884, 0.00020953429822]).as_matrix()
IMU_TO_BASE_TRANSLATION = np.array([[0, 0, 0]])

RADAR_TO_IMU_ROTATION = np.dot(IMU_TO_BASE_ROTATION.transpose(), RADAR_TO_BASE_ROTATION)
RADAR_TO_IMU_TRANSLATION = np.dot(IMU_TO_BASE_ROTATION.transpose(), RADAR_TO_BASE_TRANSLATION.transpose()) - \
                        np.dot(IMU_TO_BASE_ROTATION.transpose(), IMU_TO_BASE_TRANSLATION.transpose())
RADAR_TO_IMU_TRANSLATION = RADAR_TO_IMU_TRANSLATION.transpose()

def main():
    print("Outputting data for training/testing into hdf5 ... .")

    mode_and_folders = {
        "train": "./input_train_files/", "test": "./input_test_files/"}
    mode_and_bags = {}
    for mode, folder in mode_and_folders.items():
        mode_and_bags[mode] = utils.get_bags_from_path(folder)

    # Get the maximum length of a pointcloud in the dataset.
    all_bags = utils.get_bags_from_paths(mode_and_folders.values())
    max_pc2_size_in_data = utils.get_max_pc2_size_in_data(all_bags)
    for mode, bags in mode_and_bags.items():
        with utils.DataFactory(max_pc2_size_in_data, mode) as data_factory:
            for bag in bags:
                with rosbag.Bag(bag, 'r') as databag:
                    pointclouds_as_xyzvi_mat_list = []
                    gt_time = np.empty(shape=(0, 1), dtype=np.float64)
                    radar_time = np.empty(shape=(0, 1), dtype=np.float64)
                    gt_position = np.empty(shape=(0, 3), dtype=np.float64)
                    gt_orientation = np.empty(shape=(0, 4), dtype=np.float64)
                    for topic, msg, _ in databag.read_messages():
                        if topic == "/twins_cns4/vrpn_client/raw_pose" and msg.pose:
                            gt_time = np.append(gt_time, np.asarray(
                                [msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs], dtype=np.float64)[np.newaxis, :], axis=0)
                            gt_position = np.append(gt_position, np.asarray(
                                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)[np.newaxis, :], axis=0)
                            gt_orientation = np.append(gt_orientation, np.asarray(
                                [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w], dtype=np.float32)[np.newaxis, :], axis=0)
                        if topic == "/ti_mmwave/radar_scan_pcl" and msg.data:
                            # Nx1
                            radar_time = np.append(radar_time, np.asarray(
                                [msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs], dtype=np.float64)[np.newaxis, :], axis=0)
                            pointclouds_as_xyzvi_mat_list.append(utils.decode_single_pc2_into_xyzvi(
                                msg))
                    # Normalize radar and gt time/position axis.
                    radar_time = np.squeeze(radar_time - radar_time[0])
                    gt_time = np.squeeze(gt_time - gt_time[0])
                    # Bring poses to the origin.
                    gt_position = gt_position - gt_position[0]
                    gt_orientation = R.from_matrix(np.matmul(R(gt_orientation[0]).as_matrix(
                    ).transpose(), R(gt_orientation).as_matrix())).as_quat()
                    # Filter to FOV of the radar.
                    filtered_pointclouds_as_xyzvi_mat_list = utils.filter_to_fov(
                        pointclouds_as_xyzvi_mat_list)
                    # Apply transform IMU-RADAR to the measurements.
                    pointclouds_in_imu_frame_mat_list = utils.apply_radar_imu_transform(
                        RADAR_TO_IMU_ROTATION, RADAR_TO_IMU_TRANSLATION, filtered_pointclouds_as_xyzvi_mat_list)
                    # Construct the the ground truth and input data and write hdf5.
                    data_factory.generate_gt_and_input_data(
                        gt_time, gt_position, gt_orientation, radar_time, pointclouds_in_imu_frame_mat_list)


if __name__ == '__main__':
    main()
