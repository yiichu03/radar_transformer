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

import statistics

import numpy as np
import sys
import pathlib
import rosbag
import sensor_msgs.point_cloud2 as pc2
from radar_deep_matcher.msg import PointcloudInput
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import ConnectionPatch
import torch
import time
from matplotlib.ticker import MaxNLocator
torch.set_printoptions(precision=20)
torch.set_default_dtype(torch.float32)


PC2_TOPIC_NAME = "/mmWaveDataHdl/RScan"
MATCHINGS_TOPIC_NAME = "/pointcloud_input"
EVAL_BAG = "edgar_classroom_run0.bag"
PACKAGE_ROOT_FOLDER = "radar_transformer"
MODEL_DIR = "saved_models"
MODEL_FILE = "model.ptm"


def get_root_folder():
    current_file_path = pathlib.Path(__file__).resolve()
    package_root_index = current_file_path.parts.index(PACKAGE_ROOT_FOLDER)
    return pathlib.Path(*current_file_path.parts[:package_root_index + 1])


sys.path.append(get_root_folder().as_posix())
import utils  # nopep8
import transformer_models  # nopep8
import evaluate_matchings  # nopep8
import prepare_dataset  # nopep8


def get_input_file_folder(mode):
    file_folder = "input_" + mode + "_files"
    return get_root_folder() / file_folder


def plot_flattened_matches(flattened_matches, pred, pc1_size, pc2_size, matches_dict):
    prediction = torch.transpose(
        pred, 0, 1).detach().numpy()
    # Unflatten matches.
    matches = np.reshape(flattened_matches, (-1, 6))
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(
        r"Correspondences between point clouds $i$ and $i+1$ in x-y plane", fontsize=50)
    # set up the axes for the first plot
    ax_1 = fig.add_subplot(1, 2, 1)
    ax_1.set_xlim(left=0, right=15)
    ax_1.set_ylim(bottom=-5, top=5)
    # set up the axes for the second plot
    ax_2 = fig.add_subplot(
        1, 2, 2, sharey=ax_1, sharex=ax_1)
    # Draw pointclouds (x, y).
    ax_1.plot(matches[:, 0],
              matches[:, 1],
              'ko', markersize=20)
    ax_2.plot(matches[:, 3],
              matches[:, 4],
              'ko', markersize=20)
    ax_1.set_xlabel('x [m]', fontsize=50)
    ax_1.set_ylabel('y [m]', fontsize=50)
    ax_2.set_xlabel('x [m]', fontsize=50)
    ax_2.set_ylabel('y [m]', fontsize=50)
    ax_2.yaxis.set_label_position("right")
    ax_1.set_title(r"$i$", fontsize=50)
    ax_2.set_title(r"$i+1$", fontsize=50)
    ax_1.xaxis.set_minor_locator(AutoMinorLocator())
    ax_1.yaxis.set_minor_locator(AutoMinorLocator())
    ax_1.tick_params(axis='both', labelsize=50)
    ax_2.xaxis.set_minor_locator(AutoMinorLocator())
    ax_2.yaxis.set_minor_locator(AutoMinorLocator())
    ax_2.tick_params(axis='both', labelsize=50)
    ax_1.grid(True)
    ax_2.grid(True)
    # xz plot.
    fig1 = plt.figure(figsize=(10, 5))
    fig1.suptitle(
        r"Correspondences between point clouds $i$ and $i+1$ in x-z plane", fontsize=50)
    # set up the axes for the first plot
    ax_12 = fig1.add_subplot(1, 2, 1)
    ax_12.set_xlim(left=0, right=15)
    ax_12.set_ylim(bottom=-7.5, top=7.5)
    # set up the axes for the second plot
    ax_22 = fig1.add_subplot(
        1, 2, 2, sharey=ax_12, sharex=ax_12)
    # Draw pointclouds (x, y).
    ax_12.plot(matches[:, 0],
               matches[:, 2],
               'ko', markersize=20)
    ax_22.plot(matches[:, 3],
               matches[:, 5],
               'ko', markersize=20)
    ax_12.set_xlabel('x [m]', fontsize=50)
    ax_12.set_ylabel('z [m]', fontsize=50)
    ax_22.set_xlabel('x [m]', fontsize=50)
    ax_22.set_ylabel('z [m]', fontsize=50)
    ax_22.yaxis.set_label_position("right")
    ax_12.set_title(r"$i$", fontsize=50)
    ax_22.set_title(r"$i + 1$", fontsize=50)
    ax_12.xaxis.set_minor_locator(AutoMinorLocator())
    ax_12.yaxis.set_minor_locator(AutoMinorLocator())
    ax_12.tick_params(axis='both', labelsize=50)
    ax_22.xaxis.set_minor_locator(AutoMinorLocator())
    ax_22.yaxis.set_minor_locator(AutoMinorLocator())
    ax_22.tick_params(axis='both', labelsize=50)
    ax_12.grid(True)
    ax_22.grid(True)
    # matrix
    fig_h, ax_h = plt.subplots()
    fig_h.suptitle('Correspondence likelihood', fontsize=50)
    im = ax_h.imshow(prediction[1:pc1_size, 1:pc2_size],
                     cmap='jet', interpolation='nearest')
    ax_h.set_xlabel(r"Point indices in point cloud $i+1$", fontsize=50)
    ax_h.set_ylabel(r"Point indices in point cloud $i$", fontsize=50)
    ax_h.tick_params(axis='both', labelsize=50)
    ax_h.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_h.xaxis.set_major_locator(MaxNLocator(integer=True))
    cbar = ax_h.figure.colorbar(im, ax=ax_h)
    cbar.ax.tick_params(labelsize=50)
    # Full likelihood matrix padded - uncomment below.
    # fig1_h, ax1_h = plt.subplots()
    # fig1_h.suptitle('Correspondence likelihood', fontsize=50)
    # im1 = ax1_h.imshow(prediction,
    #                   cmap='jet', interpolation='nearest')
    # ax1_h.set_xlabel(r"Point indices in padded point cloud $i+1$", fontsize=50)
    # ax1_h.set_ylabel(r"Point indices in padded point cloud $i$", fontsize=50)
    # ax1_h.tick_params(axis='both', labelsize=50)
    # ax1_h.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax1_h.xaxis.set_major_locator(MaxNLocator(integer=True))
    # cbar1 = ax1_h.figure.colorbar(im, ax=ax1_h)
    # cbar1.ax.tick_params(labelsize=50)
    m_list = list(matches_dict.items())
    print(matches_dict)
    # annotations positions
    a = []
    for i in range(1000):
        length = np.random.uniform(90, 150)
        angle = np.pi * np.random.uniform(0, 100)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        a.append([x, y])

    for idx, row in enumerate(matches):
        # xy
        xy_ax_2 = (row[3],
                   row[4])
        xy_ax_1 = (row[0],
                   row[1])
        connection = ConnectionPatch(xyA=xy_ax_1, xyB=xy_ax_2, coordsA="data", coordsB="data",
                                     axesA=ax_1, axesB=ax_2, color="green")
        coords = a[int(np.random.uniform(
            0, 1000))]
        ax_1.annotate(str(idx),
                      xy=xy_ax_1, xycoords='data',
                      xytext=coords, textcoords='offset points',
                      arrowprops=dict(facecolor='black', width=1, shrink=0.1), fontsize=45)

        ax_2.add_artist(connection)
        ax_1.plot(row[0],
                  row[1], 'ro', markersize=25)
        ax_2.plot(row[3],
                  row[4], 'ro', markersize=25)
        # xz
        xz_ax_2 = (row[3],
                   row[5])
        xz_ax_1 = (row[0],
                   row[2])
        connection = ConnectionPatch(xyA=xz_ax_1, xyB=xz_ax_2, coordsA="data", coordsB="data",
                                     axesA=ax_12, axesB=ax_22, color="green")

        ax_12.annotate(str(idx),
                       xy=xz_ax_1, xycoords='data',
                       xytext=coords, textcoords='offset points',
                       arrowprops=dict(facecolor='black', width=1, shrink=0.1), fontsize=45)

        ax_22.add_artist(connection)

        ax_12.plot(row[0],
                   row[2], 'ro', markersize=25)
        ax_22.plot(row[3],
                   row[5], 'ro', markersize=25)

        # arrows
        ax_h.annotate(str(idx), fontsize=50, xy=(m_list[idx][1], m_list[idx][0]),
                      xycoords='data', xytext=(100, -6),
                      textcoords='offset points',
                      arrowprops=dict(arrowstyle="->",
                                      linewidth=5.,
                                      color='black')
                      )

    plt.draw()
    plt.show(block=True)


def plot_matches(matches_dict, model_input):
    model_input = model_input.reshape(
        (model_input.size()[1] // 3, 3)).detach().numpy()
    input_pc_1 = model_input[:(model_input.shape[0] // 2), :]
    input_pc_2 = model_input[(model_input.shape[0] // 2):, :]
    pc1_size, pc2_size = evaluate_matchings.calculate_non_padded_pc_sizes(
        model_input, model_input.shape[0] // 2)
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('Matched points between radar scans', fontsize=50)
    # set up the axes for the first plot
    ax_1 = fig.add_subplot(1, 2, 1)
    ax_1.set_xlim(left=0, right=15)
    ax_1.set_ylim(bottom=-15, top=15)
    # set up the axes for the second plot
    ax_2 = fig.add_subplot(
        1, 2, 2, sharey=ax_1, sharex=ax_1)
    # Draw pointclouds (x, y).
    ax_1.plot(input_pc_1[1:pc1_size, 0],
              input_pc_1[1:pc1_size, 1],
              'ko', markersize=10)
    ax_2.plot(input_pc_2[1:pc2_size, 0],
              input_pc_2[1:pc2_size, 1],
              'ko', markersize=10)
    ax_1.set_xlabel('x [m]', fontsize=50)
    ax_1.set_ylabel('y [m]', fontsize=50)
    ax_2.set_xlabel('x [m]', fontsize=50)
    ax_2.set_ylabel('y [m]', fontsize=50)
    ax_2.yaxis.set_label_position("right")
    ax_1.set_title('Previous', fontsize=50)
    ax_2.set_title('Current', fontsize=50)
    ax_1.xaxis.set_minor_locator(AutoMinorLocator())
    ax_1.yaxis.set_minor_locator(AutoMinorLocator())
    ax_1.tick_params(axis='both', labelsize=40)
    ax_2.xaxis.set_minor_locator(AutoMinorLocator())
    ax_2.yaxis.set_minor_locator(AutoMinorLocator())
    ax_2.tick_params(axis='both', labelsize=40)
    ax_1.grid(True)
    ax_2.grid(True)
    for pc1_idx, pc2_idx in matches_dict.items():
        xy_ax_2 = (input_pc_2[pc2_idx + 1, 0],
                   input_pc_2[pc2_idx + 1, 1])
        xy_ax_1 = (input_pc_1[pc1_idx + 1, 0],
                   input_pc_1[pc1_idx + 1, 1])
        connection = ConnectionPatch(xyA=xy_ax_1, xyB=xy_ax_2, coordsA="data", coordsB="data",
                                     axesA=ax_1, axesB=ax_2, color="green")
        ax_2.add_artist(connection)
        ax_1.plot(input_pc_1[pc1_idx + 1, 0],
                  input_pc_1[pc1_idx + 1, 1], 'ro', markersize=10)
        ax_2.plot(input_pc_2[pc2_idx + 1, 0],
                  input_pc_2[pc2_idx + 1, 1], 'ro', markersize=10)

    plt.draw()
    plt.show(block=True)


class PointcloudPreprocessor:
    def __init__(self, input_bag_folders=None, max_pc2_size_in_data=121):
        self.is_previous_pc2_nonempty = False
        if input_bag_folders is not None:
            # Get the maximum length of a pointcloud in the dataset.
            _ = utils.get_bags_from_paths(input_bag_folders)
            # self.max_pc2_size_in_data = utils.get_max_pc2_size_in_data(all_bags)
            self.max_pc2_size_in_data = max_pc2_size_in_data
        else:
            self.max_pc2_size_in_data = max_pc2_size_in_data

    def build_input_to_model(self, current_pc2):
        current_pc2_as_mat = \
            utils.apply_radar_imu_transform_to_single_pointcloud(prepare_dataset.RADAR_TO_IMU_ROTATION,
                                                                 prepare_dataset.RADAR_TO_IMU_TRANSLATION,
                                                                 utils.decode_single_pc2_into_xyzvi(current_pc2)).astype(np.float32)
        if not self.is_previous_pc2_nonempty:
            pointcloud_input = None
            input_non_empty = False
            self.is_previous_pc2_nonempty = True
        else:
            pointcloud_input = utils.build_input_to_model(
                self.previous_pc2_as_mat, current_pc2_as_mat, self.max_pc2_size_in_data)
            input_non_empty = True
        self.previous_pc2_as_mat = current_pc2_as_mat
        return pointcloud_input, input_non_empty

    def get_matches_from_prediction(self, prediction, model_input):
        # TODO: include some basic outlier removal here.
        prediction = torch.transpose(
            prediction, 0, 1).detach().numpy()
        model_input = model_input.reshape(
            (model_input.size()[1] // 3, 3)).detach().numpy()
        pc1_size, pc2_size = evaluate_matchings.calculate_non_padded_pc_sizes(
            model_input, model_input.shape[0] // 2)
        # Run LSA on the calculated matrix.
        row_ind, col_ind = linear_sum_assignment(
            prediction[1:pc1_size, 1:pc2_size], maximize=True)
        # Take the max for thresholding. Higher the thresholds, more outliers.
        threshold = np.amax(prediction[1:pc1_size, 1:pc2_size]) / 1.3
        # Matches are such that: matches = {point_in_pc1: point_in_pc2}
        matches = {}
        for idx in range(row_ind.shape[0]):
            if prediction[row_ind[idx] + 1, col_ind[idx] + 1] > threshold:
                matches[row_ind[idx]] = col_ind[idx]
        return matches, pc1_size, pc2_size

    def build_output_msg_from_prediction(self, matches_dict, model_input, pointcloud_input_msg, ind, pred, pc1_size, pc2_size):
        # Output the matches as an array where indices are indices of points in pc1
        # (previous) and values are indexes in pc2 (current).
        if matches_dict:
            model_input_as_np = model_input.reshape(
                (model_input.size()[1] // 3, 3)).detach().numpy()
            model_input_as_np = utils.apply_inv_radar_imu_transform_to_single_pointcloud(prepare_dataset.RADAR_TO_IMU_ROTATION,
                                                                                         prepare_dataset.RADAR_TO_IMU_TRANSLATION,
                                                                                         model_input_as_np)
            half_size = model_input_as_np.shape[0] // 2
            # First three entries previous pc, last current pc.
            matches_array = np.empty(shape=(0, 6), dtype=np.float64)
            matches_vis_dict = {}
            for pc1_idx, pc2_idx in matches_dict.items():
                is_in_fov = utils.is_in_fov(
                    np.vstack((model_input_as_np[pc1_idx + 1, :], model_input_as_np[half_size + pc2_idx + 1, :])))
                if is_in_fov:
                    matches_array = np.append(matches_array,
                                              np.concatenate((model_input_as_np[pc1_idx + 1, :],
                                                              model_input_as_np[half_size + pc2_idx + 1, :]),
                                                             axis=None)[np.newaxis, :], axis=0)
                    matches_vis_dict[pc1_idx] = pc2_idx
            matches_array = matches_array.flatten(order='C')
            # Uncomment below to plot matches.
            # if ind % 10 == 0:
            #    # if ind == 410:
            #    plot_flattened_matches(
            #        matches_array, pred, pc1_size, pc2_size, matches_vis_dict)
            pointcloud_input_msg.matches_flattened = matches_array
            pointcloud_input_msg.has_matches = True


def main():
    input_filepath = get_input_file_folder("test") / EVAL_BAG
    input_bag = rosbag.Bag(input_filepath)
    output_bag = rosbag.Bag(EVAL_BAG.split(
        '.')[0] + "_DL_replayed" + ".bag", 'w')

    input_bag_folders = []
    for mode in ["train", "test"]:
        input_bag_folders.append(get_input_file_folder(mode))

    pointcloud_preprocessor = PointcloudPreprocessor(input_bag_folders)
    input_length = 2 * (3 * pointcloud_preprocessor.max_pc2_size_in_data + 3)

    model_filepath = get_root_folder() / MODEL_DIR / MODEL_FILE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformer_models.RadarDeepMatcher(input_length)
    model.eval()
    model.load_state_dict(torch.load(
        model_filepath, weights_only=True, map_location=torch.device(device)))
    i = 0
    for topic, msg, t in input_bag.read_messages():
        if topic == PC2_TOPIC_NAME and msg.fields:
            pointcloud_input_msg = PointcloudInput()
            pointcloud_input_msg.header = msg.header
            pointcloud_input_msg.has_matches = False
            pointcloud_input_msg.current_pointcloud = msg
            input, input_non_empty = pointcloud_preprocessor.build_input_to_model(
                msg)
            if input_non_empty:
                # Do inference and write the matches into the message.
                input_tensor = torch.from_numpy(input).to(device)
                prediction = model(input_tensor)
                matches, pc1_size, pc2_size = pointcloud_preprocessor.get_matches_from_prediction(
                    prediction[0, :, :], input_tensor)
                pointcloud_preprocessor.build_output_msg_from_prediction(
                    matches, input_tensor, pointcloud_input_msg, i, prediction[0, :, :], pc1_size, pc2_size)
            output_bag.write(MATCHINGS_TOPIC_NAME, pointcloud_input_msg, t)
            i = i + 1
        # Write through all messages.
        output_bag.write(topic, msg, t)
    input_bag.close()
    output_bag.close()


if __name__ == '__main__':
    main()
