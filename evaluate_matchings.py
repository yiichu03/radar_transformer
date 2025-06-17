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

import hdf5_dataloader
import transformer_models
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import torch
import argparse
from pathlib import Path
import transformer_models
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import ConnectionPatch
import numpy as np
from scipy.optimize import linear_sum_assignment
import utils

torch.set_printoptions(precision=20)
torch.set_default_dtype(torch.float32)


def get_pos_dict_from_input(input):
    """Input is the input to the network - vector of stacked quituples (x, y, z, v, i).
       This function returns a dict {"pc": {point_index_in_input: [x, y, z, v, i]}}.
       This function accepts a single input (not a minibatched one).
    """
    # Divide input into two pointclouds.
    input = input.transpose(0, 1)
    single_pc_length = input.shape[0] // 2
    pcs_input = [input[:single_pc_length, 0].cpu().detach(
    ).numpy(), input[single_pc_length:, 0].cpu().detach().numpy()]
    pcs_coords = {"pc_1": {}, "pc_2": {}}
    for idx, pc in enumerate(pcs_coords.keys()):
        for point_idx in range(single_pc_length // 5):
            slice = point_idx * 5
            point_item = {point_idx: pcs_input[idx][slice:(slice + 5)]}
            pcs_coords[pc].update(point_item)
    return pcs_coords


def get_pos_vecs_from_dict(pos_dict):
    pc_1 = np.empty(shape=(0, 2), dtype=np.float64)
    pc_2 = np.empty(shape=(0, 2), dtype=np.float64)
    for point_pc_1, point_pc_2 in zip(pos_dict["pc_1"].values(), pos_dict["pc_2"].values()):
        pc_1 = np.append(pc_1, point_pc_1[
                         :2][np.newaxis, :], axis=0)
        pc_2 = np.append(pc_2, point_pc_2[
                         :2][np.newaxis, :], axis=0)
    return pc_1, pc_2


def get_matches_from_softmax(softmax_output, pos_dict):
    """Output has the form: [x1, y1, x2, y2]. Coords correspond to
    matched points in the respective pointcloud.
    """
    matches = np.empty(shape=(0, 4), dtype=np.float64)
    max = np.argmax(softmax_output, axis=0).cpu().detach().numpy()
    for idx, max_idx in enumerate(range(max.shape[0])):
        match = np.asarray(
            [pos_dict["pc_1"][idx][0], pos_dict["pc_1"][idx][1],
             pos_dict["pc_2"][max_idx][0], pos_dict["pc_2"][max_idx][1]], dtype=np.float64)[np.newaxis, :]
        cond_1 = np.any(match[0, :2]) and np.any(match[0, 2:])
        cond_2 = softmax_output.cpu().detach().numpy()[max_idx, idx] > 0.5
        if cond_1 and cond_2:
            matches = np.append(matches, match, axis=0)
    return matches


def plot_matches_from_prediction(predicted_matchings, input_pc):
    predicted_matchings = torch.transpose(
        predicted_matchings, 0, 1).detach().numpy()
    input_pc = input_pc.reshape(
        (input_pc.size()[1] // 3, 3)).detach().numpy()
    input_pc_1 = input_pc[:(input_pc.shape[0] // 2), :]
    input_pc_2 = input_pc[(input_pc.shape[0] // 2):, :]
    pc1_size, pc2_size = calculate_non_padded_pc_sizes(
        input_pc, input_pc.shape[0] // 2)
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('Matched points between radar scans xy', fontsize=50)
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
    # xz plot.
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('Matched points between radar scans xz', fontsize=50)
    # set up the axes for the first plot
    ax_12 = fig.add_subplot(1, 2, 1)
    ax_12.set_xlim(left=0, right=15)
    ax_12.set_ylim(bottom=-15, top=15)
    # set up the axes for the second plot
    ax_22 = fig.add_subplot(
        1, 2, 2, sharey=ax_1, sharex=ax_1)
    # Draw pointclouds (x, z).
    ax_12.plot(input_pc_1[1:pc1_size, 0],
               input_pc_1[1:pc1_size, 2],
               'ko', markersize=10)
    ax_22.plot(input_pc_2[1:pc2_size, 0],
               input_pc_2[1:pc2_size, 2],
               'ko', markersize=10)
    ax_12.set_xlabel('x [m]', fontsize=50)
    ax_12.set_ylabel('z [m]', fontsize=50)
    ax_22.set_xlabel('x [m]', fontsize=50)
    ax_22.set_ylabel('z [m]', fontsize=50)
    ax_22.yaxis.set_label_position("right")
    ax_12.set_title('Previous', fontsize=50)
    ax_22.set_title('Current', fontsize=50)
    ax_12.xaxis.set_minor_locator(AutoMinorLocator())
    ax_12.yaxis.set_minor_locator(AutoMinorLocator())
    ax_12.tick_params(axis='both', labelsize=40)
    ax_22.xaxis.set_minor_locator(AutoMinorLocator())
    ax_22.yaxis.set_minor_locator(AutoMinorLocator())
    ax_22.tick_params(axis='both', labelsize=40)
    ax_12.grid(True)
    ax_22.grid(True)
    # Run LSA on the calculated matrix.
    row_ind, col_ind = linear_sum_assignment(
        predicted_matchings[1:pc1_size, 1:pc2_size], maximize=True)
    # Take the max for thresholding. Higher the thresholds more outliers.
    threshold = np.amax(predicted_matchings[1:pc1_size, 1:pc2_size]) / 1.2
    for idx in range(row_ind.shape[0]):
        is_in_fov = utils.is_in_fov(
            np.vstack((input_pc_2[col_ind[idx] + 1, :], input_pc_1[row_ind[idx] + 1, :])))
        if is_in_fov:
            if (predicted_matchings[row_ind[idx] + 1, col_ind[idx] + 1] > threshold):
                xy_ax_2 = (input_pc_2[col_ind[idx] + 1, 0],
                           input_pc_2[col_ind[idx] + 1, 1])
                xy_ax_1 = (input_pc_1[row_ind[idx] + 1, 0],
                           input_pc_1[row_ind[idx] + 1, 1])
                connection = ConnectionPatch(xyA=xy_ax_1, xyB=xy_ax_2, coordsA="data", coordsB="data",
                                             axesA=ax_1, axesB=ax_2, color="green")
                ax_2.add_artist(connection)
                ax_1.plot(input_pc_1[row_ind[idx] + 1, 0],
                          input_pc_1[row_ind[idx] + 1, 1], 'ro', markersize=10)
                ax_2.plot(input_pc_2[col_ind[idx] + 1, 0],
                          input_pc_2[col_ind[idx] + 1, 1], 'ro', markersize=10)
                # xz.
                xz_ax_2 = (input_pc_2[col_ind[idx] + 1, 0],
                           input_pc_2[col_ind[idx] + 1, 2])
                xz_ax_1 = (input_pc_1[row_ind[idx] + 1, 0],
                           input_pc_1[row_ind[idx] + 1, 2])
                connection = ConnectionPatch(xyA=xz_ax_1, xyB=xz_ax_2, coordsA="data", coordsB="data",
                                             axesA=ax_12, axesB=ax_22, color="green")
                ax_22.add_artist(connection)
                ax_12.plot(input_pc_1[row_ind[idx] + 1, 0],
                           input_pc_1[row_ind[idx] + 1, 2], 'ro', markersize=10)
                ax_22.plot(input_pc_2[col_ind[idx] + 1, 0],
                           input_pc_2[col_ind[idx] + 1, 2], 'ro', markersize=10)
    plt.draw()
    # plt.show(block=True)


def calculate_non_padded_pc_sizes(X, half_input):
    pc1_size = np.max(np.count_nonzero(
        X[1:half_input, :], axis=0), axis=0)
    pc2_size = np.max(np.count_nonzero(
        X[(half_input + 1):, :], axis=0), axis=0)
    return pc1_size, pc2_size


def main(args):
    # Process commandline args.
    model_output_dir_as_string = "saved_models"
    if args.model_output_dir:
        model_output_dir_as_string = args.model_output_dir
    model_name_as_string = "model.ptm"
    if args.model_name:
        model_name_as_string = args.model_output_dir

    path_to_model = Path(model_output_dir_as_string) / model_name_as_string

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataloaders.
    train_data = hdf5_dataloader.HDF5Dataset(
        "./data/train/pointclouds.hdf5", "./data/train/labels.hdf5")
    test_data = hdf5_dataloader.HDF5Dataset(
        "./data/test/pointclouds.hdf5", "./data/test/labels.hdf5")

    test_dataloader = DataLoader(
        test_data, batch_size=transformer_models.MINIBATCH_SIZE, shuffle=True)
    train_dataloader = DataLoader(
        train_data, batch_size=transformer_models.MINIBATCH_SIZE, shuffle=True)

    model = transformer_models.RadarDeepMatcher(train_data.get_input_length())
    model.eval()
    model.load_state_dict(torch.load(
        path_to_model, weights_only=True, map_location=torch.device(device)))

    for batch_index, (X, y) in enumerate(test_dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        for i in range(transformer_models.MINIBATCH_SIZE):
            plot_matches_from_prediction(pred[i, :, :], X[i, :, :])
            # Below for plotting heatmap.
            matching_prob = pred.detach().numpy()
            # for i in range(matching_prob.shape[0]):
            fig, ax = plt.subplots()
            im = ax.imshow(matching_prob[i, :, :],
                           cmap='hot', interpolation='nearest')
            fig.colorbar(im)
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir",
                        help="Folder where the models are stored.")
    parser.add_argument("--model_name",
                        help="Saved model name.")
    args = parser.parse_args()
    main(args)
