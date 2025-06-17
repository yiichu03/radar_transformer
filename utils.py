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

from pathlib import Path
import struct
import numpy as np
import rosbag
import h5py
from scipy.spatial.transform import Rotation as R


def get_bags_from_path(path):
    """Returns a list of relative paths to bagfiles as strings.
    """
    dataset_path = Path(path)
    files_in_dataset_path = dataset_path.iterdir()
    bags = []
    for item in files_in_dataset_path:
        if item.is_file():
            bags.append(item)
    return bags


def get_bags_from_paths(paths):
    """Returns a list of relative paths to bagfiles as strings.
    """
    bags = []
    for path in paths:
        bags.extend(get_bags_from_path(path))
    return bags


def decode_single_pc2_into_xyzvi(pc2, point_only=True):
    """Returns np.array([[x, y, z, v, i],
                         [x, y, z, v, i], ...]).
    """
    n_of_variables_in_point = 5
    if point_only:
        n_of_variables_in_point = 3
    points_xyzvi = np.empty(
        shape=(0, n_of_variables_in_point), dtype=np.float32)
    for i in range(pc2.width):
        point_as_bytes = pc2.data[(i * pc2.point_step)
                                   :((i + 1) * pc2.point_step)]
        point_as_float = list(struct.unpack(
            'ffffffff', point_as_bytes))
        points_xyzvi = np.append(points_xyzvi, np.asarray(
            point_as_float[:n_of_variables_in_point], dtype=np.float32)[np.newaxis, :], axis=0)
    return points_xyzvi


def get_max_pc2_size_in_data(bags, radar_topic="/ti_mmwave/radar_scan_pcl", point_only=True):
    """Returns the maximum length of the pointcloud found
    inside the dataset. This is used for padding and keeping
    the length of the input to the network uniform.
    """
    max_size = 0
    for bag in bags:
        with rosbag.Bag(bag, 'r') as databag:
            for topic, msg, _ in databag.read_messages():
                if topic == radar_topic and msg.data:
                    current_size = decode_single_pc2_into_xyzvi(
                        msg, point_only=point_only).shape[0]
                    if current_size > max_size:
                        max_size = current_size
    return max_size


def remove_dc_noise(points_xyzvi, threshold_x=0.1, threshold_y=0.1):
    cond_x_1 = points_xyzvi[:, 0] < threshold_x
    cond_y_1 = points_xyzvi[:, 1] < threshold_y
    cond_y_2 = points_xyzvi[:, 1] > -threshold_y
    joint_cond = cond_x_1 & (cond_y_1 & cond_y_2)
    points_xyzvi = np.delete(points_xyzvi, np.where(joint_cond), axis=0)
    return points_xyzvi


def filter_to_fov(points_xyzvi_as_list):
    # 60 degrees.
    elevation_up = np.array([[-0.7660444, 0, 0.6427876]]).T
    elevation_down = np.array([[-0.7660444, 0, -0.6427876]]).T
    # 60 degrees.
    azimuth_left = np.array([[-0.7660444, 0.6427876, 0]]).T
    azimuth_right = np.array([[-0.7660444, -0.6427876, 0]]).T
    # Take the dot products.
    filtered_points = []
    for idx in range(len(points_xyzvi_as_list)):
        # e_up = np.dot(points_xyzvi_as_list[idx], elevation_up) > 0
        # e_down = np.dot(points_xyzvi_as_list[idx], elevation_down) > 0
        a_left = np.dot(points_xyzvi_as_list[idx], azimuth_left) > 0
        a_right = np.dot(points_xyzvi_as_list[idx], azimuth_right) > 0
        final_mask = ~(a_left | a_right)
        filtered_points.append(
            points_xyzvi_as_list[idx][final_mask.reshape((final_mask.size, )), :])
    return filtered_points


def is_in_fov(points):
    """Check if all points are in fov. [[x, y, z],
                                        [x, y, z]]"""
    # 60 degrees.
    azimuth_left = np.array([[-0.7660444, 0.6427876, 0]]).T
    azimuth_right = np.array([[-0.7660444, -0.6427876, 0]]).T
    a_left = np.dot(points, azimuth_left) > 0
    a_right = np.dot(points, azimuth_right) > 0
    final_mask = ~(a_left | a_right)
    return np.all(final_mask == True)


def apply_radar_imu_transform_to_single_pointcloud(rotation, translation, pointcloud):
    pointcloud_transformed = np.matmul(
        rotation, pointcloud.transpose()) + translation.transpose()
    return pointcloud_transformed.transpose()


def apply_inv_radar_imu_transform_to_single_pointcloud(rotation, translation, pointcloud):
    pointcloud_transformed = np.matmul(
        rotation.transpose(), pointcloud.transpose() - translation.transpose())
    return pointcloud_transformed.transpose()


def apply_radar_imu_transform(rotation, translation, pointclouds_as_mat_list):
    pointclouds_transformed = []
    for pointcloud in pointclouds_as_mat_list:
        # if pointcloud.size != 0:
        pointcloud_transformed = apply_radar_imu_transform_to_single_pointcloud(
            rotation, translation, pointcloud)
        pointclouds_transformed.append(pointcloud_transformed)
    return pointclouds_transformed


def build_input_to_model(previous_pointcloud, current_pointcloud, max_input_size):
    first_pointcloud = remove_dc_noise(
        previous_pointcloud)
    second_pointcloud = remove_dc_noise(
        current_pointcloud)
    needed_zeros_first_pointcloud = max_input_size - \
        first_pointcloud.shape[0] if max_input_size - \
        first_pointcloud.shape[0] > 0 else 0
    first_pointcloud_padded = np.pad(
        first_pointcloud[:max_input_size, :], ((1, needed_zeros_first_pointcloud), (0, 0)), "constant", constant_values=(0, 0))
    # 3. Pad with zeros after transformation.
    needed_zeros_second_pointcloud = max_input_size - \
        second_pointcloud.shape[0] if max_input_size - \
        second_pointcloud.shape[0] > 0 else 0
    second_pointcloud_padded = np.pad(
        second_pointcloud[:max_input_size, :], ((1, needed_zeros_second_pointcloud), (0, 0)), "constant", constant_values=(0, 0))
    # 4. Concatenate both pointclouds as one input entry.
    input_data_block = np.hstack((first_pointcloud_padded.reshape((1, first_pointcloud_padded.size)),
                                 second_pointcloud_padded.reshape((1, second_pointcloud_padded.size))))
    return input_data_block


class DataFactory:
    """write pc2 and gt for training from all the bags into files
    inside `data` folder. Write as `.hdf5` files. Files are:
    `pointclouds.hdf5` and `labels.hdf5`.
    """

    def __enter__(self):
        # Remove old files.
        if self.mode == "train":
            examples_path = "./data/train/pointclouds.hdf5"
            labels_path = "./data/train/labels.hdf5"
        elif self.mode == "test":
            examples_path = "./data/test/pointclouds.hdf5"
            labels_path = "./data/test/labels.hdf5"
        else:
            raise ValueError("Unknown mode.")

        Path(examples_path).unlink(missing_ok=True)
        Path(labels_path).unlink(missing_ok=True)

        # Training files.
        self.pointclouds_file = h5py.File(examples_path, 'a')
        self.labels_file = h5py.File(labels_path, 'a')
        n_pointclouds_in_input = 2
        n_coords_in_point = 3
        # Add 1 to make the non-matched class.
        self.examples_dset = self.pointclouds_file.create_dataset(
            "examples", shape=(1, n_pointclouds_in_input * n_coords_in_point * self.pc2_max_size_in_data + 6),
            chunks=True, maxshape=(None, n_pointclouds_in_input * n_coords_in_point * self.pc2_max_size_in_data + 6))
        self.labels_dset = self.labels_file.create_dataset(
            "labels", shape=(1, n_coords_in_point * self.pc2_max_size_in_data),
            chunks=True, maxshape=(None, n_coords_in_point * self.pc2_max_size_in_data))
        # Below only relevant when normalizing the data.
        if self.mode == "train":
            # Normalization file-mapped arrays.
            self.pointclouds_norm_file = np.memmap(
                self.pointclouds_norm_path, dtype='float32', mode='w+', shape=(1, 5))
            self.labels_norm_file = np.memmap(
                self.labels_norm_path, dtype='float32', mode='w+', shape=(1, 3))
        return self

    def add_examples_and_labels_for_normalization(self, examples, labels):
        labels = np.diff(labels, axis=0)
        # debug
        self.gt_debug_non_interp = np.vstack(
            (self.gt_debug_non_interp, labels))
        # debug
        # Remove DC-corrupted values.
        examples = np.vstack(examples)
        examples = remove_dc_noise(examples)
        self.pointclouds_norm_file = np.vstack(
            (self.pointclouds_norm_file, examples))
        self.labels_norm_file = np.vstack((self.labels_norm_file, labels))

    def DEBUG_flush_gt_buffer(self):
        # debug
        self.gt_debug = np.empty(shape=(0, 3 + 4), dtype=np.float64)
        self.gt_debug_non_interp = np.empty(shape=(0, 3), dtype=np.float64)
        self.gt_debug_time = np.empty(shape=(0, 1), dtype=np.float64)
        # debug

    def calculate_normalization_params(self, normalizer):
        # Remove first rows which have zeros.
        self.pointclouds_norm_file = np.delete(
            self.pointclouds_norm_file, (0), axis=0)
        self.labels_norm_file = np.delete(
            self.labels_norm_file, (0), axis=0)
        normalizer.calculate_normalization_params(
            self.pointclouds_norm_file, self.labels_norm_file)

    def __exit__(self, exc_type, exc_value, traceback):
        self.pointclouds_file.close()
        self.labels_file.close()
        if self.mode == "train":
            print(
                "Removing file-mapped arrays used for calculating normalization params ... .")
            Path(self.pointclouds_norm_path).unlink(missing_ok=True)
            Path(self.labels_norm_path).unlink(missing_ok=True)

    def __init__(self, pc2_max_size_in_data, mode):
        # debug
        self.gt_debug = np.empty(shape=(0, 3 + 4), dtype=np.float64)
        self.gt_debug_non_interp = np.empty(shape=(0, 3), dtype=np.float64)
        self.gt_debug_time = np.empty(shape=(0, 1), dtype=np.float64)
        # debug
        self.mode = mode
        self.pc2_max_size_in_data = pc2_max_size_in_data
        self.row_number = 0
        if self.mode == "train":
            # Normalization files.
            self.pointclouds_norm_path = "./data/norm/pointclouds.memmap"
            self.labels_norm_path = "./data/norm/labels.memmap"

    def _write_to_hdf5(self, gt_data_block, input_data_block):
        """Here we pad the data to the maximum length and write into hdf5.
        """
        self.row_number = self.row_number + 1
        self.labels_dset.resize(
            self.row_number, axis=0)
        self.labels_dset[(self.row_number - 1):] = gt_data_block
        self.examples_dset.resize(
            self.row_number, axis=0)
        self.examples_dset[(self.row_number - 1):] = input_data_block

    def _find_closest(self, A, target):
        # A must be sorted.
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A)-1)
        left = A[idx - 1]
        right = A[idx]
        idx -= target - left < right - target
        return idx

    def generate_gt_and_input_data(self, gt_time, gt_position, gt_orientation, radar_time, pointclouds_as_xyzvi_mat_list):
        # Get indexes of time instants of the GT closest to radar measurements intstants.
        closest_gt_to_meas_idx = self._find_closest(gt_time, radar_time)
        # Input contains [x1, y1, z1]
        #                     ...
        #                [x2transformed, y2transformed, z2transformed]
        #                     ...
        # Points from the second pointcloud are transformed into the frame of the
        # measurement of the first pointcloud.
        # GT contains [x1transformed, y1transformed, z1transformed].
        # Points from the first pointcloud are transformed within the frame
        # of the first pointcloud using the relative transformed ("moved").
        for idx in range(closest_gt_to_meas_idx.shape[0] - 1):
            # 1. Calculate the relative transform between the consecutive poses - only for GT.
            # Orientation.
            r1 = R.from_quat(
                gt_orientation[closest_gt_to_meas_idx[idx]]).as_matrix()
            r2 = R.from_quat(
                gt_orientation[closest_gt_to_meas_idx[idx + 1]]).as_matrix()
            rel_rot = np.matmul(r1.transpose(), r2)
            # Position.
            rel_pos = gt_position[closest_gt_to_meas_idx[idx + 1]
                                  ] - gt_position[closest_gt_to_meas_idx[idx]]
            rel_pos = rel_pos[np.newaxis, :]
            # 2. Process and concatenate both pointclouds as one input entry.
            input_data_block = build_input_to_model(pointclouds_as_xyzvi_mat_list[idx],
                                                    pointclouds_as_xyzvi_mat_list[idx + 1],
                                                    self.pc2_max_size_in_data)
            # 3. Generate GT from the first pointcloud (first pointcloud in the second pointcloud's frame).
            first_pointcloud = remove_dc_noise(
                pointclouds_as_xyzvi_mat_list[idx])
            needed_zeros_first_pointcloud = self.pc2_max_size_in_data - \
                first_pointcloud.shape[0]
            transformed_first_pointcloud = np.matmul(
                rel_rot.transpose(), (first_pointcloud - rel_pos).transpose()).transpose()
            transformed_first_pointcloud = np.pad(
                transformed_first_pointcloud, ((0, needed_zeros_first_pointcloud), (0, 0)), "constant", constant_values=(0, 0))
            gt_data_block = transformed_first_pointcloud.reshape(
                (1, transformed_first_pointcloud.size))
            # 4. Write to hdf5.
            self._write_to_hdf5(gt_data_block, input_data_block)


class MinMaxNormalizer:
    def __init__(self, feature_range=(-1, 1)):
        self.transform_dict = None
        self.feature_range = feature_range

    def calculate_normalization_params(self, examples_dset, gt_dset):
        self.transform_dict = {"x": [np.min(examples_dset[:, 0], axis=0),
                                     np.max(examples_dset[:, 0], axis=0)],
                               "y": [np.min(examples_dset[:, 1], axis=0),
                                     np.max(examples_dset[:, 1], axis=0)],
                               "z": [np.min(examples_dset[:, 2], axis=0),
                                     np.max(examples_dset[:, 2], axis=0)],
                               "v": [np.min(examples_dset[:, 3], axis=0),
                                     np.max(examples_dset[:, 3], axis=0)],
                               "i": [np.min(examples_dset[:, 4], axis=0),
                                     np.max(examples_dset[:, 4], axis=0)],
                               "gt_x": [np.min(gt_dset[:, 0], axis=0),
                                        np.max(gt_dset[:, 0], axis=0)],
                               "gt_y": [np.min(gt_dset[:, 1], axis=0),
                                        np.max(gt_dset[:, 1], axis=0)],
                               "gt_z": [np.min(gt_dset[:, 2], axis=0),
                                        np.max(gt_dset[:, 2], axis=0)]}
        print(self.transform_dict)

    def _normalize(self, data, minmax):
        data_norm = (data - minmax[0]) / (minmax[1] - minmax[0])
        data_norm_scaled = data_norm * \
            (self.feature_range[1] - self.feature_range[0]
             ) + self.feature_range[0]
        return data_norm_scaled

    def normalize_points_xyzvi(self, points_xyzvi):
        points_xyzvi[:, 0] = self._normalize(
            points_xyzvi[:, 0], self.transform_dict["x"])
        points_xyzvi[:, 1] = self._normalize(
            points_xyzvi[:, 1], self.transform_dict["y"])
        points_xyzvi[:, 2] = self._normalize(
            points_xyzvi[:, 2], self.transform_dict["z"])
        points_xyzvi[:, 3] = self._normalize(
            points_xyzvi[:, 3], self.transform_dict["v"])
        points_xyzvi[:, 4] = self._normalize(
            points_xyzvi[:, 4], self.transform_dict["i"])
        return points_xyzvi

    def normalize_gt(self, gt):
        gt[0, 0] = self._normalize(
            gt[0, 0], self.transform_dict["gt_x"])
        gt[0, 1] = self._normalize(
            gt[0, 1], self.transform_dict["gt_y"])
        gt[0, 2] = self._normalize(
            gt[0, 2], self.transform_dict["gt_z"])
        return gt


class ZeroMeanUnitVarNormalizer:
    def __init__(self):
        self.transform_dict = None

    def calculate_normalization_params(self, examples_dset, gt_dset):
        self.transform_dict = {"x": [np.mean(examples_dset[:, 0], axis=0),
                                     np.std(examples_dset[:, 0], axis=0)],
                               "y": [np.mean(examples_dset[:, 1], axis=0),
                                     np.std(examples_dset[:, 1], axis=0)],
                               "z": [np.mean(examples_dset[:, 2], axis=0),
                                     np.std(examples_dset[:, 2], axis=0)],
                               "v": [np.mean(examples_dset[:, 3], axis=0),
                                     np.std(examples_dset[:, 3], axis=0)],
                               "i": [np.mean(examples_dset[:, 4], axis=0),
                                     np.std(examples_dset[:, 4], axis=0)],
                               "gt_x": [np.mean(gt_dset[:, 0], axis=0),
                                        np.std(gt_dset[:, 0], axis=0)],
                               "gt_y": [np.mean(gt_dset[:, 1], axis=0),
                                        np.std(gt_dset[:, 1], axis=0)],
                               "gt_z": [np.mean(gt_dset[:, 2], axis=0),
                                        np.std(gt_dset[:, 2], axis=0)]}
        print(self.transform_dict)

    def _normalize(self, data, mean_var):
        data_norm = (data - mean_var[0]) / mean_var[1]
        return data_norm

    def normalize_points_xyzvi(self, points_xyzvi):
        points_xyzvi[:, 0] = self._normalize(
            points_xyzvi[:, 0], self.transform_dict["x"])
        points_xyzvi[:, 1] = self._normalize(
            points_xyzvi[:, 1], self.transform_dict["y"])
        points_xyzvi[:, 2] = self._normalize(
            points_xyzvi[:, 2], self.transform_dict["z"])
        points_xyzvi[:, 3] = self._normalize(
            points_xyzvi[:, 3], self.transform_dict["v"])
        points_xyzvi[:, 4] = self._normalize(
            points_xyzvi[:, 4], self.transform_dict["i"])
        return points_xyzvi

    def normalize_gt(self, gt):
        gt[0, 0] = self._normalize(
            gt[0, 0], self.transform_dict["gt_x"])
        gt[0, 1] = self._normalize(
            gt[0, 1], self.transform_dict["gt_y"])
        gt[0, 2] = self._normalize(
            gt[0, 2], self.transform_dict["gt_z"])
        return gt
