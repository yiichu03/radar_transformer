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

import sys
import numpy as np
import hdf5_dataloader
import transformer_models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import time
import argparse
from pathlib import Path
import csv
torch.set_printoptions(precision=20)
torch.set_default_dtype(torch.float32)
np.set_printoptions(threshold=sys.maxsize)


PLOT = False
SAVE = True


def train_one_epoch(dataloader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    losses_per_batch = []
    print("Training RadarTransformer.")
    for batch_index, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss.
        phi_phi = model(X)
        loss = criterion(phi_phi, X, y)
        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        losses_per_batch.append(loss.item())

    print("Epoch loss average: {:.4f}".format(
        running_loss / len(dataloader)))
    print('-' * 10)
    return running_loss / len(dataloader), losses_per_batch


def test_model(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    losses_per_batch = []
    print('Testing RadarTransformer.')
    for batch_index, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        phi_phi = model(X)
        loss = criterion(phi_phi, X, y)
        running_loss += loss.item()
        losses_per_batch.append(loss.item())

    print("Test loss average: {:.4f}".format(running_loss / len(dataloader)))
    print('-' * 10)
    return running_loss / len(dataloader), losses_per_batch


def main(args):
    torch.manual_seed(42)
    # Process commandline args.
    model_output_dir_as_string = "saved_models"
    if args.model_output_dir:
        model_output_dir_as_string = args.model_output_dir

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for param in model.parameters():
        param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]

    criterion = transformer_models.Criterion(
        model.num_points_per_pointcloud, device)
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.Adam(params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=0.1)

    # Training.
    train_losses = []
    test_losses = []
    for i in range(transformer_models.NUM_EPOCHS):
        print("Epoch {} / {}".format(i, transformer_models.NUM_EPOCHS - 1))
        print('-' * 20)
        start = time.time()
        train_loss, train_losses_per_batch = train_one_epoch(
            train_dataloader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        # train_losses.extend(train_losses_per_batch)
        test_loss, test_losses_per_batch = test_model(
            test_dataloader, model, criterion, device)
        test_losses.append(test_loss)
        # test_losses.extend(test_losses_per_batch)
        lr_scheduler.step()
        stop = time.time()
        print("Time per epoch: {}".format(stop - start))

    # Saving model.
    model_output_dir = Path(model_output_dir_as_string + "/" + "RadarTransformer_" +
                            time.strftime("%d%b%Y_%H%M%S") + ".ptm")
    model_output_dir.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_dir)
    print("Model saved as: ".format(model_output_dir))

    if PLOT:
        # Plot train/test losses.
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(test_losses, label="Test")
        plt.legend()
        plt.xlabel("Epochs")
        plt.title("Average losses")
        plt.savefig(str(model_output_dir).split(".ptm")[0] + "_losses.png")
        plt.show()
    if SAVE:
        with open("losses.csv", 'w', newline='') as losses_file:
            writer = csv.writer(losses_file, quoting=csv.QUOTE_ALL)
            writer.writerow(train_losses)
            writer.writerow(test_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir",
                        help="Folder where to store the model.")
    args = parser.parse_args()
    main(args)
