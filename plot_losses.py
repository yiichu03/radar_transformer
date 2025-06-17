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

import csv
import matplotlib.pyplot as plt


def plot(train_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.title("Average losses")
    plt.show()


def main():
    with open('losses.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        losses = []
        for row in csv_reader:
            losses.append([float(i) for i in row])
    plot(losses[0], losses[1])


if __name__ == '__main__':
    main()
