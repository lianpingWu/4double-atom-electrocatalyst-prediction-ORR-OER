# coding: utf-8

import matplotlib
matplotlib.use("Agg")

import torch
import matplotlib.pyplot as plt
import numpy as np
from Model import Model

def plot_error_records(cata_type):
    train_error_mean = np.loadtxt("./Model_save/training_error_" + cata_type + ".txt")
    test_error_mean = np.loadtxt("./Model_save/testing_error_" + cata_type + ".txt")
    best_num = np.where(test_error_mean == np.min(test_error_mean))[0]

    plt.figure(figsize=(12, 10))
    plt.plot(train_error_mean[20::], "-o", c="g", linewidth=4, ms=10, mfc="g")
    plt.plot(test_error_mean[20::], "-o", c="r", linewidth=4, ms=10, mfc="r")
#     plt.xlim([20, 300])
#     plt.ylim([0, 100])
    plt.grid(axis="y", linestyle="--", linewidth=2)
    plt.legend(["Training error", "Testing error"], fontsize=25)
    plt.title([str(train_error_mean[best_num]), str(test_error_mean[best_num])], fontsize=25)
    plt.savefig("./results/results_error_record_" + cata_type + ".jpg", dpi=100, bbox_inches="tight")
    plt.close()


def plot_regression(cata_type):
    if cata_type == "M":
        index = 0
    elif cata_type == "M-OH":
        index = 1
    elif cata_type == "M-O":
        index = 2
    elif cata_type == "M-OOH":
        index = 3
    elif cata_type == "OER":
        index = 4
    elif cata_type == "ORR":
        index = 5
    elif cata_type == "metal":
        index = 6
    elif cata_type == "carbon":
        index = 7
    elif cata_type == "carbon-metal":
        index = 8
    elif cata_type == "binding_energy":
        index = 9

    training_samples = np.load("../training_samples/inputs_training.npy")
    testing_samples = np.load("../training_samples/inputs_testing.npy")
    training_labels = np.load("../training_samples/outputs_training.npy")[:, index]
    testing_labels = np.load("../training_samples/outputs_testing.npy")[:, index]

    model_save = Model().cuda()
    model_save.load_state_dict(torch.load("./Model_save/model_" + cata_type + ".pkl"))
    inputs_train = torch.tensor(training_samples).float().cuda()
    inputs_test = torch.tensor(testing_samples).float().cuda()

    model_save.eval()
    with torch.no_grad():
        pred_train = model_save(inputs_train).detach().cpu().numpy()
        pred_test = model_save(inputs_test).detach().cpu().numpy()

    plt.figure(figsize=(12, 12)) 
    plt.scatter(pred_train, training_labels, s=150, c="g", ec="k", linewidth=1.5) 
    plt.scatter(pred_test, testing_labels, s=150, c="r", ec="k", linewidth=1.5) 
    plt.plot(
        [
            min(np.min(training_labels), np.min(pred_train), np.min(testing_labels), np.min(pred_test)),
            max(np.max(training_labels), np.max(pred_train), np.max(testing_labels), np.max(pred_test))
            ],
            
        [
            min(np.min(training_labels), np.min(pred_train), np.min(testing_labels), np.min(pred_test)),
            max(np.max(training_labels), np.max(pred_train), np.max(testing_labels), np.max(pred_test))
            ], 
        "-.k", 
        linewidth=3
    )

    plt.xlabel("Predictions", fontsize=20)
    plt.ylabel("DFT calculations", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim([
        min(np.min(training_labels), np.min(pred_train), np.min(testing_labels), np.min(pred_test)),
        max(np.max(training_labels), np.max(pred_train), np.max(testing_labels), np.max(pred_test))
        ])

    plt.ylim([
        min(np.min(training_labels), np.min(pred_train), np.min(testing_labels), np.min(pred_test)),
        max(np.max(training_labels), np.max(pred_train), np.max(testing_labels), np.max(pred_test))
        ])

    plt.legend(["Training data", "Testing data"], fontsize=25)
    plt.savefig("./results/results_regression_" + cata_type + ".jpg", dpi=100, bbox_inches="tight")
    plt.close()