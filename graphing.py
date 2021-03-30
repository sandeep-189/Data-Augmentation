# Author : Sandeep Ramachandra (sandeep.ramachandra@student.uni-siegen.de)
# Description : Python file containing functions to plot the required format

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_timeseries_2d(data, grouping = 3, label = None, time_unit = 0.01):
    # function to plot a label vs time plot for multiple sets of timeseries data
    # data is a numpy compatible timeseries array
    # grouping is a number indicating how to group the data
    # label should have data.shape[0]/grouping labels
    
    num_timeseries = data.shape[0]
    timelength = np.arange(0,data.shape[-1]*time_unit,time_unit)
    if num_timeseries % grouping != 0:
        print("Invalid grouping : ",num_timeseries," not divisible by ",grouping)
        return
    
    num_fig = int(num_timeseries/grouping)
    fig, ax = plt.subplots(nrows = num_fig, ncols = 1, sharex = True, figsize = (20, 3*num_fig))
    
    for i in range(num_fig):
        for j in range(grouping):
            ax[i].plot(timelength, data[i*grouping+j])
        if label is not None:
            ax[i].set_ylabel(label[i])
    
    ax[-1].set_xlabel("Time")
    plt.show()
    return fig, ax

def plot_activity(activity_func, activity_num, model, grouping = 3, label = None, time_unit = 0.01):
    # helper function to plot using function and trained GAN model
    # Use the first output of function as real data
    # Use model to generate one sample of synthetic data
    
    real_data = activity_func(activity_num)[0][0]
    
    model.eval()
    nl = model.noise_len
    print(nl)
    syn_data = model(torch.randn(1,nl,dtype=torch.float))

    real_fig, _ = plot_timeseries_2d(real_data, grouping = grouping, label = label, time_unit = time_unit)
    syn_fig, _ = plot_timeseries_2d(syn_data, grouping = grouping, label = label, time_unit = time_unit)
    return real_fig, syn_fig
    