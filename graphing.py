# Author : Sandeep Ramachandra (sandeep.ramachandra@student.uni-siegen.de)
# Description : Python file containing functions to plot the required format

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import helper
from GAN import GAN
import re
import random


def plot_timeseries_2d(data, grouping = 3, label = None, time_unit = 0.01, display = True,
                       save_name = None, xlabel = "Time"):
    # function to plot a label vs time plot for multiple sets of timeseries data
    # data is a numpy compatible timeseries array
    # grouping is a number indicating how to group the data
    # label should have data.shape[0]/grouping labels
    
    num_timeseries = data.shape[0]
    timelength = np.arange(0,data.shape[-1]*time_unit,time_unit)
    if num_timeseries % grouping != 0:
        print("Invalid grouping : ",num_timeseries," not divisible by ",grouping)
        return
    
    # create figures
    num_fig = int(num_timeseries/grouping)
    fig, ax = plt.subplots(nrows = num_fig, ncols = 1, sharex = True, figsize = (20, 3*num_fig))
    
    # plotting
    for i in range(num_fig):
        for j in range(grouping):
            ax[i].plot(timelength, data[i*grouping+j])
        if label is not None:
            ax[i].set_ylabel(label[i])
    
    ax[-1].set_xlabel(xlabel)
    
    # save plot
    if save_name is not None:
        plt.savefig(save_name)
        
    # displaying plot
    if display:
        plt.show()

    return fig, ax

def plot_activity(activity_func, activity_num, model, fake_suffix = None, grouping = 3, label = None,
                  time_unit = 0.01, display = True, save_prefix = None):
    # helper function to plot using function and trained GAN model
    # Use the first output of function as real data
    # Use model to generate one sample of synthetic data
    # If either fake_suffix or save_prefix is None, then figures are not saved
    
    # get real data sample
    real_data = activity_func(activity_num)
    real_sample = real_data[random.randint(1,len(real_data)-1)][0]
    
    # get fake sample
    model.eval()
    nl = model.noise_len
    with torch.no_grad():
        syn_data = model(torch.randn(1,nl,dtype=torch.float)).numpy()[0]
    
    # determine save names
    save_name_real = None
    save_name_fake = None
    if save_prefix is not None and fake_suffix is not None:
        save_name_real = save_prefix+"_real.png"
        save_name_fake = save_prefix+"_fake_"+fake_suffix+".png"
    
    # plotting
    _, _ = plot_timeseries_2d(real_sample, grouping = grouping, label = label, time_unit = time_unit,
                              display = display, save_name = save_name_real)
    _, _ = plot_timeseries_2d(syn_data, grouping = grouping, label = label, time_unit = time_unit,
                              display = display, save_name = save_name_fake)

def plot_realvsfake(datasetname, path_tensorboard_folder, fake_suffix, grouping = 3, label = None,
                    time_unit = 0.01, display = True):
    # one line loading function to identify dataset function and load each model for each activity from folder
    # and send it to plot_activity and save the figures
    # datasetname is either PAMAP2 or RWHAR
    # path_tensorboard_folder is the path to the dataset folder. model loaded will be the latest version in each activity
    
    # checking input
    if datasetname == "PAMAP2":
        total_activity = 7
        activityfunc = helper.load_PAMAP2_activity
    elif datasetname == "RWHAR":
        total_activity = 8
        activityfunc = helper.load_RWHAR_activity
    else:
        print("Invalid dataset name : Function only handles PAMAP2 or RWHAR")
        return
    
    activity_folder = [sub for sub in os.listdir(path_tensorboard_folder) if sub.startswith(datasetname)]
    if len(activity_folder) != total_activity:
        print("Not all activities are present at folder : use plot activity for individual activity")
    
    # plotting
    for activity in activity_folder:
        activity_num = int(re.findall('\d+', activity )[-1])
        
        version_folder = max(os.listdir(path_tensorboard_folder+"/"+activity))
        
        path = path_tensorboard_folder+"/"+activity+"/"+version_folder+"/checkpoints/"
        ckpt = os.listdir(path)[0]
        
        model = GAN.load_from_checkpoint(path+ckpt)
        
        plot_activity(activityfunc, activity_num, model, fake_suffix = fake_suffix,
                      save_prefix = "./figures/"+datasetname+"_"+activity, grouping = grouping,
                      label = label, time_unit = time_unit, display = display)

    
    
    
        