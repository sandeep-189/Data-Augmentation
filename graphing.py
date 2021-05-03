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
from sklearn.metrics import ConfusionMatrixDisplay


def plot_timeseries_2d(data, ncols = 1, grouping = 3, label = None, time_unit = 0.01, display = True,
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
    if num_fig % ncols != 0:
        print("Invalid number of columns. Unable to divide the figures.")
        return
    nrows = num_fig / ncols
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, figsize = (5*ncols, 3*nrows))
    
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
    
    activityfunc, total_activity = helper.get_activityfunc(datasetname)

    for model, activity_num in helper.load_tensorboard_models(datasetname, total_activity, path_tensorboard_folder):
        
        plot_activity(activityfunc, activity_num, model, fake_suffix = fake_suffix,
                      save_prefix = "./figures/"+datasetname+"_activity_"+str(activity_num), grouping = grouping,
                      label = label, time_unit = time_unit, display = display)

    
def mean_calc(data):
    # Calculate mean timeseries of input data. Pass only one activity data.
    # the function can handle a list of tuple of timeseries, target as well as list of timeseries
    if type(data[0]) is tuple:
        mean = np.zeros_like(data[0][0])
        items = 0
        for timeseries,_ in data: 
            mean += timeseries
            items += 1
        mean = mean / items
    else:
        mean = np.zeros_like(data[0])
        items = 0
        for timeseries in data: 
            mean += timeseries
            items += 1
        mean = mean / items 
    return mean

def mean_calc_real(dataset = "PAMAP2", sampling_size = 100, grouping = 3, label = None,
                    time_unit = 0.01, display = True, save = False):
    # Calculate mean for real data (only PAMAP2 or RWHAR supported) and plot the mean timeseries
    
    activityfunc, total_activity = helper.get_activityfunc(dataset)
    save_name = None
    for i in range(total_activity):
        if save:
            save_name = "./figures/"+dataset+"_activity_"+str(i+1)+"_real_mean.png"
        data = activityfunc(i+1)
        data = random.sample(data, sampling_size)
        mean = mean_calc(data)
        plot_timeseries_2d(mean, grouping = grouping, label = label, time_unit = time_unit,
                              display = display, save_name = save_name)
        
def mean_calc_fake(dataset, path_tensorboard_folder, sampling_size = 100, grouping = 3, label = None,
                    time_unit = 0.01, display = True, save = False, fake_suffix = "LSTM"):
    # Calculate mean for GAN model and plot it
    
    _, total_activity = helper.get_activityfunc(dataset)
    
    save_name = None
    for model, activity_num in helper.load_tensorboard_models(dataset, total_activity,path_tensorboard_folder):
            model.eval()
            nl = model.noise_len
            with torch.no_grad():
                syn_data = [model(torch.randn(1,nl,dtype=torch.float)).numpy()[0] for _ in range(sampling_size)]
            
            if save:
                save_name = "./figures/"+dataset+"_activity_"+str(activity_num)+"_"+fake_suffix+"_mean.png"
            mean = mean_calc(syn_data)
            plot_timeseries_2d(mean, grouping = grouping, label = label, time_unit = time_unit,
                              display = display, save_name = save_name)

def plot_confusion_matrix(model, dataloader,  classes = None):
    # Plot the comnfusion matrix given a model and a dataloader for it.
    model.eval()
    cm = np.zeros((len(classes),len(classes))) # rows is actual, cols is predicted
    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to("cuda")
            label = batch["label"].to("cuda")
            output = model(data)
            pred_o = torch.argmax(output, dim=1)
            for (y_true,y_pred) in zip(label,pred_o):
                cm[y_true.item(),y_pred.item()] += 1
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)
    disp.plot(values_format = '.0f',xticks_rotation = "vertical")