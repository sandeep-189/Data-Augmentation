import pandas as pd
import os
import numpy as np
from torch.utils.data import random_split
import torch
import dataset
import Generator_LSTM
import Discriminator_LSTM
import Generator_transformer
import Discriminator_transformer
from F1_score_check import F1_score_check
from GAN import GAN
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from DeepConvLSTM_model import DeepConvNet
from TransformerClassifier import TransformerClassifier
from Validation_model import Net

DEFAULT_RWHAR_FILEPATH = ["./data/RWHAR/"]

DEFAULT_PAMAP2_FILEPATHS = ["./data/PAMAP2/subject101.dat","./data/PAMAP2/subject102.dat","./data/PAMAP2/subject103.dat","./data/PAMAP2/subject104.dat","./data/PAMAP2/subject105.dat","./data/PAMAP2/subject106.dat","./data/PAMAP2/subject107.dat","./data/PAMAP2/subject108.dat","./data/PAMAP2/subject109.dat"]



def clean_all_files(filepaths, clean_func): 
    # filespaths is a list of text files with panda tables stored in them.
    # the clean_func is a reference to the function to be applied to clean the dataset
    
    total = pd.DataFrame()
    
    for file in filepaths:
        table = clean_func(file) # table has cleaned dataset from one file
        total = total.append(table,ignore_index = True) # joins all data into one dataframe
    return total

def load_table(filepaths = DEFAULT_PAMAP2_FILEPATHS, clean_func = dataset.clean_PAMAP2_dataset, save_file = "clean_PAMAP2.pkl", force_reload = False):
    # function to load a panda table from filepaths, clean each table using given clean_func, append them into one table and return it.
    # the table is saved and reloaded when needed or when forced to reload it.
    
    if ((not os.path.isfile(save_file)) or force_reload): 
        print("Cleaning and loading subject files")
        table = clean_all_files(filepaths, clean_func)
        print("Compressing to file")
        table.to_pickle(save_file, compression="gzip")
    else:
        print("File exists. Loading")
        table = pd.read_pickle(save_file, compression="gzip")
        
    return table

def load_PAMAP2(force_reload = False):
    # helper function for one line loading with optimizations 
    
    table = load_table(force_reload = force_reload)
    
    print("Windowing")
    data = windowing(table)
    print("Done!")
    return data

def load_PAMAP2_activity(activity_num = 0, force_reload = False):
    # helper function to load only one activty from PAMAP2 table
    
    table = load_table(force_reload = force_reload)
    
    print("Keep only activity number ", activity_num)
    table = table.where(table.iloc[:,1] == activity_num) # every activity except activity_num becomes NA
    table = table.dropna()
    
    print("Windowing")
    data = windowing(table)
    print("Done!")
    return data

def load_RWHAR(sel_location = "chest", force_reload = False):
    # helper function for one line loading with optimizations 
    
    table = load_table(filepaths = DEFAULT_RWHAR_FILEPATH,
                       clean_func = dataset.clean_RWHAR, 
                       save_file = "clean_RWHAR.pkl", 
                       force_reload = force_reload
                      )
    print("Selecting location : ",sel_location)
    table = table[table.location == dataset.RWHAR_BAND_LOCATION[sel_location]] # selecting location
    table = table.drop(columns = "location") # drop the now unnecessary location column
    table = table.dropna() # drop all entries which do not have complete information
    
    # Debugging
    table = table.where(table.activity != 3)
    table = table.dropna()
    table.replace({'activity':{4:3,5:4,6:5,7:6,8:7}}, inplace = True)
    
    print("Windowing")
    # Since each subject has different timestamps and the windowing function does not check the timestamps, we have to window each subject 
    # separately and append them together
    
    data =[]
    for sub in table.subject.unique():
        tmp = windowing(table[table.subject==sub], sampling_freq = 50, group_col_num = 6, data_columns = range(0,6))
        data.extend(tmp)
    print("Done!")
    return data

def load_RWHAR_activity(activity_num = 0, sel_location = "chest", force_reload = False):
        # helper function for one line loading of one axctivity of RWHAR dataset 
    
    table = load_table(filepaths = DEFAULT_RWHAR_FILEPATH,
                       clean_func = dataset.clean_RWHAR, 
                       save_file = "clean_RWHAR.pkl", 
                       force_reload = force_reload
                      )
    print("Selecting location : ",sel_location)
    table = table[table.location == dataset.RWHAR_BAND_LOCATION[sel_location]] # selecting location
    table = table.drop(columns = "location") # drop the now unnecessary location column
    print("Selecting Activity ",activity_num)
    table = table[table.activity == activity_num]
    table = table.dropna() # drop all entries which do not have complete information
    
    print("Windowing")
    # Since each subject has different timestamps and the windowing function does not check the timestamps, we have to window each subject 
    # separately and append them together
    
    data =[]
    for sub in table.subject.unique():
        tmp = windowing(table[table.subject==sub], sampling_freq = 50, group_col_num = 6, data_columns = range(0,6))
        data.extend(tmp)
    print("Done!")
    return data

def windowing(df, time_window = 1, sampling_freq = 100, group_col_num = 1, data_columns = range(2,29)):
    # df is the cleaned dataframe from dataset. This function will break the different activities to "window"
    # seconds of grouped data using group_col_num as the activities column with sampling frequency determining number of  
    # samples in each second. If the window cannot be filled by same activity data from subject, it will be dropped.
    # Note: we assume that the timestamps are close by since the activities are done for a continuous period of time. 
    #       when the activity changes, only then is there a jump in timestamp. The table should be fed subject by subject
    
    event_window = time_window * sampling_freq # number of events in one window
    output_shape = (len(data_columns), event_window)
    
    data = []
    ptr = 0
    while(ptr+event_window < df.shape[0]):
        activity = df.iloc[ptr,group_col_num] # which activity id to group by
        
        # check if all activities in event window have same id and can fill one full data value
        if (df.iloc[ptr:ptr+event_window, group_col_num] == activity).all() :
            d = df.iloc[ptr:ptr+event_window,data_columns].values
            d = np.reshape(d, output_shape)
            data.append((d, activity)) # append as a tuple of data and activity number
            
        else: # move pointer to nearest different activity number and then continue
            ptr += (df.iloc[ptr:ptr+event_window, group_col_num] != activity).values.argmax()
            continue
            
        ptr += event_window
        
    return data

def split(dataset, val_pc):
    # dataset is the dataset to be split. val_pc is the percentage in float split given to val dataset.
    
    valnum = int(np.floor(val_pc * len(dataset)))
    trainnum = len(dataset) - valnum
    
    return random_split(dataset, [trainnum, valnum])

def dist(dataset, num_classes):
    # function is used to get a weight distribution of each label of a classified dataset to pass to Cross entropy loss object. 
    # it helps to train faster and better for imbalanced datasets
    # dataset is expected to be a torch dataset. This will return a vector of weights of each class
    
    weight = np.zeros((num_classes,1))
    for data in dataset:
        try:
            weight[data["label"].item()] += 1
        except:
            print(data)
            raise Exception("Error")
            
    weight /= len(dataset)
    return weight

def remove_prefix_from_dict(prefix, dicti):
    # pytorch lightning saves models with each models variable name. However we need only the state dict for
    # restoring a saved model so we need to strip the prefixed name from each dictionary key.
    # prefix is the variable name (must include the dot '.')
    # dicti is the state dict taken from the saved pytorch lightning model (not needed if you do not use PL)
    
    keys = list(dicti.keys())
    for key in keys:
        if key.startswith(prefix):
            new_key = key[len(prefix):] # new key is everthing after the prefix name
            dicti[new_key] = dicti.pop(key) # the value of old key is assigned to stripped key (old key is removed from dictionary)
            
    return dicti

def conv1d_ele_size(input_size, kernel, padding, stride, dilation):
    # Convinience function to get the output size of a 1d convolution 
    
    return int((input_size + (2*padding) - dilation*(kernel - 1) - 1)/stride + 1)

def generate_pe(channel, length, period = 100, channel_cosine = True):
    # This function generates a positional embedding needed for transformer to gain awareness of position of input
    # suited for time series. If channel_cosine is false, instead of sin and cos functions across the row and column  
    # respectively, we use sin of wavelength period for each channel. we return a addable pe of size channel, length.
    # The batch dimension will be broadcasted automatically.
    
    if not channel_cosine: # pe is a sin function of length repeated across channel
        pos = torch.arange(0, length, dtype = torch.float32)
        pe = torch.sin(pos * 2 * np.pi / period)
        pe = pe.repeat((channel,1))
        
    else: # pe is a sum of sin and cos of the position of variable
        pe = torch.zeros((channel,length), dtype = torch.float32)
        for i in torch.arange(0, length, dtype = torch.long):
            for j in torch.arange(0, channel, dtype = torch.long):
                pe[j,i] = torch.sin(i * 2 * np.pi / period) + torch.cos(j * 2 * np.pi / period)
    
    return pe

def get_dataloaders(data_func, batch_size, output_size, val_pc, **kwargs):
    # Function to get the data from data function and load it to a train and validation dataloaders and the weight of the training dataset.
    # batch_size and output_size are used in dataloader options
    # val_pc can be a float or int. If float, it is the percentage split between train and validation.
    # If int, it is treated as number of validation iterations to generate a spoof validation dataloader filled with ones (used in the GANs). 
    # data_func is a function that returns 
    
    data = data_func(**kwargs)
    dtset = dataset.TimeSeriesDataset(data)
    weight = dist(dtset, output_size)
    
    if type(val_pc) is float:
        # We need to split the dataset to val_pc percentage
        train, val = split(dtset, val_pc = val_pc)
        train_weight = dist(train, output_size)
        val_weight = dist(val, output_size)
        print(train_weight, val_weight) # debug purposes
        train_iter = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = 10, pin_memory = True)
        val_iter = torch.utils.data.DataLoader(val, batch_size = batch_size, num_workers = 10, pin_memory = True)
        
        return train_iter, val_iter, train_weight
    else:
        # spoof validation dataset
        train_iter = torch.utils.data.DataLoader(dtset, batch_size = batch_size, shuffle = True, num_workers = 10, pin_memory = True)
        val = torch.ones((batch_size * val_pc, 1))
        val_iter = torch.utils.data.DataLoader(val, batch_size = batch_size, num_workers = 10, pin_memory = True)
        
        return train_iter, val_iter
    
def load_pl_model(ckpt_path, class_name, remove_prefix ="model.", strict_loading = False,  **kwargs):
    # Validation model has to be extracted from pytorch lightning modules so this is a one line loading
    # remove_prefix can be set to None if you dont want any prefix to be removed from ckpt file
    # strict _loading is a boolean used to indicate if the names has to be checked and if error should be thrown for 
    # unknown variables in state_dict of the ckpt file
    # **kwargs is passed to class being initialized/loaded
    
    val_model = class_name(**kwargs)
    state_dict = torch.load(ckpt_path)["state_dict"]
    
    if remove_prefix is not None:
        state_dict = remove_prefix_from_dict(remove_prefix, state_dict)
    val_model.load_state_dict(state_dict, strict = strict_loading)
    val_model.eval()
    return val_model

def train_LSTM_GAN(
                data_func,
                val_model,
                start_activity = 1,
                total_activities = 7,
                val_iter_size = 3,
                batch_size = 20,
                data_size = (27, 100),
                noise_len = 100,
                gen_num_layers = 2,
                gen_bidirectional = False,
                dis_hidden_size = 100,
                dis_num_layers = 2,
                dis_bidirectional = False,
                decay = 1,
                dis_lr = 0.0002,
                gen_lr = 0.0002,
                max_epochs = 100,
                tensorboard_save_dir = "LSTM_GAN_logs",
                tensorboard_name_prefix = "PAMAP2_act_",
                monitor = "val_f1_score",
                threshold_value = 0.95,**kwargs):
    # One line training using pytorch lightning abstraction framework
    # data_func is a function which takes in activity_num and gives the data of only those data
    # val_model is the trained classifier to determine if GAN output is identifiable as that activity
    
    success = {}
    
    for chosen_activity in range(start_activity,total_activities+1):
        train_iter, val_iter = get_dataloaders(data_func, batch_size = batch_size, output_size = total_activities, val_pc = val_iter_size, activity_num = chosen_activity, **kwargs)

        model = GAN(val_model = val_model, 
                    noise_len = noise_len, 
                    val_expected_output = chosen_activity-1,
                    generator = Generator_LSTM.Generator(hidden_size = noise_len, num_layers = gen_num_layers, 
                                                         bidirectional = gen_bidirectional, noise_len = noise_len, 
                                                         output_size = data_size),
                    discriminator = Discriminator_LSTM.Discriminator(hidden_size = dis_hidden_size, 
                                                                     bidirectional = dis_bidirectional, 
                                                                     num_layers = dis_num_layers, input_size = data_size),
                    num_classes = total_activities,
                    decay = decay,
                    dis_lr = dis_lr,
                    gen_lr = gen_lr,
                   )

        trainer = pl.Trainer(gpus=-1,
                             max_epochs=max_epochs,
                             callbacks = [F1_score_check(monitor, threshold_value), 
                                         ], # Early stopping callback
                             logger = TensorBoardLogger(save_dir = tensorboard_save_dir, name = tensorboard_name_prefix + str(chosen_activity)),
                             check_val_every_n_epoch = 5,
                             )
        trainer.fit(model, train_iter, val_iter)
        
        # verify if the model is trained
        if trainer.callback_metrics[monitor] >= threshold_value:
            print("Success!")
            success[chosen_activity] = trainer.logger.version
        else: # model not traineds:
            success[chosen_activity] = None

    print(success)
    
def train_transformer_GAN(
                    data_func,
                    val_model,
                    start_activity = 1,
                    total_activities = 7,
                    val_iter_size = 3,
                    batch_size = 32,
                    data_size = (27, 100),
                    noise_len = 100,
                    period = 100,
                    max_retries = 2,
                    init_dim_feedforward = 2048,
                    dim_feedforward_exponent = 5,
                    gen_nheads = 5,
                    dis_nheads = 5,
                    gen_num_layers = 1,
                    dis_num_layers = 1,
                    decay = 1,
                    dis_lr = 0.0002,
                    gen_lr = 0.0002,
                    max_epochs = 100,
                    tensorboard_save_dir = "transformer_GAN_logs",
                    tensorboard_name_prefix = "PAMAP2_act_",
                    monitor = "val_f1_score",
                    threshold_value = 0.95,**kwargs):
    # One line training using pytorch lightning abstraction framework
    # data_func is a function which takes in activity_num and gives the data of only those data
    # val_model is the trained classifier to determine if GAN output is identifiable as that activity
    
    success = {}

    for chosen_activity in range(start_activity, total_activities+1):
        try_num = 0
        dim_feedforward = init_dim_feedforward
        train_iter, val_iter = get_dataloaders(data_func, batch_size = batch_size, output_size = total_activities, val_pc = val_iter_size, activity_num = chosen_activity)

        while (try_num < max_retries):
            print("Activity ", chosen_activity,", Try ",try_num)
            model = GAN(val_model = val_model, 
                        generator = Generator_transformer.Generator(noise_len = noise_len, output_size = data_size, nheads = gen_nheads, period = period, dim_feedforward = dim_feedforward, num_layers = gen_num_layers),
                        discriminator = Discriminator_transformer.Discriminator(input_size = data_size, nheads = dis_nheads, period = period, dim_feedforward = dim_feedforward, num_layers = dis_num_layers),
                        val_expected_output = chosen_activity - 1,
                        num_classes = total_activities,
                        noise_len = noise_len,
                        decay = decay,
                        dis_lr = dis_lr,
                        gen_lr = gen_lr,
                       )

            trainer = pl.Trainer(gpus=-1,
                                 max_epochs=max_epochs,
                                 callbacks = [F1_score_check(monitor, threshold_value = threshold_value), 
                                             ], # Early stopping callback
                                 logger = TensorBoardLogger(save_dir = tensorboard_save_dir, name = tensorboard_name_prefix + str(chosen_activity)),
                                 check_val_every_n_epoch = 5,
                                 )
            result = trainer.fit(model, train_iter, val_iter)
            # verify if the model is trained
            if trainer.callback_metrics[monitor] >= threshold_value or result != 1:
                print("Success!")
                success[chosen_activity] = trainer.logger.version
                break
            else: # model not trained
                dim_feedforward *= dim_feedforward_exponent
                try_num += 1
                if try_num == max_retries:
                    success[chosen_activity] = None

    print(success)
    
def train_LSTM_validation_model(
                data_func,
                total_activities = 7,
                val_pc = 0.3,
                batch_size = 20,
                data_size = (27, 100),
                hidden_size = 128,
                conv_filter = (5,9), 
                conv_padding = (2,4),
                lr = 0.0001,
                max_epochs = 100,
                tensorboard_save_dir = "Validation_LSTM_logs",
                tensorboard_name_prefix = "PAMAP2",
                monitor = "val_f1_score",**kwargs):
    # One line training using pytorch lightning abstraction framework
    # data_func is a function which takes in activity_num and gives the data of only those data
    
    train_iter, val_iter,train_weight = get_dataloaders(data_func, batch_size = batch_size, output_size = total_activities, val_pc = val_pc, **kwargs)
        
    net = DeepConvNet(in_channels = data_size[0], input_size = data_size[-1], hidden_size = hidden_size, output_size = total_activities, conv_filter = conv_filter, conv_padding = conv_padding)
    model = Net(model = net, num_classes = total_activities, classes_weight = torch.tensor(train_weight, dtype = torch.float), lr = lr)

    trainer = pl.Trainer(gpus=-1,
                         max_epochs=max_epochs,
                         log_every_n_steps = 200,
                         callbacks = [EarlyStopping(monitor = monitor, patience = 5, mode = "max"),
                                     ModelCheckpoint(monitor = monitor, filename = '{epoch}-{val_loss:.3f}-{'+monitor+':.3f}', mode = 'max'),
                                     ],
                         logger = TensorBoardLogger(save_dir = tensorboard_save_dir, name = tensorboard_name_prefix),
                         stochastic_weight_avg=True
                         )
    trainer.fit(model, train_iter, val_iter)
    
def train_transformer_validation_model(
                data_func,
                total_activities = 7,
                val_pc = 0.3,
                batch_size = 20,
                data_size = (27, 100),
                nhead = 5,
                dim_feedforward = 2048,
                dropout = 0.3,
                num_layer = 1,
                lr = 0.0001,
                max_epochs = 100,
                tensorboard_save_dir = "Validation_transformer_logs",
                tensorboard_name_prefix = "PAMAP2",
                monitor = "val_f1_score",**kwargs):
    # One line training using pytorch lightning abstraction framework
    # data_func is a function which takes in activity_num and gives the data of only those data
    
    train_iter, val_iter,train_weight = get_dataloaders(data_func, batch_size = batch_size, output_size = total_activities, val_pc = val_pc, **kwargs)
        
    net = TransformerClassifier(in_channels = data_size[0], d_model = data_size[-1], output_size = total_activities, nhead = nhead, dim_feedforward = dim_feedforward, dropout= dropout, num_layer = num_layer)
    model = Net(model = net, num_classes = total_activities, classes_weight = torch.tensor(train_weight, dtype = torch.float), lr = lr)

    trainer = pl.Trainer(gpus=-1,
                         max_epochs=max_epochs,
                         log_every_n_steps = 200,
                         callbacks = [EarlyStopping(monitor = monitor, patience = 5, mode = "max"),
                                     ModelCheckpoint(monitor = monitor, filename = '{epoch}-{val_loss:.3f}-{'+monitor+':.3f}', mode = 'max'),
                                     ],
                         logger = TensorBoardLogger(save_dir = tensorboard_save_dir, name = tensorboard_name_prefix),
                         stochastic_weight_avg=True
                         )
    trainer.fit(model, train_iter, val_iter)