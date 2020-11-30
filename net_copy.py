from torchsummary import summary
from operator import truediv
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_left
from utils_copy import CM_TO_MUM
from coord_conv import CoordConv
import pandas as pd

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=2, pool=False):
        super().__init__()
        self.block = nn.Sequential()
#        self.block.add_module("conv", CoordConv(in_channels, out_channels,
#                                                kernel_size=(k_size,k_size), stride=1, with_r=True,padding=1))
        self.block.add_module("conv",nn.Conv2d(in_channels, out_channels,
                                                kernel_size=(k_size, k_size), stride=1, padding=1))
        if pool:
            self.block.add_module("Pool", nn.MaxPool2d((2)))
        self.block.add_module("BN", nn.BatchNorm2d(out_channels))
        self.block.add_module("Act", nn.ReLU())
        self.block.add_module("dropout", nn.Dropout(p=0.5))

    def forward(self, x):
        return self.block(x)


class SNDNet(nn.Module):
    def __init__(self, n_input_filters):
        super().__init__()
        self.model = nn.Sequential(
            Block(n_input_filters, 32, pool=True),
            Block(32, 32, pool=True),
            Block(32, 32, pool=True),
            Block(32, 32, pool=True),
            Block(32, 32, pool=True),
             
            Block(32, 32, pool=True),
            #Block(128, 128, pool=False),
            Flatten(),
            nn.Linear(2176, 1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # nn.Linear(512, 512),
            nn.ReLU(),
            #nn.Linear(1280,2)
        )
        summary(self.model, input_size=(2, 4, 4208))
    def compute_loss(self, X_batch, y_batch):
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        logits = self.model(X_batch)
        
        mse_loss_tensor = F.mse_loss(logits, y_batch, reduction='none')
        
        norm_mse_loss_tensor = torch.div(mse_loss_tensor,y_batch)
        return norm_mse_loss_tensor.mean()



    def predict(self, X_batch):
        self.model.eval()
        return self.model(X_batch.to(self.device))

    @property
    def device(self):
   #     for parameters in self.model.parameters():
    #        print (parameters)
        return next(self.model.parameters()).device


class MyDataset(Dataset):
    """
    Class defines how to preprocess data before feeding it into net.
    """
    def __init__(self, TT_df, y, parameters, data_frame_indices, n_filters):
        """
        :param TT_df: Pandas DataFrame of events
        :param y: Pandas DataFrame of true electron energy and distance
        :param parameters: Detector configuration
        :param data_frame_indices: Indices to train/test on
        :param n_filters: Number of TargetTrackers in the simulation
        """
        self.indices = data_frame_indices
        self.n_filters = n_filters
        self.X = TT_df
        self.y = y
        self.params = parameters

    def __getitem__(self, index):
        return torch.FloatTensor(digitize_signal(self.X.iloc[self.indices[index]],
                                                 self.params,
                                                 filters=self.n_filters)),\
               torch.FloatTensor(self.y.iloc[self.indices[index]])

    def __len__(self):
        return len(self.indices)


def compress_digitize_signal(event, params, filters=1):
    """
    Convert pandas DataFrame to image
    :param event: Pandas DataFrame line
    :param params: Detector configuration
    :param filters: Number of TargetTrackers in the simulation
    :return: numpy tensor of shape (n_filters, H, W).
    """    
    
    shape = (filters, 
             int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                         params.snd_params["RESOLUTION"])),
             int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                         params.snd_params["RESOLUTION"])))
    response = np.zeros(shape)
    '''
    pixels_H = int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                           params.snd_params["RESOLUTION"]))
    pixels_W = int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                         params.snd_params["RESOLUTION"]))

    print(pixels_H)
    shape = (filters, pixels_H, pixels_W)

    response = np.zeros(shape)

    for i in range(len(event['Z'])):
        if np.logical_and(event['Z'][i] >= -3041.0, event['Z'][i]<= -3037.0):
            event['Z'][i] = -3020.0
        elif np.logical_and(event['Z'][i] >= -3032.0, event['Z'][i]<= -3027.0):
            event['Z'][i] = -3015.0
        elif np.logical_and(event['Z'][i] >= -3022.0, event['Z'][i] <= -3017.0):
            event['Z'][i] = -3010.0
        elif np.logical_and(event['Z'][i] >= -3012.0, event['Z'][i] <= -3007.0):
            event['Z'][i] = -3005.0
        else:
            pass
    
    min_z_value = -3020.0
    event['Z'] = event['Z'] - min_z_value
    max_Z_val = np.max(event['Z'])
    event['Z'] = (event['Z']/max_Z_val)*(int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                         params.snd_params["RESOLUTION"]))-1)
    print(event['Z'])
    for x_index,y_index, z_pos in zip(np.floor((event['X'] + params.snd_params["X_HALF_SIZE"]) * CM_TO_MUM /
                                       params.snd_params["RESOLUTION"]).astype(int),
                                      np.floor((event['Y'] + params.snd_params["X_HALF_SIZE"]) * CM_TO_MUM /
                                       params.snd_params["RESOLUTION"]).astype(int),
                                      np.floor( event['Z']).astype(int)):

 
        response[0,z_pos,x_index] += 1
        response[1,z_pos,shape[1] - y_index - 1]+=1
    print(response[0])
    return response
    '''
    for x_index, y_index, z_pos in zip(np.floor((event['X'] + params.snd_params["X_HALF_SIZE"]) * CM_TO_MUM /
                                                params.snd_params["RESOLUTION"]).astype(int),
                                       np.floor((event['Y'] + params.snd_params["Y_HALF_SIZE"]) * CM_TO_MUM /
                                                params.snd_params["RESOLUTION"]).astype(int),
                                       event['Z']):
        response[params.tt_map[bisect_left(params.tt_positions_ravel, z_pos)],
                 shape[1] - y_index - 1,
                 x_index] += 1                
           
    return response

def digitize_signal(event, params, filters=1):

    shape = (filters,4,int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                                   params.snd_params["RESOLUTION"])))

    response = np.zeros(shape)

    first_layer_x = []
    first_layer_y = []
    second_layer_x = []
    second_layer_y = []
    third_layer_x = []
    third_layer_y = []
    fourth_layer_x = []
    fourth_layer_y = []

    for i in range(len(event['Z'])):
        if np.logical_and(event['Z'][i] >= -3041.0, event['Z'][i]<= -3037.0):
            first_layer_x.append(event['X'][i])
            first_layer_y.append(event['Y'][i])
        elif np.logical_and(event['Z'][i] >= -3032.0, event['Z'][i]<= -3027.0):
            second_layer_x.append(event['X'][i])
            second_layer_y.append(event['Y'][i])
        elif np.logical_and(event['Z'][i] >= -3022.0, event['Z'][i] <= -3017.0):
            third_layer_x.append(event['X'][i])
            third_layer_y.append(event['Y'][i])
        elif np.logical_and(event['Z'][i] >= -3012.0, event['Z'][i] <= -3007.0):
            fourth_layer_x.append(event['X'][i])
            fourth_layer_y.append(event['Y'][i])
        else:
            pass
    event_comp_x = pd.DataFrame({'z1': pd.Series(first_layer_x), 'z2': pd.Series(second_layer_x), 'z3': pd.Series(third_layer_x), 'z4': pd.Series(fourth_layer_x) })
    event_comp_x.fillna(0, inplace=True)
    event_comp_y = pd.DataFrame({'z1': pd.Series(first_layer_y), 'z2': pd.Series(second_layer_y), 'z3': pd.Series(third_layer_y), 'z4': pd.Series(fourth_layer_y) })
    event_comp_y.fillna(0, inplace=True)

    
    for x_one, x_two, x_three, x_four in zip(np.floor((event_comp_x['z1'] + params.snd_params["X_HALF_SIZE"])*CM_TO_MUM/
                                                       params.snd_params["RESOLUTION"]).astype(int), 
                                             np.floor((event_comp_x['z2'] + params.snd_params["X_HALF_SIZE"])*CM_TO_MUM/
                                                       params.snd_params["RESOLUTION"]).astype(int), 
                                             np.floor((event_comp_x['z3'] + params.snd_params["X_HALF_SIZE"])*CM_TO_MUM/
                                                       params.snd_params["RESOLUTION"]).astype(int),
                                             np.floor((event_comp_x['z4'] + params.snd_params["X_HALF_SIZE"])*CM_TO_MUM/
                                                       params.snd_params["RESOLUTION"]).astype(int)):
#        print(x_one)
#        print(x_two)
#        print(x_three)
#        print(x_four)
        response[0,0,x_one] +=1
        response[0,1, x_two] += 1
        response[0,2,x_three] +=1
        response[0,3,x_four] += 1

    for y_one, y_two, y_three, y_four in zip (np.floor((event_comp_y['z1'] + params.snd_params["Y_HALF_SIZE"])*CM_TO_MUM/
                                                        params.snd_params["RESOLUTION"]).astype(int),
                                              np.floor((event_comp_y['z2'] + params.snd_params["Y_HALF_SIZE"])*CM_TO_MUM/
                                                        params.snd_params["RESOLUTION"]).astype(int),
                                              np.floor((event_comp_y['z3'] + params.snd_params["Y_HALF_SIZE"])*CM_TO_MUM/
                                                        params.snd_params["RESOLUTION"]).astype(int),
                                              np.floor((event_comp_y['z4'] + params.snd_params["Y_HALF_SIZE"])*CM_TO_MUM/
                                                        params.snd_params["RESOLUTION"]).astype(int)):
        response[1,0,shape[1] - y_one - 1] += 1
        response[1,1,shape[1] - y_two - 1] += 1
        response[1,2,shape[1] - y_three - 1] += 1
        response[1,3,shape[1] - y_four - 1] += 1
    response = (response>=1).astype(int)    
#    print(response)
    return response    
