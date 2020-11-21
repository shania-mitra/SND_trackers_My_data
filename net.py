from operator import truediv
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_left
from utils import CM_TO_MUM
from coord_conv import CoordConv


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, pool=False):
        super().__init__()
        self.block = nn.Sequential()
        self.block.add_module("conv", CoordConv(in_channels, out_channels,
                                                kernel_size=(k_size, k_size), stride=1, with_r=True))

        if pool:
            self.block.add_module("Pool", nn.MaxPool2d(2))
        self.block.add_module("BN", nn.BatchNorm2d(out_channels))
        self.block.add_module("Act", nn.ReLU())
        # self.block.add_module("dropout", nn.Dropout(p=0.5))

    def forward(self, x):
        return self.block(x)


class SNDNet(nn.Module):
    def __init__(self, n_input_filters):
        super().__init__()
        self.model = nn.Sequential(
            Block(n_input_filters, 32, pool=True),
            Block(32, 32, pool=True),
            Block(32, 64, pool=True),
            Block(64, 64, pool=True),
            #Block(32, 32, pool=True),
            #Block(128, 128, pool=False),
            Flatten(),
            nn.Linear(256, 1),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            #nn.Linear(1280,2)
        )

    def compute_loss(self, X_batch, y_batch):
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        logits = self.model(X_batch)
        
        l1_loss_tensor = F.l1_loss(logits, y_batch, reduction='none')
        
        norm_l1_loss_tensor = torch.div(l1_loss_tensor,y_batch)
        return norm_l1_loss_tensor.mean()



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


def digitize_signal(event, params, filters=1, margin=1):
    """
    Convert pandas DataFrame to image
    :param event: Pandas DataFrame line
    :param params: Detector configuration
    :param filters: Number of TargetTrackers in the simulation
    :return: numpy tensor of shape (n_filters, H, W).
    """    
    '''
    shape = (filters, 
             int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                         params.snd_params["RESOLUTION"])),
             int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                         params.snd_params["RESOLUTION"])))
    '''
    pixels_H = int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                           params.snd_params["RESOLUTION"]))
    pixels_W = int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                         params.snd_params["RESOLUTION"]))

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
    '''       
#    return response
