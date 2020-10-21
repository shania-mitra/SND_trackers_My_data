#!/usr/bin/env python3
# Import Class from utils.py & net.py file
from utils import DataPreprocess, Parameters, Parameters_reduced
from net import SNDNet, MyDataset, digitize_signal
# usful module 
import torch
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm
from IPython import display
import os
import gc # Gabage collector interface (to debug stuff)

# Test to see if cuda is available or not + listed the CUDA devices that are available
try:
    assert(torch.cuda.is_available())
except:
    raise Exception("CUDA is not available")
n_devices = torch.cuda.device_count()
print("\nWelcome!\n\nCUDA devices available:\n")
for i in range(n_devices):
    print("\t{}\twith CUDA capability {}".format(torch.cuda.get_device_name(device=i), torch.cuda.get_device_capability(device=i)))
print("\n")
device = torch.device("cuda", 0)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Turn off interactive plotting: for long run it makes screwing up everything
plt.ioff()

# Here we choose the geometry with 9 time the radiation length
params = Parameters_reduced("4X0")  #!!!!!!!!!!!!!!!!!!!!!CHANGE THE DIMENTION !!!!!!!!!!!!!!!!
processed_file_path = os.path.expandvars("/home/debryas/DS5/ship_tt_processed_data") #!!!!!!!!!!!!!!!!!!!!!CHANGE THE PATH !!!!!!!!!!!!!!!!
step_size = 5000    # size of a chunk
#file_size = 180000  # size of the BigFile.root file
file_size = 120000
n_steps = int(file_size / step_size) # number of chunks

# ------------------------------------------ LOAD THE reindex_TT_df & reindex_y_full PD.DATAFRAME --------------------------------------------------------------------------

chunklist_TT_df = []  # list of the TT_df file of each chunk
chunklist_y_full = [] # list of the y_full file of each chunk

# It is reading and analysing data by chunk instead of all at the time (memory leak problem)
print("\nReading the tt_cleared_reduced.pkl & y_cleared.pkl files by chunk")
#First 2 
outpath = processed_file_path + "/{}".format(0)
chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl")))
chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
outpath = processed_file_path + "/{}".format(1)
chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl")))
chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))

reindex_TT_df = pd.concat([chunklist_TT_df[0],chunklist_TT_df[1]],ignore_index=True)

reindex_y_full = pd.concat([chunklist_y_full[0],chunklist_y_full[1]], ignore_index=True)


for i in tqdm(range(n_steps-2)):  # tqdm: make your loops show a progress bar in terminal
    outpath = processed_file_path + "/{}".format(i+2)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl"))) # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl"))) # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
    reindex_TT_df = pd.concat([reindex_TT_df,chunklist_TT_df[i+2]], ignore_index=True)
    reindex_y_full = pd.concat([reindex_y_full,chunklist_y_full[i+2]], ignore_index=True)

 
     
# reset to empty space
chunklist_TT_df = []
chunklist_y_full = []
nb_of_plane = len(params.snd_params[params.configuration]["TT_POSITIONS"])

#----------------------------------------- Ploting figure of the 6 component of TT_df
'''
index=2
response = digitize_signal(reindex_TT_df.iloc[index], params=params, filters=nb_of_plane)
print("Response shape:",response.shape) # gives  (6, 250, 298) unreduced and (6, 132, 154) for reduced data --> change the network 
number_of_paramater = response.shape[0]
plt.figure(figsize=(18,number_of_paramater))
for i in range(number_of_paramater):
    plt.subplot(1,number_of_paramater,i+1)
    plt.imshow(response[i].astype("uint8") * 255, cmap='gray')
'''

# True value of NRJ/dist for each true electron event
#y = reindex_y_full[["E", "Z","THETA"]]
y = reindex_y_full[["E"]]
NORM = 1. / 100
y["E"] *= NORM
#y["Z"] *= -1
#y["THETA"] *= (180/np.pi)


# reset to empty space
#reindex_y_full = []

# Spliting
print("\nSplitting the data into a training and a testing sample")

indeces = np.arange(len(reindex_TT_df))
train_indeces, test_indeces, _, _ = train_test_split(indeces, indeces, train_size=0.9, random_state=1543)
print("length test_indeces: {0}".format(len(test_indeces)))


#print(len(test_indeces))

def indices_by_condition(df, train_indices, column_name, lower_bound, upper_bound):
     # define dataframe containing only test values
    test_df = df.drop(train_indices)

              # filter out all values where energy is wrong
    filtered_df = test_df[test_df[column_name] >= lower_bound] 
    filtered_df = filtered_df[filtered_df[column_name] <= upper_bound] 
    return filtered_df.index.tolist()

final_test_indeces_1 = indices_by_condition(reindex_y_full,train_indeces,'E', 200,250)
print("length final_test_indeces_1: {0}".format(len(final_test_indeces_1)))

final_test_indeces_2 = indices_by_condition(reindex_y_full, train_indeces, 'E',250, 300)
print("length final_test_indeces_2: {0}".format(len(final_test_indeces_2)))
final_test_indeces_3 = indices_by_condition(reindex_y_full, train_indeces, 'E', 300, 350)
print("length final_test_indeces_3: {0}".format(len(final_test_indeces_3)))
final_test_indeces_4 = indices_by_condition(reindex_y_full, train_indeces, 'E', 350, 400)
print("length final_test_indeces_4: {0}".format(len(final_test_indeces_4))) 
#print(len(final_test_indeces_1))
#print(len(final_test_indeces_2))
#print(len(final_test_indeces_3))
#print(len(final_test_indeces_4))
print(len(train_indeces))


#reset to empty space
reindex_y_full = []

#batch_size = 512
batch_size = 150

train_dataset = MyDataset(reindex_TT_df, y, params, train_indeces, n_filters=nb_of_plane)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset_1 = MyDataset(reindex_TT_df, y, params, final_test_indeces_1, n_filters=nb_of_plane)
test_batch_gen_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_2 = MyDataset(reindex_TT_df, y, params, final_test_indeces_2, n_filters=nb_of_plane)
test_batch_gen_2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_3 = MyDataset(reindex_TT_df, y, params, final_test_indeces_3, n_filters=nb_of_plane)
test_batch_gen_3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_4 = MyDataset(reindex_TT_df, y, params, final_test_indeces_4, n_filters=nb_of_plane)
test_batch_gen_4 = torch.utils.data.DataLoader(test_dataset_4, batch_size=batch_size, shuffle=False, num_workers=0)

# reset to empty space
reindex_TT_df=[]

# Saving the true Energy for the test sample
TrueE_test_1=y["E"][final_test_indeces_1]
TrueE_test_2=y["E"][final_test_indeces_2]
TrueE_test_3=y["E"][final_test_indeces_3]
TrueE_test_4=y["E"][final_test_indeces_4]
np.save("TrueE_test_1.npy",TrueE_test_1)
np.save("TrueE_test_2.npy",TrueE_test_2)
np.save("TrueE_test_3.npy",TrueE_test_3)
np.save("TrueE_test_4.npy",TrueE_test_4)

# Creating the network
#net = SNDNet(n_input_filters=nb_of_plane).to(device)

# Loose rate, num epoch and weight decay parameters of our network backprop actions
#lr = 1e-3
#opt = torch.optim.Adam(net.model.parameters(), lr=lr, weight_decay=0.01)
#num_epochs = 40






# Saving the prediction at each epoch

# Create a directory where to store the prediction files
os.system("mkdir PredE_file")

for i in[39]:
    net = torch.load("9X0_file/" + str(i) + "_9X0_coordconv.pt")
    preds = []
    with torch.no_grad():
        for (X_batch, y_batch) in test_batch_gen_1:
            preds.append(net.predict(X_batch))
    ans = np.concatenate([p.detach().cpu().numpy() for p in preds])
    np.save("PredE_file/" + str(i) + "_PredE_test_1.npy",ans[:, 0])
    print("Save Prediction for epoch "+ str(i))

for i in[39]:
    net = torch.load("9X0_file/" + str(i) + "_9X0_coordconv.pt")
    preds = []
    with torch.no_grad():
        for (X_batch, y_batch) in test_batch_gen_2:
            preds.append(net.predict(X_batch))
    ans = np.concatenate([p.detach().cpu().numpy() for p in preds])
    np.save("PredE_file/" + str(i) + "_PredE_test_2.npy",ans[:, 0])
    print("Save Prediction for epoch "+ str(i))

for i in[39]:
    net = torch.load("9X0_file/" + str(i) + "_9X0_coordconv.pt")
    preds = []
    with torch.no_grad():
        for (X_batch, y_batch) in test_batch_gen_3:
            preds.append(net.predict(X_batch))
    ans = np.concatenate([p.detach().cpu().numpy() for p in preds])
    np.save("PredE_file/" + str(i) + "_PredE_test_3.npy",ans[:, 0])
    print("Save Prediction for epoch "+ str(i))

for i in[39]:
    net = torch.load("9X0_file/" + str(i) + "_9X0_coordconv.pt")
    preds = []
    with torch.no_grad():
        for (X_batch, y_batch) in test_batch_gen_4:
            preds.append(net.predict(X_batch))
    ans = np.concatenate([p.detach().cpu().numpy() for p in preds])
    np.save("PredE_file/" + str(i) + "_PredE_test_4.npy",ans[:, 0])
    print("Save Prediction for epoch "+ str(i))

