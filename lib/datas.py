"""
Functions for loading data

author: @yuningw
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt

###############################
## beta-VAE
###############################

#---------------------------------------------------------------------
def loadData(file, printer=False):
    """
    Read flow field dataset
    
    Args: 
            file    :   (str) Path of database

            printer :   (bool) print the summary of datasets
    
    Returns:

            u_scaled:   (NumpyArray) The scaled data
            
            mean    :   (float) mean of data 
            
            std     :   (float) std of data 

    """

    with h5py.File(file, 'r') as f:
        # u_scaled = f['UV'][:]
        # mean = f['mean'][:]
        '''
        array([[[[0.21258125]],

        [[0.04647906]]]], dtype=float32)
        '''
        # std = f['std'][()]
        velocity = f["Velocity"][:]
        pressure = f["pressure"][:]
        u_scaled = np.concatenate((velocity, pressure), axis=3).astype(np.float32)
        mean = np.mean(u_scaled, axis=0, keepdims=True).astype(np.float32)
        std = np.array([np.std(u_scaled[:, :, :, i]) for i in range(u_scaled.shape[-1])]).astype(np.float32)
        u_scaled = (u_scaled - mean) / std
        # 1000, 128, 128, 2
        # temp = f["Velocity"][:]
        # # 1000, 2, 128, 128
        # u_scaled = np.zeros((temp.shape[0],temp.shape[-1],temp.shape[1], temp.shape[2]))
        # u_scaled[:, 0, :, :] = temp[:, :, :, 0]
        # u_scaled[:, 1, :, :] = temp[:, :, :, 1]
        # u_scaled = u_scaled.astype(np.float32)
        # mean = np.mean(u_scaled, axis=0, keepdims=True).astype(np.float32)
        # std = np.std(u_scaled, axis=0, keepdims=True).astype(np.float32)
        # breakpoint()
    u_scaled = np.moveaxis(u_scaled, -1, 1)
    mean = np.moveaxis(mean, -1, 1)
    # std = np.moveaxis(std, -1, 1)
    std = std.reshape((1, 3, 1, 1))

    
    if printer:
        print('u_scaled: ', u_scaled.shape)
        print('mean: ', mean.shape)
        print('std: ', std)

    # fig, ax = plt.subplots(nrows=10, ncols=3)
    # for i in range(u_scaled.shape[0]):
    #     if i % 100 == 0:
    #         ax[i // 100, 0].imshow(u_scaled[i, 0, :, :])
    #         ax[i // 100, 1].imshow(u_scaled[i, 1, :, :])
    #         ax[i // 100, 2].imshow(u_scaled[i, 2, :, :])
    #         ax[i // 100, 0].axis('off')
    #         ax[i // 100, 1].axis('off')
    #         ax[i // 100, 2].axis('off')
    
    # plt.savefig('timeseries_of_data.png')
    return u_scaled, mean, std


#---------------------------------------------------------------------
def get_vae_DataLoader(d_train, n_train, device, batch_size):
    """
    make tensor data loader for training

    Args:
        d_train: (NumpyArray) Train DataSet 
        
        n_train  : (int) Training samples

        device  : (str) Device
        
        batch_size: (int) Batch size
        

    Return: 
        train_dl, val_dl: The train and validation DataLoader
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if ('cuda' in device):
        train_dl = torch.utils.data.DataLoader(dataset=torch.from_numpy(d_train[:n_train]).to(device),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=0)
        val_dl = torch.utils.data.DataLoader(dataset=torch.from_numpy(d_train[n_train:]).to(device),
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    else:
        train_dl = torch.utils.data.DataLoader(dataset=torch.from_numpy(d_train[:n_train]), batch_size=batch_size,
                                               shuffle=True, pin_memory=True, num_workers=4,
                                               persistent_workers=True)
        val_dl = torch.utils.data.DataLoader(dataset=torch.from_numpy(d_train[n_train:]), batch_size=batch_size,
                                             shuffle=False, pin_memory=True, num_workers=4,
                                             persistent_workers=True)

    return train_dl, val_dl



###############################
## Temporal Prediction
###############################

#---------------------------------------------------------------------
def make_Sequence(cfg,data):
    """
    Generate time-delay sequence data 

    Args: 
        cfg: A class contain the configuration of data 
        data: A numpy array follows [Ntime, Nmode] shape

    Returns:
        X: Numpy array for Input 
        Y: Numpy array for Output
    """

    from tqdm import tqdm 
    import numpy as np 
    if len(data.shape) <=2:
        data    = np.expand_dims(data,0)
    seqLen      = cfg.in_dim
    nSamples    = (data.shape[1]-seqLen)
    X           = np.empty([nSamples, seqLen, data.shape[-1]])
    Y           = np.empty([nSamples, cfg.next_step,data.shape[-1]])
    # Fill the input and output arrays with data
    k = 0
    for i in tqdm(np.arange(data.shape[0])):
        for j in np.arange(data.shape[1]-seqLen- cfg.next_step):
            X[k] = data[i, j        :j+seqLen]
            Y[k] = data[i, j+seqLen :j+seqLen+cfg.next_step]
            k    = k + 1
    print(f"The training data has been generated, has shape of {X.shape, Y.shape}")

    return X, Y


#---------------------------------------------------------------------
def make_DataLoader(X,y,batch_size,
                    drop_last=False,train_split = 0.8):
    """
    make tensor data loader for training

    Args:
        X: Tensor of features
        y: Tensor of target
        batch_size: Batch size
        drop_last: If drop the last batch which does not have same number of mini batch
        train_split: A ratio of train and validation split 

    Return: 
        train_dl, val_dl: The train and validation DataLoader
    """

    from torch.utils.data import DataLoader, TensorDataset,random_split
    try: 
        dataset = TensorDataset(X,y)
    except:
        print("The data is not torch.tenor!")

    len_d = len(dataset)
    train_size = int(train_split * len_d)
    valid_size = len_d - train_size

    train_d , val_d = random_split(dataset,[train_size, valid_size])
    
    train_dl = DataLoader(train_d,batch_size=batch_size,drop_last=drop_last,shuffle=True)
    val_dl = DataLoader(val_d,batch_size=batch_size, drop_last=drop_last,shuffle=True)
    
    return train_dl, val_dl