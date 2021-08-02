import numpy as np

# load the seeds:
seeds = np.load('seeds.npy')

# select a seed:
SEED = seeds[0] 


import torch
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np
np.random.seed(SEED)

from utils import score, get_tetris_data, build_mlgp, build_vanilla, build_baseline, save_checkpoint



if __name__ == '__main__':

    # get the data:
    (Xtrain, Ytrain), (Xval, Yval) = get_tetris_data(total_size=10000, train_size=1000, shuffle_data=True, distortion=0.0)

    # or, e.g., for the noisy theta-split experiment:
    # (Xtrain, Ytrain), (Xval, Yval) = get_tetris_data(total_size=10000, train_size=1000, shuffle_data=True, distortion=0.2,
    #                                                  theta_train=[[0.0, 1/2], [1/8, 5/8]], theta_test=[[1/8, 5/8], [1/2, 1.0]])

    output_dim = len(set(Ytrain.numpy()))

    # set the seed here:
    torch.manual_seed(SEED)


    # select and build the model:
    model_name = 'mlgp_clean'

    ##### 1) Multilayer Geometric Perceptron (ours)
    model = build_mlgp(input_shape=Xtrain.shape[1:], output_dim=output_dim, hidden_layer_sizes=[4], bias=False)

    ##### 2) Baseline (Multilayer Hypersphere Perceptron)
    ## NOTE:    if the baseline is to be used, uncomment the following lines to reshape the data appropriately:
    # sample_size = torch.tensor(Xtrain[0].shape).prod().item()
    # Xtrain, Xval = Xtrain.reshape(-1, 1, sample_size), Xval.reshape(-1, 1, sample_size)
    # model = build_baseline(input_shape=Xtrain.shape[1:], output_dim=output_dim, hidden_layer_sizes=[5], bias=False)

    ##### 3) Vanilla Multilayer Perceptron
    # model = build_vanilla(input_shape=Xtrain.shape[1:], output_dim=output_dim, hidden_layer_sizes=[6], bias=True, activation=nn.functional.relu)  


    # print(model)
    print('total number of parameters:', sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print()

    if torch.cuda.is_available():
        model = model.cuda()
        Xtrain, Ytrain = Xtrain.float().cuda(), Ytrain.cuda()
        Xval, Yval = Xval.float().cuda(), Yval.cuda()


    # define the loss and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20000
    batch_size = len(Xtrain)
    n_batches = len(Xtrain) // batch_size


    # train the model:
    for i in range(epochs): 
        for j in range(n_batches):          
            Xbatch = Xtrain[j*batch_size:(j+1)*batch_size,]
            Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]

            y_pred = model(Xbatch)
            loss = criterion(y_pred, Ybatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred.detach_()
            acc = score(y_pred, Ybatch)

            y_val_pred = model(Xval)
            y_val_pred.detach_()
            val_loss = criterion(y_val_pred, Yval)
            val_acc = score(y_val_pred, Yval)

            if i % 500 == 0:
                print('epoch: %d,  batch: %d,  cost: %.3f,  val_cost: %.3f,  acc:  %.3f,  val_acc: %.3f' % (i, j, loss.item(), val_loss.item(), acc, val_acc))

    print('epoch: %d,  batch: %d,  cost: %.3f,  val_cost: %.3f,  acc:  %.3f,  val_acc: %.3f' % (i, j, loss.item(), val_loss.item(), acc, val_acc))


    # save the model:
    # model_name = '[model_type]_[data_type]'
    save_checkpoint(
        save_dir='pretrained_models',
        state={
            'model': model, 
            'name': model_name,
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'seed': SEED,
        }
    )
