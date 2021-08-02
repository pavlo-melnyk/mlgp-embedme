import torch
import torch.nn as nn

import numpy as np



class PointCMLP(nn.Module):
    '''A single class to create all types of models in the experiments.'''
    
    def __init__(self, input_shape, output_dim, hidden_layer_sizes=[], activation=lambda x: x, bias=False, version=1):
        ''' 
        Args: 
            input_shape:        a list/tuple of 2 integers; the size of one input sample, i.e., (n_rows, n_columns);
                                the model input is, however, expected to be a 3D array of shape (n_samples, n_rows, n_columns);
            hidden_layer_sizes: a list of integers containing the number of hidden units in each layer;
            activation:         activation function, e.g., nn.functional.relu;
            output_dim:         integer; the number of output units.
            version:            either 0 or 1: 
                                0 to create a vanilla MLP (the input will be vectorized in the forward function);
                                1 to create the proposed MLGP or the baseline MLHP.
                                For the former, the embedding of the input is row-wise.
                                In order to create the latter, one needs to vectorize each sample in the input, i.e.,
                                reshape the input to (n_samples, 1, n_rows*n_columns).
        '''
        super().__init__()

        self.input_shape = input_shape
        self.f = activation
        
        # create hidden layers:
        hidden_layers = []

        if version == 0:
            # for vanilla MLP:
            M1 = np.prod(input_shape)
            for M2 in hidden_layer_sizes:
                layer = nn.Linear(M1, M2, bias=bias)
                hidden_layers.append(layer)
                M1 = M2

            self.hidden_layers = nn.ModuleList(hidden_layers)

            # the output layer:
            try:
                self.out_layer = nn.Linear(M2, output_dim, bias=bias)   
            except UnboundLocalError:
                self.out_layer = nn.Linear(M1, output_dim, bias=bias)                   

            self.forward = self.forward_0


        elif version == 1:
            # for the proposed MLGP and the baseline MLHP

            # for input_shape = (k, 3), e.g., 3D shape coordinates, 
            # each row of the input sample (R^3) is embedded in R^5 (ME^3);
            # the resulting (k x 5)-array is vectorized row-wise and fed 
            # to the first layer;
            # each subsequent hidden layer output R^n is first embedded in R^(n+2)
            # and then fed to the next layer

            M1 = input_shape[0] * (input_shape[1] + 2)
            for M2 in hidden_layer_sizes:
                layer = nn.Linear(M1, M2, bias=bias)
                hidden_layers.append(layer)
                M1 = M2 + 2

            self.hidden_layers = nn.ModuleList(hidden_layers)

            #  the output layer:
            try:
                self.out_layer = nn.Linear(M2 + 2, output_dim, bias=bias)   
            except UnboundLocalError:
                self.out_layer = nn.Linear(M1, output_dim, bias=bias)   

            self.forward = self.forward_1
                        
        
    def forward_0(self, x):
        # for the vanilla MLP

        # vectorize each sample:
        x = x.view(-1, x.shape[1]*x.shape[2]) 

        for layer in self.hidden_layers:
            x = self.f(layer(x))
        x = self.out_layer(x)

        return x


    def forward_1(self, x): 
        # for the proposed MLGP and the baseline MLHP   

        embed_term_1 = -torch.ones(len(x), x.shape[1], 1).float()
        embed_term_2 = -torch.sum(x**2, axis=2) / 2 

        if torch.cuda.is_available():
            embed_term_1 = embed_term_1.cuda()

        x = torch.cat((x, embed_term_1, embed_term_2.view(-1, x.shape[1], 1)), dim=2)
        x = x.view(-1, x.shape[1]*x.shape[2]) 

        for layer in self.hidden_layers:
            x = self.f(layer(x))

            embed_term_1 = -torch.ones(len(x), 1).float()
            embed_term_2 = -torch.sum(x**2, axis=1).view(-1, 1) / 2

            if torch.cuda.is_available():
                embed_term_1 = embed_term_1.cuda()

            x = torch.cat((x, embed_term_1, embed_term_2), dim=1)

        x = self.out_layer(x)

        return x
