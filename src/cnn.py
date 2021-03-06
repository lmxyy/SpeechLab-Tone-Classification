#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

class CNN(nn.Module):
    def __init__(self,conv_dims,hidden_dims,input_dim,connect_conv = None,use_batchnorm = False):
        super(CNN,self).__init__()
        self.num_conv_layers = len(conv_dims)
        self.num_fc_layers = len(hidden_dims)
        self.use_batchnorm = use_batchnorm
        
        convs = []
        conv_params = [{'stride':1,'pad':(conv_dims[i][1]-1)>>1} for i in range(self.num_conv_layers)]
        if connect_conv != None:
            conv_params.append({'stride':1,'pad':(connect_conv[1] - 1)>>1})
        pool_param = {'pool_size': (1,2), 'stride': (1,2)}
        
        for i in range(self.num_conv_layers):
            if i == 0:
                current_in_channels = input_dim[0]
                current_out_channels = conv_dims[i][0]
                current_kernel_size = conv_dims[i][1]
            else:
                current_in_channels = conv_dims[i-1][0]
                current_out_channels = conv_dims[i][0]
                current_kernel_size = conv_dims[i][1]
            if use_batchnorm:
                convs.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels = current_in_channels,
                        out_channels = current_out_channels,
                        kernel_size = current_kernel_size,
                        stride = conv_params[i]['stride'],
                        padding = conv_params[i]['pad']
                    ),
                    nn.BatchNorm2d(
                        current_out_channels,
                        eps = 1e-5,
                        momentum = 0.1
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        kernel_size = pool_param['pool_size'],
                        stride = pool_param['stride']
                    )
                ))
            else:
                convs.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels = current_in_channels,
                        out_channels = current_out_channels,
                        kernel_size = current_kernel_size,
                        stride = conv_params[i]['stride'],
                        padding = conv_params[i]['pad']
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        kernel_size = pool_param['pool_size'],
                        stride = pool_param['stride']
                    )
                ))

        if connect_conv != None:
            if use_batchnorm:
                convs.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels = conv_dims[-1][0],
                            out_channels = connect_conv[0],
                            kernel_size = connect_conv[1],
                            stride = conv_params[i]['stride'],
                            padding = conv_params[i]['pad']
                        ),
                        nn.BatchNorm2d(
                            connect_conv[0],
                            eps = 1e-5,
                            momentum = 0.1
                        ),
                        nn.ReLU()
                    ))
            else:
                convs.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels = conv_dims[-1][0],
                            out_channels = connect_conv[0],
                            kernel_size = connect_conv[1],
                            stride = conv_params[i]['stride'],
                            padding = conv_params[i]['pad']
                        ),
                        nn.ReLU()
                    ))
        self.convs = nn.Sequential(*convs)
                
        affines = []
        for i in range(self.num_fc_layers):
            current_out_features = hidden_dims[i]
            if i == 0:
                if connect_conv != None:
                    current_in_features = (connect_conv[0]*input_dim[1]*input_dim[2])>>(self.num_conv_layers)
                    # current_in_features = (connect_conv[0]*input_dim[1]*input_dim[2])
                else:
                    current_in_features = (conv_dims[-1][0]*input_dim[1]*input_dim[2])>>(self.num_conv_layers)
                    # current_in_features = (conv_dims[-1][0]*input_dim[1]*input_dim[2])
            else:
                current_in_features = hidden_dims[i-1]
            if use_batchnorm:
                affines.append(nn.Sequential(
                        nn.Linear(
                            in_features = current_in_features,
                            out_features = current_out_features
                        ),
                        nn.BatchNorm1d(
                            current_out_features,
                            eps = 1e-5,
                            momentum = 0.1
                        ),
                        nn.ReLU()
                    ))
            else:
                affines.append(nn.Sequential(
                        nn.Linear(
                            in_features = current_in_features,
                            out_features = current_out_features
                        ),
                        nn.ReLU()
                    ))
        self.affines = nn.Sequential(*affines)
        self.out = nn.Softmax(dim = 1)
        

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)  
        x = self.affines(x)
        output = self.out(x)
        return output

