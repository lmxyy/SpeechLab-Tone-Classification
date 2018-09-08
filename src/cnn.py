#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

class CNN(nn.Module):
    def __init__(self,conv_dims,hidden_dims,input_dim,connvect_conv = None,use_batchnorm = False):
        super(CNN,self).__init__()
        self.num_conv_layers = len(conv_dims)
        self.num_fc_layers = len(hidden_dims)
        self.use_batchnorm = use_batchnorm
        
        self.convs = []
        conv_params = [{'stride':1,'pad':(conv_dims[i][1]-1)>>1} for i in range(self.num_conv_layers)]
        if use_connect_conv != None:
            conv_params.append({'stride':1},'pad':(connect_conv[1] - 1)>>1)
        pool_param = {'pool_size': (2,2), 'stride': 2}
        
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
                self.convs.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels = current_in_channels,
                        out_channels = current_out_channels,
                        kernel_size = current_kernel_size,
                        stride = conv_params[i]['stride'],
                        padding = conv_params[i]['pad']
                    ),
                    nn.BatchNorm2D(
                        current_out_channels,
                        eps = 1e-5,
                        momentum = 0.1
                    ),
                    nn.ReLu(),
                    nn.MaxPool2d(
                        kernel_size=self.pool_param['pool_size'],
                        stride = self.pool_param['stride']
                    )
                ))
            else:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels = current_in_channels,
                        out_channels = current_out_channels,
                        kernel_size = current_kernel_size,
                        stride = conv_params[i]['stride'],
                        padding = conv_params[i]['pad']
                    ),
                    nn.ReLu(),
                    nn.MaxPool2d(
                        kernel_size=self.pool_param['pool_size'],
                        stride = self.pool_param['stride']
                    )
                ))

        if use_connect_conv != None:
            if use_batchnorm:
                self.convs.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels = conv_dims[-1][0],
                            out_channels = connect_conv[0],
                            kernel_size = connect_conv[1],
                            stride = conv_params[i]['stride'],
                            padding = conv_params[i]['pad']
                        ),
                        nn.BatchNorm2D(
                            connect_conv[0],
                            eps = 1e-5,
                            momentum = 0.1
                        ),
                        nn.ReLu()
                    ))
            else:
                self.convs.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels = conv_dims[-1][0],
                            out_channels = connect_conv[0],
                            kernel_size = connect_conv[1],
                            stride = conv_params[i]['stride'],
                            padding = conv_params[i]['pad']
                        ),
                        nn.ReLu()
                    ))

        self.affines = []
        for i in range(self.num_fc_layers):
            current_out_features = hidden_dims[i]
            if i == 0:
                if use_connect_conv != None:
                    current_in_features = connect_conv[0]*input_dim[1]*input_dim[2])>>(2*self.num_conv_layers)
                else:
                    current_in_features = conv_dims[-1][0]*input_dim[1]*input_dim[2])>>(2*self.num_conv_layers)
            else:
                current_in_features = hidden_dims[i-1]
            if use_batchnorm:
                self.convs.append(nn.Sequential(
                        nn.Linear(
                            in_features = current_in_features,
                            out_features = current_out_features
                        ),
                        nn.BatchNorm2D(
                            current_out_features,
                            eps = 1e-5,
                            momentum = 0.1
                        ),
                        nn.ReLu()
                    ))
            else:
                self.convs.append(nn.Sequential(
                        nn.Linear(
                            in_features = current_in_features,
                            out_features = current_out_features
                        ),
                        nn.ReLu()
                    ))

        self.out = nn.Softmax()

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        for affine in self.affines:
            x = affine(x)
        x = x.view(x.size(0), -1)  
        output = self.out(x)
        return output

