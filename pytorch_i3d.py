import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import math
import os
import sys
from collections import OrderedDict


class Sync(nn.Module):
    def __init__(self,featd):
        super(Sync,self).__init__()
        layer1_num = 128
        layer2_num = 256


        self.state_size = featd

        # self.layers = nn.Sequential(
		# 	nn.Linear(self.state_size,layer1_num),
        #     nn.BatchNorm1d(layer1_num),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(layer1_num,1)
		# 	)
        self.layers = nn.Sequential(
			nn.Linear(self.state_size,layer1_num),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(layer1_num,1)
			)
        for layer in self.layers.modules():
            if isinstance(layer, nn.Linear):
                # init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                # layer.bias.data.fill_(0.0)
                # init.normal_(layer.weight, mean=0, std=0.1)
                # layer.weight.data.clamp_(-2, 2)
                layer.bias.data.fill_(0.0)
                init.trunc_normal_(layer.weight, mean=0.0, std=0.1, a=- 2.0, b=2.0)

    def forward(self,x):
        x = x.squeeze(2)
        x = x.view(x.size(0),-1)
        return self.layers(x)

class Domain_Classifier(nn.Module):
    def __init__(self,featd,context=False):
        super(Domain_Classifier, self).__init__()
        self.state_size=featd
        # self.layers = nn.Sequential(
		# 	nn.Linear(self.state_size,256),
        #     nn.BatchNorm1d(256),
		# 	nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(256,128),
        #     nn.BatchNorm1d(128),
		# 	nn.LeakyReLU(negative_slope=0.2),
		# 	nn.Linear(128,1)    
		# 	)
        self.layers = nn.Sequential(
			nn.Linear(self.state_size,128),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(128,1)    
			)
        for layer in self.layers.modules():
            if isinstance(layer, nn.Linear):
                # init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                # layer.bias.data.fill_(0.0)
                # init.normal_(layer.weight, mean=0, std=0.1)
                # layer.weight.data.clamp_(-2, 2)
                layer.bias.data.fill_(0.0)
                init.trunc_normal_(layer.weight, mean=0.0, std=0.1, a=- 2.0, b=2.0)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.layers(x)
        #print(x)
        return torch.sigmoid(x)



# class Domain_Classifier(nn.Module):
#     def __init__(self,context=False):
#         super(Domain_Classifier, self).__init__()

#         self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1,
#                                padding=0, bias=False)
#         self.conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1,
#                                padding=0, bias=False)
#         self.conv3 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
#                                padding=0, bias=False)
#         self.conv4 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
#                                padding=0, bias=False)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
#         self.bn = nn.BatchNorm2d(1, eps=0.001, momentum=0.01)
#         self.context = context
#         self._init_weights()
#     def _init_weights(self):
#         def normal_init(m, mean, stddev, truncated=True):
#             """
#             weight initalizer: truncated normal and random normal.
#             """
#             # x is a parameter
#             if truncated:
#                 m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
#             else:
#                 m.weight.data.normal_(mean, stddev)
#           #m.bias.data.zero_()
#         normal_init(self.conv1, 0, 0.1)
#         normal_init(self.conv2, 0, 0.1)
#         normal_init(self.conv4, 0, 0.1)
#         normal_init(self.conv3, 0, 0.1)
#     def forward(self, x):
#         #print("x.shape:",x)
#         x = self.leaky_relu(self.conv1(x))
#         #print("x.shape in conv1 in pixelD:",x.shape)
#         x = self.leaky_relu(self.conv2(x))
#         x = self.leaky_relu(self.conv3(x))
#         #print("x.shape in conv2 in pixelD:",x.shape)
#         #x = self.leaky_relu(self.conv3(x))
#         #print("x.shape in conv3 in pixelD:",x.shape)
#         if self.context:
#             feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
#             #print("feat.shape",feat.shape)
#             # feat = x
#             #x = F.sigmoid(self.conv4(x))
#             x = torch.sigmoid(self.conv4(x))
#             #print("x.shape in conv4 in pixelD:",x.shape)
#             return x, feat  # torch.cat((feat1,feat2),1)#F
#         else:
#             x = self.conv4(x)
#             x = self.bn(x)
#             #print("x.shape:",x)
#             return torch.sigmoid(x)#F.sigmoid(x)



class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):

        #print("unin3d",x.size())
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class Tostate(nn.Module):
    def __init__(self):
        super(Tostate, self).__init__()

        # self.pool = nn.AvgPool3d(kernel_size=[2, 3, 3],
        #                              stride=(1, 1, 1))                     

        # self.pool1 = nn.AvgPool3d(kernel_size=[1, 3, 3],
        #                              stride=(1, 2, 2))


        # self.pool = nn.Conv3d(in_channels=384+384+128+128, out_channels=1024,
        #                      kernel_size=(1, 3, 3),
        #                      padding=0,
        #                      stride=(1, 1, 1))
                        
        # self.pool1 = nn.AvgPool3d(kernel_size=[2, 3, 3],
        #                             stride=(1, 2, 2))

        # self.bn = nn.BatchNorm3d(1024, eps=0.001, momentum=0.01)

        self.pool1 = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                    stride=(1, 1, 1))
        
        
        
                
                                     

    def forward(self,x):
     #

        #pool1 = self.pool(x)
        #pool1 = self.bn(pool1)

        # pooled = self.pool1(pool1)


        pooled = self.pool1(x)

        
        # else:
        #     pool1 = self.pool2(x)
        #     pooled = self.pool1(pool1)

        return pooled


    

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'features',
        'dropout',
        'logits_aux',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,aux_classifier=False,flip_classifier_gradient=False, flip_weight=1.0):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self._aux_classifier = aux_classifier
        self.flip_classifier_gradient = flip_classifier_gradient
        self.flip_weight = flip_weight
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'features'
    
        self.end_points[end_point] = Tostate()
        if self._final_endpoint == end_point: return

        end_point = 'dropout'

        self.dropout = nn.Dropout(dropout_keep_prob)

        self.end_points['dropout'] = self.dropout

        # end_point = 'Upsample'

        # self.Upsample = Unit3D(in_channels=384+384+128+128, output_channels=384+384+128+128,
        #                      kernel_shape=[16, 5, 5],
        #                      padding=0,
        #                      stride=[1, 1, 1],
        #                      activation_fn=None,
        #                      use_batch_norm=False,
        #                      use_bias=True,
        #                      name='Upsample')

        # self.end_points['Upsample'] = self.Upsample
        
        end_point = 'logits_aux'

        logits_aux = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits_aux')
        self.end_points['logits_aux'] = logits_aux
            


        end_point = 'Logits'
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')


        self.avg_pool = self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        
        # else:
        #     self.avg_pool = self.avg_pool = nn.AvgPool3d(kernel_size=[1, 7, 7],
        #                              stride=(1, 1, 1))
            

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        #print("in forward",x)
        for end_point in self.VALID_ENDPOINTS:
            #print("end_point",end_point)
            if end_point in self.end_points:

                if end_point=='features':

                    features = self._modules[end_point](x)
                    #print(end_point,features.size())
                    continue

                if self.flip_classifier_gradient and end_point=='Logits':
                    x = flip_gradient(x, self.flip_weight)
                if self._aux_classifier and end_point=='logits_aux':
                    aux_logits = x 
                    aux_logits = self._modules[end_point](aux_logits)
                    if self._spatial_squeeze:
                        aux_logits = aux_logits.squeeze(3).squeeze(3)
                elif end_point=='logits_aux':
                    continue
                
                x = self._modules[end_point](x) # use _modules to work with dataparallel
                
        #print("x",x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        #print("x",x)
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        if self._aux_classifier:
            return logits, features, aux_logits
        else:
            return logits, features
        

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)