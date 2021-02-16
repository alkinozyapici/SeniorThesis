"""
Author: Rex Geng

quantization API for FX training

todo: you can use this template to write all of your quantization code

Modified by Alkin Ozyapici from the original document of Rex Geng.

The Conv and Linear Layers model the Deep In Memory Architecture developed by Naresh Shanbhag.
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BW=6
BX=6
VBLMAX = 0.8 
T0 = 100e-12
kn = 220e-6
Vt = 0.4
alpha = 1.8
CBL = 270e-15
VWL = 0.9
Icell = kn*np.power(VWL-Vt,alpha) # Ideal cell current of the discharge path
delta_VBL_LSB = T0*Icell/CBL #The voltage difference on VBL created by the LSB
kclip = VBLMAX/delta_VBL_LSB 
sigma_Vt = 23.8e-3
sigma_D = alpha*sigma_Vt/(VWL-Vt)

def DIMA(model):
    """
    generate a quantized model
    :param model:
    :param args:
    :return:
    """
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            bias = False
            if m.bias is not None:
                bias = True
            conv_args = {'in_channels': m.in_channels, 'out_channels': m.out_channels, 'kernel_size': m.kernel_size,
                         'stride': m.stride, 'padding': m.padding, 'groups': m.groups, 'bias': bias, 'sigma_D' : sigma_D,
                         'layer_index' : n}
            conv = DIMAConv2d(**conv_args)
            rsetattr(model, n, conv)
            print('CONV layer ' + n + ' quantized' )

        if isinstance(m, nn.Linear):
            bias = False
            if m.bias is not None:
                bias = True
            fc_args = {'in_features': m.in_features, 'out_features': m.out_features, 'bias': bias}
            init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data if bias else None}
            lin = DIMALinear(**fc_args)
            print('FC layer ' + n + 'quantized')
            rsetattr(model, n, lin)
    return model


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))



def quantizeWeight(W,BW):
    W = torch.min(W,(1.0-(2**(-(BW-1.0))))*torch.ones_like(W))
    Wbs = []
    Wbi = torch.lt(W,torch.zeros_like(W)).float()
    Wbs.append(Wbi)
    W = (W + Wbi)
    for i in range(BW-1):
        Wbi = torch.ge(W,0.5*torch.ones_like(W)).float()
        Wbs.append(Wbi)
        W = 2.0*W - Wbi
    carry = torch.ge(W,0.5*torch.ones_like(W)).float()
    for i in range(BW):#-1):
        j = BW-1-i
        Wbs[j] = Wbs[j]+carry
        carry = torch.gt(Wbs[j],1.5*torch.ones_like(Wbs[j])).float()
        Wbs[j] = Wbs[j]*torch.ne(Wbs[j],2.0*torch.ones_like(Wbs[j]))
    return Wbs

def reconstructWeight(Wbs,BW):
    W = torch.zeros_like(Wbs[0])
    for j in range(BW):
        multiplier = (0.5)**j
        if (j == 0):
            multiplier = -1.0
        W += Wbs[j] * multiplier
    return W

def quantize_activations(input, layer_index):
    if(layer_index != 0):
        input = torch.clamp(input,0,6) / 6
        input = 6 * torch.min(round_f(input*(2**BX))*(2**(-BX)) ,(1.0-(2**(-BX)))*torch.ones_like(input))
    else:
        input = torch.min(round_f(input*(2**BX))*(2**(-BX)) ,(1.0-(2**(-BX)))*torch.ones_like(input))
    return input

def quantize_outputs(output):
    output = torch.clamp(output,-6,6)
    output = torch.min(round_f((output/6)*(2**(BW-1.0)))*(2.0**(1.0-BW)),(1.0-(2.0**(1.0-BW)))*torch.ones_like(output))
    output = output * 6
    return output 
            
def round_f(x): #rounds a number to the nearest integer with STE for gradients
    x_r = torch.round(x)
    x_g = x
    return (x_r - x_g).detach() + x_g

class DIMAConv2d(nn.Conv2d):
    def __init__(
        self,
        sigma_D = 0,
        layer_index = 0,
        *kargs,
        **kwargs
    ):
        super(DIMAConv2d, self).__init__(*kargs,**kwargs)
        self.layer_index = layer_index
        self.noise = np.random.normal(0,sigma_D,(BW,self.weight.size()[0],self.weight.size()[1]
                                                 ,self.weight.size()[2],self.weight.size()[3]))
        
    
    def quantize_weights(self):
        weight_q = quantizeWeight(self.weight.data,BW)
        for b in range(BW-1):
            weight_q[b+1] = weight_q[b+1]*(1+self.noise[b])
        weight = reconstructWeight(weight_q,BW)
        Wmax = kclip*np.power(2.0,-(BW-1))
        weight = torch.clamp(weight,-Wmax,Wmax)
        return (weight - self.weight).detach() + self.weight
        
    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.quantize_weights()
        input = quantize_activations(input,self.layer_index)
        output_h = int((((input.size()[2] + (2 * self.padding[0]) - 
                            ( self.dilation[0] * (self.kernel_size[0] - 1) ) - 1 )/ self.stride[0]) + 1) // 1)
        output_w = int((((input.size()[3] + (2 * self.padding[1]) - 
                            ( self.dilation[1] * (self.kernel_size[1] - 1) ) - 1 )/ self.stride[1]) + 1) // 1)
        
        if(self.kernel_size[0]*self.kernel_size[1]*self.in_channels > 256):
            weights = []
            inputs = []
            val = ((self.kernel_size[0]*self.kernel_size[1]*self.in_channels) // 256) + 1
            coeff = self.in_channels // val
            for i in range (val):
                if(i != val-1):
                    temp_weight = torch.zeros_like(weight)
                    temp_input = torch.zeros_like(input)
                    temp_weight[:,i*coeff:(i+1)*coeff,:,:] = weight[:,i*coeff:(i+1)*coeff,:,:]
                    weights.append(temp_weight)
                    temp_input[:,i*coeff:(i+1)*coeff,:,:] = input[:,i*coeff:(i+1)*coeff,:,:]
                    inputs.append(temp_input)
                else:
                    temp_weight = torch.zeros_like(weight)
                    temp_input = torch.zeros_like(input)
                    temp_weight[:,i*coeff:,:,:] = weight[:,i*coeff:,:,:]
                    weights.append(temp_weight)
                    temp_input[:,i*coeff:,:,:] = input[:,i*coeff:,:,:]
                    inputs.append(temp_input)
            output = torch.zeros((input.size()[0],self.out_channels,output_h,output_w))
            for i in range (val):
                out = self._conv_forward(inputs[i],weights[i])
                output += self.quantize_outputs(out)  

        else:
            output = self._conv_forward(input, weight)
            output = quantize_outputs(output)
        return output

class DIMALinear(nn.Linear):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, var: sigma_D = 0, layer_index = 0) -> None:
        super(DIMALinear, self).__init__(in_features,out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.noise = torch.normal(0,sigma_D,(BW,self.weight.size()[0],self.weight.size()[1]))
        self.bias_noise = torch.normal(0,sigma_D,(BW, self.bias.size()[0]))
        self.layer_index = layer_index
    
    def quantize_weights(self):
        weight_q = quantizeWeight(self.weight.data,BW)
        for b in range(BW-1):
            weight_q[b+1] = weight_q[b+1]*(1+self.noise[b])
        weight = reconstructWeight(weight_q,BW)
        Wmax = kclip*np.power(2.0,-(BW-1))
        weight = torch.clamp(weight,-Wmax,Wmax)
        return (weight - self.weight).detach() + self.weight
    
    def quantize_bias(self):
        bias_q = quantizeWeight(self.bias.data,BW)
        for b in range(BW-1):
            bias_q[b+1] = bias_q[b+1]*(1+self.bias_noise[b])
        bias = reconstructWeight(bias_q,BW)
        Bmax = kclip*np.power(2.0,-(BW-1))
        bias = torch.clamp(bias,-Bmax,Bmax)
        return (bias - self.bias).detach() + self.bias
        
    def forward(self, input: Tensor) -> Tensor:
        weight = self.quantize_weights()
        if(self.bias != None):
            bias = self.quantize_bias()
        input = quantize_activations(input, self.layer_index)
        if(self.weight.size()[1] > 256):
            inputs = []
            weights = []
            val = (self.weight.size()[1] // 256) + 1
            for i in range (val):
                if(i != val-1):
                    inputs.append(input[:,(i*256):(256+(i*256))])
                    weights.append(weight[:,(i*256):(256+(i*256))])
                else:
                    m = nn.ZeroPad2d((0,(256*(i+1))-self.weight.size()[1],0,0))
                    temp_w = m(weight[:,(i*256):])
                    temp_i = m(input[:,(i*256):])
                    inputs.append(temp_i)
                    weights.append(temp_w)
            output = torch.zeros(input.size()[0],self.weight.size()[0])
            for i in range (val):
                if(i == val - 1 and self.bias != None):
                    output += self.quantize_outputs(F.linear(inputs[i],weights[i], bias))
                else:
                    output += self.quantize_outputs(F.linear(inputs[i],weights[i]))                    
    
        else:
            output = F.linear(input, weight, bias)
            output = quantize_outputs(output)
        return output