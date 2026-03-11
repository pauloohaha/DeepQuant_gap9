import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from . import fastba

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import  GatedResidual, SoftAgg, GAP9SoftmaxAgg, GradientClip
from .deeploy_placeholder import CustomLayerNorm, liner_implementation, TCneighborgather

autocast = torch.cuda.amp.autocast

DIM = 384


class UpdateONNX(nn.Module):
    def __init__(self, p, dim=DIM, use_pyramid=True, use_softagg=True, use_gru=True, use_ctx_features=True, use_temp=True):
        super(UpdateONNX, self).__init__()
        self.dim = dim
        self.use_pyramid = use_pyramid
        self.use_softagg = use_softagg
        self.use_gru = use_gru
        self.use_temp = use_temp
        self.use_ctx_features = use_ctx_features
        
        if self.use_temp:
            self.c1 = nn.Sequential(
                liner_implementation(dim, dim),
                nn.ReLU(inplace=True),
                liner_implementation(dim, dim))

            self.c2 = nn.Sequential(
                liner_implementation(dim, dim),
                nn.ReLU(inplace=True),
                liner_implementation(dim, dim))
        else:
            #skip the temporal convolution in this case
            self.c1 = nn.Identity()
            self.c2 = nn.Identity()

        self.norm = CustomLayerNorm(dim, eps=1e-3)

        if self.use_softagg:
            self.agg_kk = GAP9SoftmaxAgg(0)
            self.agg_ij = GAP9SoftmaxAgg(1)
        else:
            self.agg_kk = nn.Identity()
            self.agg_ij = nn.Identity()

        if use_gru:
            self.gru = nn.Sequential(
                CustomLayerNorm(dim, eps=1e-3),
                GatedResidual(dim),
                CustomLayerNorm(dim, eps=1e-3),
                GatedResidual(dim),
            )
        else:
            self.gru = nn.Sequential(
                CustomLayerNorm(dim, eps=1e-3),
                liner_implementation(dim, dim),
                nn.ReLU(inplace=True),
                liner_implementation(dim, dim),
                CustomLayerNorm(dim, eps=1e-3),
                liner_implementation(dim, dim),
                nn.ReLU(inplace=True),
                liner_implementation(dim, dim)
            )
            

        if(use_pyramid == False) :
            #SMALLER VERSION : ONLY THE HIGH RESOLUTION MATCHING FEATURES ARE USED (1/4 ON, 1/16 NOT ADDED) --> NOT PYRAMID.
            # only half the features are used
            self.corr = nn.Sequential(
                liner_implementation(49*p*p, dim),
                nn.ReLU(inplace=True),
                liner_implementation(dim, dim),
                CustomLayerNorm(dim, eps=1e-3),
                nn.ReLU(inplace=True),
                liner_implementation(dim, dim)
            )
        else:
            # DEFAULT CASE, with matching features downsampled at 1/4 and 1/16 and stacked toghether, with output flattened = 2*49*p*p
            self.corr = nn.Sequential(
                liner_implementation(2*49*p*p, dim),
                nn.ReLU(inplace=True),
                liner_implementation(dim, dim),
                CustomLayerNorm(dim, eps=1e-3),
                nn.ReLU(inplace=True),
                liner_implementation(dim, dim),
        )



        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            liner_implementation(dim, 2))

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            liner_implementation(dim, 2),
            nn.Sigmoid())
        
        self.tc_neighbor_gather_ix  = TCneighborgather(0)
        self.tc_neighbor_gather_jx  = TCneighborgather(1)


    def forward(self, net, kk):
        """ update operator """

        net = self.norm(net) # (b,edges,384)
        
        #apply the temporal convolution to the features
        gathered_ix_net = self.tc_neighbor_gather_ix(net, kk)
        net = net + self.c1(gathered_ix_net)

        gathered_jx_net = self.tc_neighbor_gather_jx(net, kk)
        net = net + self.c2(gathered_jx_net)

        #default configuration
        net = self.agg_kk(net, kk)
        net = self.agg_ij(net, kk)

        net = self.gru(net)
        patch_flow=self.d(net)
        confidence_weigths = self.w(net)

        return net, patch_flow, confidence_weigths



class UpdateOriginal(nn.Module):
    def __init__(self, p, dim=DIM, use_pyramid=True, use_softagg=True, use_gru=True, use_ctx_features=True, use_temp=True):
        super(UpdateOriginal, self).__init__()
        self.dim = dim
        self.use_pyramid = use_pyramid
        self.use_softagg = use_softagg
        self.use_gru = use_gru
        self.use_temp = use_temp
        self.use_ctx_features = use_ctx_features
        
        if self.use_temp:
            self.c1 = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim))

            self.c2 = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim))
        else:
            #skip the temporal convolution in this case
            self.c1 = nn.Identity()
            self.c2 = nn.Identity()

        self.norm = nn.LayerNorm(dim, eps=1e-3)

        if self.use_softagg:
            self.agg_kk = SoftAgg(dim)
            self.agg_ij = SoftAgg(dim)
        else:
            self.agg_kk = nn.Identity()
            self.agg_ij = nn.Identity()

        if use_gru:
            self.gru = nn.Sequential(
                nn.LayerNorm(dim, eps=1e-3),
                GatedResidual(dim),
                nn.LayerNorm(dim, eps=1e-3),
                GatedResidual(dim),
            )
        else:
            self.gru = nn.Sequential(
                nn.LayerNorm(dim, eps=1e-3),
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim, eps=1e-3),
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
            )
            

        if(use_pyramid == False) :
            #SMALLER VERSION : ONLY THE HIGH RESOLUTION MATCHING FEATURES ARE USED (1/4 ON, 1/16 NOT ADDED) --> NOT PYRAMID.
            # only half the features are used
            self.corr = nn.Sequential(
                nn.Linear(49*p*p, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
            )
        else:
            # DEFAULT CASE, with matching features downsampled at 1/4 and 1/16 and stacked toghether, with output flattened = 2*49*p*p
            self.corr = nn.Sequential(
                nn.Linear(2*49*p*p, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
        )



        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        if inp is None:
            inp = torch.zeros_like(net) #to avoid error of context features

        net = net + inp + self.corr(corr) # Entry point for Deeploy
        net = self.norm(net) # (b,edges,384)


        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)
        
        if self.use_temp:
            #apply the temporal convolution to the features
            net = net + self.c1(mask_ix * net[:,ix])
            net = net + self.c2(mask_jx * net[:,jx])

        if self.use_softagg:
            #default configuration
            net = net + self.agg_kk(net, kk)
            net = net + self.agg_ij(net, ii*12345 + jj)



        net = self.gru(net)
        patch_flow=self.d(net)
        confidence_weigths = self.w(net)

        return net, (patch_flow, confidence_weigths, None)



if __name__ == "__main__":
      pass