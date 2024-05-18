from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import A_net_init
from model import t_net_init
from model import A_net
from model import t_net
from model import J_net

def make_model(args, parent=False):
    return Dehaze_net(args)

class A_net(nn.Module):
    def __init__(self, args):
        super(A_net, self).__init__()

        self.args = args
        m_unet = [unet.make_model(1, 1, ngf = args.n_feats)]
        self.unet = nn.Sequential(*m_unet)

    def forward(self, I_A_t_J):
        I, A, t, J = I_A_t_J[:,0:3,:,:], I_A_t_J[:,3:4,:,:], I_A_t_J[:,4:5,:,:], I_A_t_J[:,5:8,:,:]
        delta_f = (J * t + A * (1.0 - t) - I) * (1 - t)
        delta_g = ((I - A)/torch.clamp(t, self.args.t_clamp) + A - J) * (1-1/torch.clamp(t, self.args.t_clamp))
        A_prior = self.unet(A)
        A = A - self.args.weight_A_prior * (torch.mean(delta_f, 1, keepdim=True) + torch.mean(delta_g, 1, keepdim=True) + A_prior)
        return A

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
                
class J_net(nn.Module):
    def __init__(self, args):
        super(J_net, self).__init__()

        self.args = args
        m_unet = [unet.make_model(3, 3, ngf = args.n_feats)]
        self.unet = nn.Sequential(*m_unet)

    def forward(self, I_A_t_J):
        I, A, t, J = I_A_t_J[:,0:3,:,:], I_A_t_J[:,3:4,:,:], I_A_t_J[:,4:5,:,:], I_A_t_J[:,5:8,:,:] 
        delta_f = (J * t + A * (1.0 - t) - I) * torch.clamp(t, self.args.t_clamp)
        delta_g = ((I - A)/torch.clamp(t, self.args.t_clamp) + A - J) * (-1)
        J_prior = self.unet(J)
        J = J - self.args.weight_J_prior * (delta_f + delta_g + J_prior)
        return J

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

class t_net(nn.Module):
    def __init__(self, args):
        super(t_net, self).__init__()

        self.args = args
        m_unet = [unet.make_model(1, 1, ngf = args.n_feats)]
        self.unet = nn.Sequential(*m_unet)

    def forward(self, I_A_t_J):
        I, A, t, J = I_A_t_J[:,0:3,:,:], I_A_t_J[:,3:4,:,:], I_A_t_J[:,4:5,:,:], I_A_t_J[:,5:8,:,:]
        delta_f = (J * t + A * (1.0 - t) - I) * (J - A)
        delta_g = ((I - A)/torch.clamp(t, self.args.t_clamp) + A - J) * (-(I - A)/torch.pow(torch.clamp(t, self.args.t_clamp),2))
        t_prior = self.unet(t)
        t = t - self.args.weight_t_prior * (torch.mean(delta_f, 1, keepdim=True) + torch.mean(delta_g, 1, keepdim=True) + t_prior)
        return t

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

class Dehaze_net(nn.Module):
    def __init__(self, args):
        super(Dehaze_net, self).__init__()

        self.args = args

        A_net_iter1 = [A_net_init.make_model(self.args)]
        t_net_iter1 = [t_net_init.make_model(self.args)]

        self.args.weight_A_prior = self.args.weight_A_prior_phase1
        self.args.weight_t_prior = self.args.weight_t_prior_phase1
        self.args.weight_J_prior = self.args.weight_J_prior_phase1
        A_net_iter2 = [A_net.make_model(self.args)]
        t_net_iter2 = [t_net.make_model(self.args)]
        J_net_iter2 = [J_net.make_model(self.args)]

        self.args.weight_A_prior = self.args.weight_A_prior_phase2
        self.args.weight_t_prior = self.args.weight_t_prior_phase2
        self.args.weight_J_prior = self.args.weight_J_prior_phase2
        A_net_iter3 = [A_net.make_model(self.args)]
        t_net_iter3 = [t_net.make_model(self.args)]
        J_net_iter3 = [J_net.make_model(self.args)]

        A_net_iter4 = [A_net.make_model(self.args)]
        t_net_iter4 = [t_net.make_model(self.args)]
        J_net_iter4 = [J_net.make_model(self.args)]

        self.A_net_iter1 = nn.Sequential(*A_net_iter1)
        self.t_net_iter1 = nn.Sequential(*t_net_iter1)
        self.A_net_iter2 = nn.Sequential(*A_net_iter2)
        self.t_net_iter2 = nn.Sequential(*t_net_iter2)
        self.J_net_iter2 = nn.Sequential(*J_net_iter2)
        self.A_net_iter3 = nn.Sequential(*A_net_iter3)
        self.t_net_iter3 = nn.Sequential(*t_net_iter3)
        self.J_net_iter3 = nn.Sequential(*J_net_iter3)
        self.A_net_iter4 = nn.Sequential(*A_net_iter4)
        self.t_net_iter4 = nn.Sequential(*t_net_iter4)
        self.J_net_iter4 = nn.Sequential(*J_net_iter4)

    def forward(self, I):

        A_iter1 = self.A_net_iter1(I)
        t_iter1 = self.t_net_iter1(I)
        J_iter1 = (I - A_iter1)/torch.clamp(t_iter1, self.args.t_clamp) + A_iter1

        I_A_t_J = torch.cat((I, A_iter1.detach().clone(), t_iter1.detach().clone(), J_iter1.detach().clone()), dim=1)
        A_iter2 = self.A_net_iter2(I_A_t_J)
        t_iter2 = self.t_net_iter2(I_A_t_J)
        J_iter2 = (I - A_iter2)/torch.clamp(t_iter2, self.args.t_clamp) + A_iter2
        I_A_t_J = torch.cat((I, A_iter2, t_iter2, J_iter2), dim=1)
        J_iter2 = self.J_net_iter2(I_A_t_J)

        I_A_t_J = torch.cat((I, A_iter2.detach().clone(), t_iter2.detach().clone(), J_iter2.detach().clone()), dim=1)
        A_iter3 = self.A_net_iter3(I_A_t_J)
        t_iter3 = self.t_net_iter3(I_A_t_J)
        J_iter3 = (I - A_iter3)/torch.clamp(t_iter3, self.args.t_clamp) + A_iter3
        I_A_t_J = torch.cat((I, A_iter3, t_iter3, J_iter3), dim=1)
        J_iter3 = self.J_net_iter3(I_A_t_J)

        I_A_t_J = torch.cat((I, A_iter3.detach().clone(), t_iter3.detach().clone(), J_iter3.detach().clone()), dim=1)
        A_iter4 = self.A_net_iter4(I_A_t_J)
        t_iter4 = self.t_net_iter4(I_A_t_J)
        J_iter4 = (I - A_iter4)/torch.clamp(t_iter4, self.args.t_clamp) + A_iter4
        I_A_t_J = torch.cat((I, A_iter4, t_iter4, J_iter4), dim=1)
        J_iter4 = self.J_net_iter4(I_A_t_J)

        return A_iter1, t_iter1, J_iter1, A_iter2, t_iter2, J_iter2, A_iter3, t_iter3, J_iter3, A_iter4, t_iter4, J_iter4

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

