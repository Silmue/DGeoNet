import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn import AvgPool2d
from ImagePyramid import ImagePyramidLayer
from BilinearSampling import grid_bilinear_sampling
from MatInverse import inv
import os

import numpy as np


class Twist2Mat(nn.Module):
    def __init__(self):
        super(Twist2Mat, self).__init__()
        self.register_buffer('o', torch.zeros(1,1).float())
        self.register_buffer('E', torch.eye(3).float())

    def cprodmat_batch(self, a_batch):
        batch_size, _ = a_batch.size()
        o = Variable(self.o).expand(batch_size, 1)
        a0 = a_batch[:, 0:1]
        a1 = a_batch[:, 1:2]
        a2 = a_batch[:, 2:3]
        return torch.cat((o, -a2, a1, a2, o, -a0, -a1, a0, o), 1).view(batch_size, 3, 3)

    def forward(self, twist_batch):
        batch_size, _ = twist_batch.size()
        rot_angle = twist_batch.norm(p=2, dim=1).view(batch_size, 1).clamp(min=1e-5)
        rot_axis = twist_batch / rot_angle.expand(batch_size, 3)
        A = self.cprodmat_batch(rot_axis)
        return Variable(self.E).view(1, 3, 3).expand(batch_size, 3, 3)\
            + A*rot_angle.sin().view(batch_size, 1, 1).expand(batch_size, 3, 3)\
            + A.bmm(A)*((1-rot_angle.cos()).view(batch_size, 1, 1).expand(batch_size, 3, 3))

class Mat2Twist(nn.Module):
    def __init__(self):
        super(Mat2Twist, self).__init__()

    def mattoaxis(self, mat_batch, rot_batch):
        batch_size, _, _  = mat_batch.size()
        s_theta = (rot_batch.sin() * 2).view(batch_size, 1)
        w1 = (mat_batch[:, 2, 1] - mat_batch[:, 1, 2]).view(batch_size, 1)
        w2 = (mat_batch[:, 0, 2] - mat_batch[:, 2, 0]).view(batch_size, 1)
        w3 = (mat_batch[:, 1, 0] - mat_batch[:, 0, 1]).view(batch_size, 1)
        return torch.cat((w1, w2, w3), 1)/s_theta

    def forward(self, mat_batch):
        batch_size, _, _ = mat_batch.size()
        rot_angle = torch.cat(tuple([((mat_batch[_].trace()-1)/2).acos().view(1,1,1) for _ in range(batch_size)])).view(batch_size, 1)
        rot_axis = self.mattoaxis(mat_batch, rot_angle)
        return (rot_angle*rot_axis).view(batch_size, 3)

if __name__=='__main__':
    a = torch.from_numpy(np.random.rand(9)).view(3, 3).float()
    twist2mat = Twist2Mat()
    mat2twist = Mat2Twist()
    print(a)
    b = twist2mat(a)
    print(b)
    c = mat2twist(b)
    print(c)
    print(a-c)