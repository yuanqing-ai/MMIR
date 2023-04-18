import os
import torch
import torchvision
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import transforms
from torch.multiprocessing import Process
import numpy as np


# T1 = torch.tensor([[1, 2, 3],
#         		[4, 5, 6],
#         		[7, 8, 9]]).float()
# # 假设是时间步T2的输出
# T2 = torch.tensor([[10, 20, 30],
#         		[40, 50, 60],
#         		[70, 80, 90]]).float()
# t = torch.stack((T1,T2))
# A = torch.mean(t,dim=0)
# print(t,A)
# half = 5
# shuffle_index=torch.randperm(10)
# print("shuffle_index",shuffle_index)
# sychron_s =np.array([0]*half+[1]*half)   
# sychron_s = sychron_s[shuffle_index]     
# print("sychron_s",sychron_s)               
# sychron_s = torch.from_numpy(sychron_s).view(-1,1)
# train_id = torch.nonzero(sychron_s.squeeze()==0).squeeze()
# print("train_id",train_id)
b = torch.randn((3,5))
a = torch.nn.init.trunc_normal_(b, mean=0.0, std=0.1, a=- 2.0, b=2.0)
print(a)