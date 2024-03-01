# coding: utf-8 

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalNetwork(nn.Module):
    def __init__(self):
        super(GlobalNetwork, self).__init__()
        self.conv_0 = nn.Conv1d(7, 1024, 3)
        self.conv_1 = nn.Conv1d(1024, 512, 3)
        self.conv_2 = nn.Conv1d(512, 128, 3)
        self.conv_3 = nn.Conv1d(128, 64, 2)
        
        self.bn_0 = nn.BatchNorm1d(1024)
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        net = F.relu(self.bn_0(self.conv_0(x)))
        net = F.relu(self.bn_1(self.conv_1(net)))
        net = F.relu(self.bn_2(self.conv_2(net)))
        net = F.relu(self.bn_3(self.conv_3(net)))
        outputs = net.view(-1, 64)
        return outputs
    
    
class MetaNetwork(nn.Module):
    def __init__(self):
        super(MetaNetwork, self).__init__()
        self.conv_0 = nn.Conv1d(7, 1024, 2)
        self.conv_1 = nn.Conv1d(1024, 512, 2)
        self.conv_2 = nn.Conv1d(512, 128, 2)
        self.conv_3 = nn.Conv1d(128, 64, 1)
        
        self.bn_0 = nn.BatchNorm1d(1024)
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        net = F.relu(self.bn_0(self.conv_0(x)))
        net = F.relu(self.bn_1(self.conv_1(net)))
        net = F.relu(self.bn_2(self.conv_2(net)))
        net = F.relu(self.bn_3(self.conv_3(net)))
        outputs = net.view(-1, 64)
        return outputs
    
    
class MetalNetwork(nn.Module):
    def __init__(self):
        super(MetalNetwork, self).__init__()
        self.conv_0 = nn.Conv1d(7, 1024, 1)
        self.conv_1 = nn.Conv1d(1024, 512, 1)
        self.conv_2 = nn.Conv1d(512, 128, 2)
        self.conv_3 = nn.Conv1d(128, 64, 1)
        
        self.bn_0 = nn.BatchNorm1d(1024)
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        net = F.relu(self.bn_0(self.conv_0(x)))
        net = F.relu(self.bn_1(self.conv_1(net)))
        net = F.relu(self.bn_2(self.conv_2(net)))
        net = F.relu(self.bn_3(self.conv_3(net)))
        outputs = net.view(-1, 64)
        return outputs
    
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.meta_feature = MetaNetwork()
        self.metal_feature = MetalNetwork()
        
        self.fc_0 = nn.Linear(192, 1024)
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 1)
        
        self.bn_0 = nn.BatchNorm1d(1024)
        self.bn_1 = nn.BatchNorm1d(512)
        
        
    def forward(self, x):
        meta_0 = self.meta_feature(x[:,:, [0, 1, 2, 6]])
        meta_1 = self.meta_feature(x[:,:, [3, 4, 5, 7]])
        metal = self.metal_feature(x[:,:, [6, 7]])
        
        net = torch.cat([meta_0, meta_1, metal], dim=1)
        net = F.relu(self.bn_0(self.fc_0(net)))
        net = F.relu(self.bn_1(self.fc_1(net)))
        outputs = self.fc_2(net)
        return outputs


if __name__ == "__main__":
    test_tensor = torch.randn([16, 7, 8]).cuda()
    test_net = Model().cuda()
    outputs = test_net(test_tensor)
    print(outputs.shape)