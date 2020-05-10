import torch.nn as nn
import torch.nn.functional as F
import torch
import utils
from torch.autograd import Variable
import math

# parts of code referred from https://github.com/penghu-cs/SDML/blob/master/model.py

class Dense_Net(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Dense_Net, self).__init__()
        mid_num = 1024
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)
        self.fc3 = nn.Linear(mid_num, out_dim)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2)
        norm_x = torch.norm(out3, dim=1, keepdim=True)
        out3 = out3 / norm_x
        return [out1, out2, out3]

class Dense_Net_with_softmax(nn.Module):
    def __init__(self, input_dim=28*28, out_dim=20, num_classes=10):
        super(Dense_Net_with_softmax, self).__init__()
        mid_num = 1024
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)
        self.fc3 = nn.Linear(mid_num, out_dim)
        self.fc4 = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2)
        norm_x = torch.norm(out3, dim=1, keepdim=True)
        out3 = out3 / norm_x
        out4 = F.softmax(self.fc4(out3), dim=1)
        return [out1, out2, out3, out4]

class Classification_Net(nn.Module):
    def __init__(self, input_image_feat_dims=512, input_text_feat_dims=512):
        super(Classification_Net, self).__init__()
        self.fc1 = nn.Linear(input_text_feat_dims+input_image_feat_dims, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, img_feats, text_feats):
        concat_vector = torch.cat((img_feats, text_feats), dim=1)
        out1 = self.dropout_5(self.relu(self.fc1(concat_vector)))
        out2 = self.dropout_5(self.relu(self.fc2(out1)))
        out3 = self.dropout_5(self.relu(self.fc3(out2)))
        out4 = self.fc4(out3)
        return out4

class Unimodal_Baseline(nn.Module):
    def __init__(self, input_dims):
        super(Unimodal_Baseline, self).__init__()
        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, feats):
        out = self.relu(self.fc1(feats))
        out = self.fc2(out)
        return out

class Multimodal_Baseline(nn.Module):
    def __init__(self, input_dims1, input_dims2):
        super(Multimodal_Baseline, self).__init__()
        self.fc1 = nn.Linear(input_dims1+input_dims2, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, img_feats, text_feats):
        concat_vector = torch.cat((img_feats, text_feats), dim=1)
        out = self.relu(self.fc1(concat_vector))
        out = self.fc2(out)
        return out
