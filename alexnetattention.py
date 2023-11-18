import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.nn.functional.softmax(energy, dim=1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class AlexNet(nn.Module):
    def __init__(self, n_classes=4, **kwargs):
        super(AlexNet, self).__init__()  # Corrected super call
        nb_filters = 8
        self.n_classes = n_classes
        self.conv2d_1 = nn.Conv2d(9, nb_filters, 11, stride=4)
        self.attention1 = SelfAttention(nb_filters)
        self.conv2d_2 = nn.Conv2d(nb_filters, nb_filters * 2, 5, padding=2)
        self.conv2d_3 = nn.Conv2d(nb_filters * 2, nb_filters * 4, 3, padding=1)
        self.attention3 = SelfAttention(nb_filters * 4)
        self.conv2d_4 = nn.Conv2d(nb_filters * 4, nb_filters * 8, 3, padding=1)
        self.conv2d_5 = nn.Conv2d(nb_filters * 8, 256, 3, padding=1)
        self.linear_1 = nn.Linear(9216, 4096)
        self.linear_2 = nn.Linear(4096, 2048)
        self.linear_3 = nn.Linear(2048, n_classes)
        self.maxpool2d = nn.MaxPool2d(3, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0)

    def forward(self, X, **kwargs):
        x1 = self.relu(self.conv2d_1(X))
        x1 = self.attention1(x1)
        maxpool1 = self.maxpool2d(x1)
        maxpool1 = self.dropout(maxpool1)
        x2 = self.relu(self.conv2d_2(maxpool1))
        maxpool2 = self.maxpool2d(x2)
        maxpool2 = self.dropout(maxpool2)
        x3 = self.relu(self.conv2d_3(maxpool2))
        x3 = self.attention3(x3)
        x4 = self.relu(self.conv2d_4(x3))
        x5 = self.relu(self.conv2d_5(x4))
        x6 = self.maxpool2d(x5)
        x6 = self.dropout(x6)
        x6 = x6.view(x6.size(0), -1)
        x7 = self.relu(self.linear_1(x6))
        x8 = self.relu(self.linear_2(x7))
        x9 = self.linear_3(x8)
        return x9




