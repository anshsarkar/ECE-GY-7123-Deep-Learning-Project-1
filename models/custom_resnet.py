import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel=3, shortcut_kernel=1, dropout = 0.0):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=shortcut_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        # Apply first convolution, batch norm, and ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Apply second convolution and batch norm
        out = self.bn2(self.conv2(out))
        # Add shortcut connection
        out += self.shortcut(x)
        # Apply final ReLU
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channels, strides, kernel_size, shortcut_kernel_size, pool_size, dropout, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = channels[0]
        self.kernel_size = kernel_size
        self.shortcut_kernel_size = shortcut_kernel_size
        self.pool_size = pool_size

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])


        self.layers = nn.ModuleList()
        for i in range(len(num_blocks)):
            self.layers.append(
                self._make_layer(block[i], channels[i], num_blocks[i], stride=strides[i], dropout = dropout)
            )

        # Fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(channels[-1] * block[-1].expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout):
        strides = [stride] + [1] * (num_blocks - 1)  # First block uses the specified stride, others use stride=1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.kernel_size, self.shortcut_kernel_size, dropout))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Apply initial convolution, batch norm, and ReLU
        out = F.relu(self.bn1(self.conv1(x)))

        # Pass through all layers
        for layer in self.layers:
            out = layer(out)

        # Apply average pooling and flatten
        out = F.avgpool(out)
        out = torch.flatten(out, 1)

        # Apply fully connected layer
        out = self.dropout(out)
        out = self.linear(out)
        return out
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)