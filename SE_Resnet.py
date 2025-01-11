import torch
import torch.nn as nn
from torch.optim import Adam

class SENet(nn.Module):
    def __init__(self, in_channels, scale_factor=16):
        super(SENet, self).__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(self.in_channels, self.in_channels // self.scale_factor, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.conv_up = nn.Conv2d(self.in_channels // self.scale_factor, self.in_channels, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.global_pooling(x)
        x = self.conv_down(x)
        x = self.relu(x)
        x = self.conv_up(x)
        x = self.sigmoid(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, identity_downsample=None, add_se_module=False):
        super(Block, self).__init__()
        self.expansion = 4
        
        self.add_se_module = add_se_module
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
        # SE Module
        if self.add_se_module:
            self.se_module = SENet(out_channels * self.expansion)
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        if self.add_se_module:
            x_se_module = self.se_module(x)
            x = x * x_se_module
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x = x + identity
        x = self.relu(x)
            
        return x
    
class Resnet(nn.Module):
    def __init__(self, Block, layers, image_channels, num_classes):
        super(Resnet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        
        self.layer1 = self._make_layer(Block, out_channels=64, num_blocks_layer=layers[0], stride=1)
        self.layer2 = self._make_layer(Block, out_channels=128, num_blocks_layer=layers[1], stride=2)
        self.layer3 = self._make_layer(Block, out_channels=256, num_blocks_layer=layers[2], stride=2)
        self.layer4 = self._make_layer(Block, out_channels=512, num_blocks_layer=layers[3], stride=2)
        
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512*4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def _make_layer(self, Block, out_channels, num_blocks_layer, stride):
        identity_downsample = None
        layer = []
        
        # Downsample
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                                        nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride, padding=0),
                                        nn.BatchNorm2d(out_channels * 4)
                                    )
            
        layer.append(Block(self.in_channels, out_channels, stride, identity_downsample, add_se_module=True))
        self.in_channels = out_channels*4
        
        for i in range(num_blocks_layer - 1):
            # if i == num_blocks_layer - 2:
            #     layer.append(Block(self.in_channels, out_channels, add_se_module=True))
                
            layer.append(Block(self.in_channels, out_channels, add_se_module=True))
        
        return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.max_pooling(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pooling(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
    
        return x

def SE_Resnet50(num_classes, image_channels=3):
    return Resnet(Block, [3, 4, 6, 3], image_channels=image_channels, num_classes=num_classes)

def SE_Resnet101(num_classes, image_channel=3):
    return Resnet(Block, [3, 4, 23, 3], image_channels=image_channel, num_classes=num_classes)

def SE_Resnet152(num_classes, image_channels=3):
    return Resnet(Block, [3, 8, 36, 3], image_channels=image_channels, num_classes=num_classes)


### Testing model and training

# datapoint = torch.randn(1, 3, 224, 224)
# true_label = torch.randint(low=0, high=10, size=(1,))
# true_label = nn.functional.one_hot(true_label, num_classes=10).to(dtype=torch.float32)
# print(true_label)
# resnet50 = SE_Resnet50(num_classes=10)
# label = resnet50(datapoint)
# print(label)
# print(resnet50)
# print(summary(resnet50, (3, 224, 224)))

# def training_loop(model, input, num_epochs):
#     optim = Adam(model.parameters(), lr=0.001)
#     loss_fn = nn.CrossEntropyLoss()
#     model.train()
    
#     optim.zero_grad()
#     outputs = model(input)
#     print(outputs)
#     loss = loss_fn(outputs, true_label)
#     loss.backward()
#     optim.step()
    
# training_loop(model=resnet50, input=datapoint, num_epochs=1)