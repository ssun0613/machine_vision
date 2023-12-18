import torch.nn as nn

class convBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(convBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residual_block, self).__init__()
        self.residual_function = nn.Sequential(
            convBlock(in_channel=in_channels, out_channel=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_connection = nn.Sequential()

        # Projection mapping
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.residual_function(x) + self.skip_connection(x)
        return self.relu(x)

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()

        block = residual_block
        num_block = [2, 2, 2, 2]

        self.in_channels = 64
        self.conv1 = convBlock(3, self.in_channels, 3, 1, 0)
        self.conv2 = convBlock(self.in_channels, self.in_channels, 3, 1, 1)

        self.conv3_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv4_x = self._make_layer(block, 128, num_block[1], 1)
        self.conv5_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv6_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 8)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.conv6_x(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def predict(self, out):
        sig_out = self.sigmoid(out) # 각각의 값들을 0~1사이의 값으로 변환하는 것이 sigmoid 함수  --> sigmoid를 통과 시키는 이유는 확률화 하기 위해서임.
        sig_out[sig_out > 0.5] = 1 # 변환한 확률 값이 특정 값보다 크면 1로 바꾸고, 특정 값보다 작거나 같으면 0으로 바뀜.
        sig_out[sig_out <= 0.5] = 0
        return sig_out # 예측한 값이 0 혹은 1이 나옴.

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
