import torch.nn as nn

class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.alexnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(96),
            #
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            #
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            #
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            #
            nn.AdaptiveMaxPool2d(3),
            nn.Flatten(),
            #
            nn.Dropout(0.5),
            #
            nn.Linear(in_features=384*3*3, out_features=1024),
            nn.ReLU(),
            #
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            #
            nn.Linear(in_features=256, out_features=8)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.alexnet(x)

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

