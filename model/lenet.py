import torch.nn as nn

class lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Tanh(),
            #
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Tanh(),
            #
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.Tanh(),
            #
            nn.Flatten(),
            nn.Linear(in_features=4320, out_features=2160),
            nn.Tanh(),
            #
            nn.Linear(in_features=2160, out_features=8))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.lenet(x)

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
