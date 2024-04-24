import torch.nn as nn



class EEGNet(nn.Module):
    def __init__(self, n_classes=30, channels=64, samples=128*5,
                 dropoutRate=0.5, kernelLength=64, kernelLength2=16,
                 F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.drop_out = dropoutRate

        block1 = nn.Sequential(
            # input shape (1, C, T)
            nn.Conv2d(
                in_channels=1,
                out_channels=self.F1, # F1
                kernel_size=(1, self.kernelLength), # (1, half the sampling rate)
                stride=1,
                padding=(0, self.kernelLength//2),
                bias=False
            ), # output shape (F1, C, T)
            nn.BatchNorm2d(num_features=self.F1)
            # output shape (F1, C, T)
        )


        block2 = nn.Sequential(
            # input shape (F1, C, T)
            # Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
            #                      groups=self.F1, bias=False),
            nn.Conv2d(
                in_channels=self.F1,
                out_channels=self.F1*self.D, # D*F1
                kernel_size=(self.channels, 1), # (C, 1)
                groups=self.F1, # When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a “depthwise convolution”.
                bias=False
            ), # output shape (self.F1*self.D, 1, T)
            nn.BatchNorm2d(num_features=self.F1*self.D),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 4),
                stride=4,), # output shape (self.F1*self.D, 1, T/4)
            nn.Dropout(p=self.drop_out)
            # output shape (self.F1*self.D, 1, T/4)
        )

        block3 = nn.Sequential(
            # nn.ZeroPad2d((7, 8, 0, 0)),
            # input shape (self.F1*self.D, 1, T/4)
            nn.Conv2d(
                in_channels=self.F2,
                out_channels=self.F2, # F2 = D*F1
                kernel_size=(1, self.kernelLength2),
                stride=1,
                padding=(0, self.kernelLength2//2),
                groups=self.F1*self.D,
                bias=False
            ), # output shape (self.F2, 1, T/4)
            # input shape (self.F2, 1, T/4)
            nn.Conv2d(
                in_channels=self.F1*self.D,
                out_channels=self.F2, # F2 = D*F1
                kernel_size=(1, 1),
                stride=1,
                bias=False
            ), # output shape (self.F2, 1, T/4)
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=8),  # output shape (self.F2, 1, T/4/8)
            nn.Dropout(p=self.drop_out)
            # output shape (self.F2, 1, T/32)
        )

        self.EEGNetLayer = nn.Sequential(block1, block2, block3)
        self.ClassifierBlock = nn.Sequential(nn.Linear(in_features=self.F2*round(self.samples//32),
                                                       out_features=self.n_classes,
                                                       bias=False),
                                             nn.Softmax(dim=1))

    def forward(self, x):
        x = self.EEGNetLayer(x)
        x = x.view(x.size()[0], -1)
        x = self.ClassifierBlock(x)
        return x



