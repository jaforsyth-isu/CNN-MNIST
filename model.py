import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    #MNIST model with 5 hidden convolutional layers and 2 hidden linear layers
    #Convolutional operation uses 3x3 kernels with a stride of 1 and padding of 2
    #Convolutional layers are similarly composed with depth increasing
    #Linear hidden layers have 2700 neurons each

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(num_features=25),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=75, kernel_size=3, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(num_features=75),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=75, out_channels=150, kernel_size=3, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(num_features=150),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=150, out_channels=300, kernel_size=3, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(num_features=300),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=300, out_channels=600, kernel_size=3, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(num_features=600),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.layer6 = nn.Sequential(
            nn.Linear(in_features=2700, out_features=2700),
            nn.LeakyReLU()
        )

        self.layer7 = nn.Sequential(
            nn.Linear(in_features=2700, out_features=2700),
            nn.LeakyReLU()
        )

        # Fully connected layer to 10 outputs
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=2700, out_features=10),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer6(out.flatten(1))
        out = self.layer7(out)
        out = self.output_layer(out)

        return out