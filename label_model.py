import torch

class LabelModel(torch.nn.Module):

    def __init__(self):
        super(LabelModel, self).__init__()

        self.linear1 = torch.nn.Linear(2048, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 200)
        self.linear3 = torch.nn.Linear(200, 3)
        # self.sigmoid = torch.nn.sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)

        # x = self.sigmoid(x)
        return x
