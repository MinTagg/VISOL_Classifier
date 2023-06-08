import torch
# make resnet 18 base model that classify 34 class

class MODEL(torch.nn.Module):
    def __init__(self, freeze=True, network = 'resnet50'):
        super(MODEL, self).__init__()
        # load pretrained resnet50 base model. freeze the backbone
        self.resnet18 = torch.hub.load('pytorch/vision:v0.6.0', network, pretrained=True)
        # freeze the backbone
        if freeze:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        # change last layer to classify 34 class
        if network == 'resnet50':
            self.resnet18.fc = torch.nn.Linear(2048, 34)
        elif network == 'resnet18':
            self.resnet18.fc = torch.nn.Linear(512, 34)
        #self.resnet18.fc = torch.nn.Linear(2048, 34)
        # activation function
        #self.softmax = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.softmax(x)
        return x