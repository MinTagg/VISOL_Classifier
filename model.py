import torch
from torchvision import models
# make resnet 18 base model that classify 34 class

class MODEL(torch.nn.Module):
    def __init__(self, freeze=True, network = 'resnet50', one_hot = True):
        super(MODEL, self).__init__()
        # load pretrained resnet50 base model. freeze the backbone
        if network == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif network == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        elif network == 'resnet101':
            self.resnet = models.resnet101(pretrained=True)
        elif network == 'resnet152':
            self.resnet = models.resnet152(pretrained=True)
        else:
            print('network error')
            exit()
        # freeze the backbone
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        target = 34
        if one_hot:
            target = 35
        # change last layer to classify 34 class
        if network == 'resnet18':
            self.resnet.fc = torch.nn.Linear(512, target)
        else:
            self.resnet.fc = torch.nn.Linear(2048, target)
        
        #self.resnet18.fc = torch.nn.Linear(2048, 34)
        # activation function
        #self.softmax = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.softmax(x)
        return x