import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo

class Resnet18(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet18, self).__init__(imagemodels.resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet18']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class Resnet34(imagemodels.ResNet):
    def __init__(self, n_class=10, pretrained=False):
        super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
        if pretrained:
          self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet34']))
        
          for child in self.conv1.children():
            for p in child.parameters():
              p.requires_grad = False

          for child in self.layer1.children():
            for p in child.parameters():
              p.requires_grad = False

          for child in self.layer2.children():
            for p in child.parameters():
              p.requires_grad = False

          for child in self.layer3.children():
            for p in child.parameters():
              p.requires_grad = False

          for child in self.layer4.children():
            for p in child.parameters():
              p.requires_grad = False
          
          for child in list(self.avgpool.children()):
            for p in child.parameters():
              p.requires_grad = False

        self.fc = nn.Linear(512, n_class)
        #self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x, save_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        embed = x.view(x.size()[0], -1)
        out = self.fc(embed)
        if save_features:
          return embed, out
        else:
          return out

class Resnet50(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet50']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class VGG16(nn.Module):
    def __init__(self, n_class=10, pretrained=False):
        super(VGG16, self).__init__()
        '''
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        last_layer_index = len(list(seed_model.children()))
        seed_model.add_module(str(last_layer_index),
            nn.Conv2d(512, embedding_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.image_model = seed_model
        '''
        self.features = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        if pretrained: 
          for child in self.features.children():
            for p in child.parameters():
              p.requires_grad = False

        classifier = imagemodels.__dict__['vgg16'](pretrained=pretrained).classifier
        classifier = nn.Sequential(*list(classifier.children())[:-2])
        for child in classifier.children():
          for p in child.parameters():
            p.requires_grad = False

        penult_layer_index = len(list(classifier.children()))
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        classifier.add_module(str(penult_layer_index), nn.Linear(4096, 512))
        classifier.add_module(str(penult_layer_index + 1), nn.ReLU(True))
        classifier.add_module(str(penult_layer_index + 2), nn.Dropout())
        classifier.add_module(str(penult_layer_index + 3), nn.Linear(512, n_class))
        classifier.add_module(str(penult_layer_index + 4), nn.Dropout())
        self.classifier = classifier

    def forward(self, x, save_features=False):
        x = self.features(x)
        #x = self.avgpool(x) 
        x = x.view(x.size(0), -1)        
        #print(x.size())
        if save_features:
          # VGG16 penultimate layer 
          embedder1 = nn.Sequential(*list(self.classifier.children())[:-6])
          # Compressed 512-dim hidden layer
          embedder2 = nn.Sequential(*list(self.classifier.children())[-6:-3])
          embed1 = embedder1(x)
          embed2 = embedder2(embed1)
          output = self.classifier(x)
          return embed1, embed2, output
        else:
          x = self.classifier(x)
        return x
