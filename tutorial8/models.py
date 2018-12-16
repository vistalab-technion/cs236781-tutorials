import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# APPROXIMATE LABEL MATCHER (ALM)
# the generator model transforms target data as though it is sampled from source distribution
class alm_generator(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(alm_generator, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, output_ch, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(.5)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))                   
        x = self.drop(F.leaky_relu(self.bn(self.conv2(x))))
        x = self.drop(self.bn(self.conv2(x)))
        x = self.conv3(x)
        return F.tanh(x)

# the discriminator model tries to distinguish between source and target data
class alm_discriminator(nn.Module):
    def __init__(self, input_ch, hidden_size):
        super(alm_discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn = nn.BatchNorm2d(20)
        self.drop = nn.Dropout2d(.25)
        self.fc1 = nn.Linear(320, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
        x = F.leaky_relu(F.max_pool2d(self.drop(self.bn(self.conv2(x))), 2))
        x = x.view(-1, 320)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)

# the classifier model tries to guess the label of both source and target data
class alm_classifier(nn.Module):
    def __init__(self, input_ch, hidden_size):
        super(alm_classifier, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.bn = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(432, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.drop = nn.Dropout2d(.25)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=1)
        x = F.max_pool2d(self.bn(self.conv2(x)), kernel_size=2, stride=1)
        x = x.view(-1, 432)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = F.leaky_relu(self.drop(self.fc2(x)))
        return F.softmax(self.fc3(x))

# DOMAIN ADAPTATION NEURAL NETWORK (DANN)
# the gradient reverse layer multiplies the gradient by a negative constant
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

# the feature extractor passes a lower-dimensional representation of source/target images to the classifiers
class dann_f_extract(nn.Module):
    def __init__(self, n_ch):
        super(dann_f_extract, self).__init__()
        self.conv1 = nn.Conv2d(n_ch, 32, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        return x

# the domain classifier predicts whether the feature vectors correspond to source or target images.
class dann_domain_clf(nn.Module):
    def __init__(self):
        super(dann_domain_clf, self).__init__()
        self.fc1 = nn.Linear(1200, 100) 
        self.fc2 = nn.Linear(100, 1)
        self.drop = nn.Dropout2d(0.25)

    def set_lambda(self, lambd):
        self.lambd = lambd
        
    def forward(self, x):
        x = grad_reverse(x, self.lambd)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)

# the class classifier predicts the class membership of the feature vectors
class dann_class_clf(nn.Module):
    def __init__(self):
        super(dann_class_clf, self).__init__()
        self.fc1 = nn.Linear(1200, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.drop = nn.Dropout2d(0.2)
        
    def forward(self, x):
        x = F.relu(self.drop(self.fc1(x)))
        x = F.relu(self.drop(self.fc2(x)))
        x = self.fc3(x)
        return F.softmax(x)

# LOADER FUNCTIONS
# instantiates alm model
def ALM(input_ch, output_ch, d_hidden_size, c_hidden_size):
    g = alm_generator(input_ch=input_ch, output_ch=output_ch)
    d = alm_discriminator(input_ch=output_ch, hidden_size=d_hidden_size)
    c = alm_classifier(input_ch=output_ch, hidden_size=c_hidden_size)
    return g, d, c

# instantiates dann model
def DANN(input_ch):
    g = dann_f_extract(input_ch)
    d = dann_domain_clf()
    c = dann_class_clf()
    return g, d, c
