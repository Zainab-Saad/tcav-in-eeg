from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from torch import nn
from torch import optim
from tsai.models.XceptionTimePlus import XceptionTimePlus


def xceptiontime_v5(device="cpu", c_in=20, class_weights=None):
    net = NeuralNetClassifier(module=XceptionTimePlus, module__c_in=c_in, module__c_out=2,
                              module__nf=12, module__act=nn.LeakyReLU,
                              max_epochs=100,
                              iterator_train__shuffle=True,
                              criterion=nn.CrossEntropyLoss, criterion__weight=class_weights, optimizer=optim.AdamW,
                              train_split=ValidSplit(cv=0.1, stratified=True, random_state=42),
                              batch_size=1, lr=0.001,
                              device=device)
    return net
