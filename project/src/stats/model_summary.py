from torchinfo import summary
import torch
from torch import nn
from torch import optim
import numpy as np
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from tsai.models.XceptionTimePlus import XceptionTimePlus
from src.settings import cp_xceptiontime_v5

checkpoint = cp_xceptiontime_v5
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_estimator(data_shape, class_weights, device):
    net = NeuralNetClassifier(module=XceptionTimePlus, module__c_in=data_shape, module__c_out=2,
                              module__nf=12, module__act=nn.LeakyReLU,
                              max_epochs=100,
                              iterator_train__shuffle=True,
                              criterion=nn.CrossEntropyLoss, criterion__weight=class_weights, optimizer=optim.AdamW,
                              train_split=ValidSplit(cv=0.1, stratified=True, random_state=42),
                              batch_size=1, lr=0.001,
                              device=device)
    return net


# Get model
estimator = get_estimator(20, class_weights=None, device=device)
estimator.initialize()
estimator.load_params(f_params=f"{checkpoint}train_end_params.pt")
_ = estimator.predict(np.zeros([1, 20, 9000], dtype="float32"))
model = estimator.module_
model.to(device)

summary(model, input_size=(32, 20, 9000), device=device)