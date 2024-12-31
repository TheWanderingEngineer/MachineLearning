from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class ModelManager():
    def __init__(self,
                model: nn.Module,
               optimizer: torch.optim.Optimizer,
               train_dl: DataLoader,
               test_dl: Optional[DataLoader] = None,
               loss_fn: Optional[nn.Module] = None,
               device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = device
        self.batch_stats = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        self.epoch_stats = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        print(f"ModelManager initialized on device: {self.device}")

    def train_test(self,epochs: int = 5,
                   prints_per_epoch: int = 5,
                   clip_grad: Optional[float] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   target_test_acc: Optional[float] = None):
        print_interval_train = max(1, len(self.train_dl)//(prints_per_epoch))
        #print_interval_test = max(1, len(self.test_dl)//(prints_per_epoch))
        for epoch in range(epochs):
            self.model.train()
            print(f"------------------------------[Epoch {epoch}]------------------------------")
            print(f"Training Phase:")
            for batch, (X,y) in enumerate(self.train_dl):
                X, y = X.to(self.device), y.to(self.device)
                train_logits = self.model(X)
                loss = self.loss_fn(train_logits,y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if batch % print_interval_train == 0:
                    print(f"Batch {batch}/{len(self.train_dl)}: Loss = {loss.item():.4f} | Accuracy {}")

    
        return self.batch_stats, self.epoch_stats
    
    def evaluate(self):
        pass

    def predict(self):
        pass

    def learning_curves(self):
        pass

    def summary(self):
        pass