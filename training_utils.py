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
        self.model.to(self.device)
        print_interval_train = max(1, len(self.train_dl)//(prints_per_epoch))
        
        for epoch in range(epochs):
            self.model.train()
            print(f"------------------------------[Epoch {epoch+1}]------------------------------")
            print(f"Training Phase:")
            train_loss, train_acc = 0,0
            for batch, (X,y) in enumerate(self.train_dl):
                X, y = X.to(self.device), y.to(self.device)
                train_logits = self.model(X)
                loss = self.loss_fn(train_logits,y)
                train_loss+=round(loss.item(),4)
                self.batch_stats['train_loss'].append(loss.item())
                acc = round((train_logits.argmax(dim=1) == y).float().mean().item(),4)
                train_acc+=acc
                self.batch_stats['train_acc'].append(acc)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch % print_interval_train == 0:
                    print(f"Batch {batch}/{len(self.train_dl)}: Loss = {loss.item():.4f} | Accuracy = {acc*100}%")
            train_loss_epoch = round(train_loss/len(self.train_dl),4)
            train_acc_epoch = round(train_acc/len(self.train_dl),4)
            self.epoch_stats["train_loss"].append(train_loss_epoch)
            self.epoch_stats["train_acc"].append(train_acc_epoch)
            print(f"Training Epoch {epoch+1} stats:\nAverage Loss: {train_loss_epoch: }\nAverage Accuracy: {(train_acc_epoch)*100}%")
            if self.test_dl:
                test_loss,test_acc = 0,0
                print_interval_test = max(1, len(self.test_dl)//(prints_per_epoch))
                self.model.eval()
                with torch.inference_mode():
                    for batch, (X,y) in enumerate(self.test_dl):
                        X, y = X.to(self.device), y.to(self.device)
                        test_logits = self.model(X)
                        loss = self.loss_fn(test_logits,y)
                        test_loss+=round(loss.item(),4)
                        self.batch_stats["test_loss"].append(loss)
                        acc = round((test_logits.argmax(dim=1) == y).float().mean().item(),4)
                        test_acc+= acc
                        self.batch_stats["test_acc"].append(acc)
                        if batch % print_interval_test == 0:
                            print(f"Batch {batch}/{len(self.test_dl)}: Loss = {loss.item():.4f} | Accuracy = {acc*100}%")
                    test_loss_epoch = round(test_loss/len(self.test_dl),4)
                    test_acc_epoch = round(test_acc/len(self.test_dl),4)
                    print(f"Testing Epoch {epoch+1} stats:\nAverage Loss: {test_loss_epoch: }\nAverage Accuracy: {(test_acc_epoch)*100}%")
                    self.epoch_stats["test_loss"].append(test_loss_epoch)
                    self.epoch_stats["test_acc"].append(test_acc_epoch)

        return self.batch_stats, self.epoch_stats
    
    def evaluate(self):
        pass

    def predict(self):
        pass

    def learning_curves(self):
        pass

    def summary(self):
        pass