from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from time import time
import numpy as np
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
        self.train_test_time = None
        print(f"ModelManager initialized on device: {self.device}")

    def train_test(self,epochs: int = 5,
                   prints_per_epoch: int = 5,
                   clip_grad: Optional[float] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   target_test_acc: Optional[float] = None):
        start_time = time()
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
                    print(f"Batch {batch}/{len(self.train_dl)}: Loss = {loss.item():.4f} | Accuracy = {acc*100:.4f}%")
            train_loss_epoch = round(train_loss/len(self.train_dl),4)
            train_acc_epoch = round(train_acc/len(self.train_dl),4)
            self.epoch_stats["train_loss"].append(train_loss_epoch)
            self.epoch_stats["train_acc"].append(train_acc_epoch)
            print(f"Training Summary for Epoch {epoch+1}:\nAverage Loss: {train_loss_epoch: }\nAverage Accuracy: {(train_acc_epoch)*100}%")
            if self.test_dl:
                print(f"\nTesting Phase:")
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
                    print(f"Testing Summary for Epoch {epoch+1}:\nAverage Loss: {test_loss_epoch: }\nAverage Accuracy: {(test_acc_epoch)*100}%")
                    self.epoch_stats["test_loss"].append(test_loss_epoch)
                    self.epoch_stats["test_acc"].append(test_acc_epoch)
        end_time = time()
        self.train_test_time = end_time-start_time
        if self.test_dl:
            print(f"Training and Testing completed in {self.train_test_time:.2f} seconds.")
        else:
            print(f"Training completed in {self.train_test_time:.2f} seconds.")
    
    def evaluate(self):
        pass

    def predict(self):
        pass

    def learning_curves(self, acc = False):
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,10))
        fig.suptitle(f"Learning Curves of Model: {self.model}", fontsize=16)
        steps = range(len(self.batch_stats["train_loss"]))
        ax1.plot(steps, self.batch_stats["train_loss"], color = 'blue', label = "Train Loss")
        if len(self.batch_stats["test_loss"]) > 0:
            ax1.plot(steps, self.batch_stats["test_loss"], color = 'green', label = "Test Loss")
        else:
            print("Warning: Test Loss is empty. Skipping test loss curve.")
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Steps (In Batches)")
        ax1.set_ylabel("Loss")
        ax1.legend()
        if acc:
            ax2.plot(steps, self.batch_stats["train_acc"], color = 'blue', label = "Train Accuracy")
            if len(self.batch_stats["test_acc"]) > 0:
                ax2.plot(steps, self.batch_stats["test_acc"], color = 'green', label = "Test Accuracy")
            else:
                print("Warning: Test Accuracy is empty. Skipping test accuracy curve.")
            ax2.set_title("Accuracy Curves")
            ax2.set_xlabel("Steps (In Batches)")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
        plt.tight_layout()
        plt.show()

    def summary(self, hyperparamters = False):
        print("____________________________ Train Summary ____________________________")
        print(f"Final Training Loss: {self.epoch_stats['train_loss'][-1]}")
        print(f"Final Training Accuracy: {(self.epoch_stats['train_acc'][-1]*100):.2f}%")
        if self.train_dl:
            print("____________________________ Test Summary ____________________________")
            print(f"Final Testing Loss: {self.epoch_stats['test_loss'][-1]}")
            print(f"Final Testing Accuracy: {self.epoch_stats['test_acc'][-1]*100}%")

            print(f"Best Testing Accuracy: {max(self.epoch_stats['test_acc'])*100:.2f}% at epoch {np.argmax(self.epoch_stats['test_acc'])+1}")
            print(f"\nTraining & Testing time: {self.train_test_time:.2f} seconds")
        else:
            print(f"\nTraining time: {self.train_test_time:.2f} seconds")
        if hyperparamters:
            print("____________________________ Hyperparamters ____________________________")
            print(f"Optimizer: {type(self.optimizer).__name__}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
            print(f"Loss Function: {type(self.loss_fn).__name__}")
            print(f"Device: {self.device}")
