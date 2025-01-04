from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from time import time
import numpy as np
import random
from torchvision.datasets import ImageFolder
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

        self.classes = train_dl.dataset.classes
        self.transforms = train_dl.dataset.transform

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
    
    def evaluate(self, custom_dl=None):
        data_loader = custom_dl if custom_dl else self.test_dl
        if not data_loader:
            print("No test dataset provided. Skipping evaluation.")
            return None

        self.model.to(self.device)
        self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                test_loss += loss.item()
                acc = (logits.argmax(dim=1) == y).float().mean().item()
                test_acc += acc

        avg_loss = round(test_loss / len(data_loader), 4)
        avg_acc = round(test_acc / len(data_loader), 4)

        print("Evaluation Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Accuracy: {avg_acc * 100:.2f}%")

        return avg_loss, avg_acc




    def predict(self, data: DataLoader = None, n = 4):
        dataloader = data if data is not None else self.test_dl
        self.model.to(self.device)
        fig = plt.figure(figsize=(10,10))
        data_size = len(dataloader.dataset)
        correct = 0
        for i in range(n*n):
            sample = dataloader.dataset[random.randint(0,data_size-1)]
            img, label = sample
            pred = self.model(img.unsqueeze(0).to(self.device)).argmax(dim=1)
            fig.add_subplot(n,n,i+1)
            plt.imshow(img.permute(1,2,0))
            if pred == label:
                plt.title(f"Actual: {self.classes[label]} \n Prediction: {self.classes[pred]}", color = 'green')
                correct+=1
            else:
                plt.title(f"Actual: {self.classes[label]} \n Prediction: {self.classes[pred]}", color = 'red')
            plt.axis('off')
        plt.suptitle(f"{n*n} random images with model predictions", fontsize=18, color="black")
        plt.figtext(0.5, 0.08, f"{correct}/{n*n} Predicited Correctly", fontsize=14, color="blue",ha="center")
        plt.tight_layout()
        plt.show()

    def learning_curves(self, acc = True, step = 'epoch'):
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,10))
        fig.suptitle(f"Learning Curves of Model: {type(self.model).__name__}", fontsize=16)

        if step == 'epoch':
            train_loss = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in self.epoch_stats["train_loss"]]
            test_loss = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in self.epoch_stats["test_loss"]]
            train_acc = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in self.epoch_stats["train_acc"]]
            test_acc = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in self.epoch_stats["test_acc"]]
        elif step == 'batch':
            train_loss = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in self.batch_stats["train_loss"]]
            test_loss = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in self.batch_stats["test_loss"]]
            train_acc = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in self.batch_stats["train_acc"]]
            test_acc = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in self.batch_stats["test_acc"]]
        else:
            raise ValueError("You must set 'step' to either 'epoch' or 'batch'!")

        
        steps = range(len(test_loss))
        ax1.plot(steps, train_loss[:len(test_loss)], color = 'blue', label = "Train Loss")
        if len(test_loss) > 0:
            ax1.plot(steps, test_loss, color = 'green', label = "Test Loss")
        else:
            print("Warning: Test Loss is empty. Skipping test loss curve.")
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Steps (In Batches)")
        ax1.set_ylabel("Loss")
        ax1.legend()
        if acc:
            ax2.plot(steps, train_acc[:len(test_loss)], color = 'blue', label = "Train Accuracy")
            if len(test_acc) > 0:
                ax2.plot(steps, test_acc, color = 'green', label = "Test Accuracy")
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
