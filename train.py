from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.utils.data import DataLoader

def train_test(model: nn.Module,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               train_dl: DataLoader,
               test_dl: Optional[DataLoader],
               epochs: int = 10,
               device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
               prints_per_epoch: int = 5,
               clip_grad: Optional[float] = None,
               scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
               target_test_acc: Optional[float] = None)-> Tuple[List[float], List[float], List[float], List[float]]:
  """
  Train and optionally test a PyTorch model.

    Args:
      model (torch.nn.Module): The PyTorch model to train.
      loss_fn (Callable): Loss function for training (e.g., nn.CrossEntropyLoss).
      optimizer (torch.optim.Optimizer): Optimizer for model parameters (e.g., Adam, SGD).
      train_dl (torch.utils.data.DataLoader): DataLoader for training data.
      test_dl (Optional[torch.utils.data.DataLoader]): DataLoader for test data (optional).
      epochs (int): Number of epochs to train. Default is 10.
      device (str): Device for training ('cuda' or 'cpu'). Default is 'cuda' if available.
      prints_per_epoch (int): Number of training status prints per epoch. Default is 5.
      clip_grad (Optional[float]): Max norm for gradient clipping. If None, no gradient clipping is applied. Default is None.
      scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler. If None, no scheduler is applied. Default is None.
      target_test_acc (Optional[float]): Target testing accuracy to stop training early. If None, no stopping condition is applied.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
          - train_losses: List of averaged training losses per epoch.
          - train_accs: List of averaged training accuracies per epoch.
          - test_losses: List of averaged test losses per epoch (if test_dl is provided).
          - test_accs: List of averaged test accuracies per epoch (if test_dl is provided).
  """
  model.to(device)
  print(f"Selected Device: {device}")
  train_losses, train_accs, test_losses, test_accs = [], [], [], []
  print_interval_train = max(1, len(train_dl)//prints_per_epoch)
  for epoch in range(epochs):
    print(f"\n---------------------------- Epoch {epoch+1}/{epochs} ----------------------------")
    print("\nTraining Phase:")
    model.train()
    train_loss, train_acc = 0,0
    for batch, (X, y) in enumerate(train_dl):
      X, y = X.to(device), y.to(device)
      y_logits = model(X)
      loss = loss_fn(y_logits, y)
      train_loss+=loss.item()
      optimizer.zero_grad()
      loss.backward()
      if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
      optimizer.step()
      if scheduler:
        scheduler.step()
      acc = (y_logits.argmax(dim=1) == y).float().mean()
      train_acc+=acc.item()

      if batch % print_interval_train == 0:
        print(f"Batch {batch}/{len(train_dl)}: Loss = {loss.item():.4f} | Accuracy = {acc.item()}")
    train_losses.append(round(train_loss/len(train_dl),4))
    train_accs.append(round(train_acc/len(train_dl),3))
    print(f"Average Train Accuracy in this epoch: {train_acc/len(train_dl):.4f}")
    if test_dl:
      print("\nTesting Phase:")
      print_interval_test = max(1, len(test_dl) // prints_per_epoch)
      model.eval()
      with torch.inference_mode():
        test_loss, test_acc = 0,0
        for batch, (X, y) in enumerate(test_dl):
          X, y = X.to(device), y.to(device)
          y_logits_test = model(X)
          loss = loss_fn(y_logits_test, y)
          test_loss+=loss.item()
          acc = (y_logits_test.argmax(dim=1) == y).float().mean()
          test_acc+=acc.item()
          if batch % print_interval_test == 0:
            print(f"Batch {batch}/{len(test_dl)}: Loss = {loss.item():.4f} | Accuracy = {acc.item()}")
        test_losses.append(round(test_loss/len(test_dl),4))
        test_accs.append(round(test_acc/len(test_dl),3))
        print(f"Average Test Accuracy in this epoch: {test_acc/len(test_dl):.4f}")
        if target_test_acc and test_acc/len(test_dl) >= target_test_acc:
          print(f"\nEarly Stopping: Average Test accuracy {test_acc/len(test_dl):.4f} reached/exceeded the target {target_test_acc:.4f}")
          return train_losses, train_accs, test_losses, test_accs
  return train_losses,train_accs,test_losses,test_accs #Values are per epoch (averaged)
      
