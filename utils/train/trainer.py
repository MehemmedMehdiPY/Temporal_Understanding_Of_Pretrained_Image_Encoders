import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple
from tqdm import tqdm

class Trainer:
    def __init__(self, model: torch.nn.Module, train_loader, val_loader,
                 optimizer: torch.optim.Adam, loss_fn: torch.nn.modules.loss.MSELoss, 
                 epochs: int, filepath: str, device: str = None):
        """The class to support training process
        Args:
            model:                    Model to train
            train_loader:             Data Loader for training set
            val_loader:               Data Loader for validation set
            optimizer:                Optimizer
            loss_fn:                  Loss function
            epochs:                   The number of epochs for training
            filepath:                 Filepath to save the training model
            device:                   Device to use trainer object
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_len = len(train_loader)
        self.val_len = len(val_loader)
        
        self.epochs = epochs
        self.filepath = filepath

        self.device = (device if device is not None
                       else "cpu")

        if not os.path.exists('./best_result'):
            os.mkdir('./best_result')

    def train_model(self, epoch: int) -> float:
        """The function to train the model

            Args:
                epoch            int

            Returns:
                train_loss       float
        """

        self.model.train()
        train_loss = 0
        with tqdm(self.train_loader) as loop:
            # Description of current epoch
            loop.set_description('Epoch {}/{}'.format(epoch + 1, self.epochs))

            for X, Y in loop:
                X = X.to(self.device)
                Y = Y.to(self.device)

                self.optimizer.zero_grad()

                out = self.model(X)
                loss = self.loss_fn(out, Y)

                loss_item = loss.item()
                train_loss += loss_item

                loop.set_postfix_str('Loss: ' + str(round(loss_item, 3)))
                loop.update(1)

                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / self.train_len
        return train_loss

    def evaluate_model(self) -> Tuple[float]:
        """The function to evaluate the model performance.

            Returns:
                  test_loss              float
        """

        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            for X, Y in self.val_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)

                out = self.model(X)

                loss = self.loss_fn(out, Y)
                test_loss += loss.item()

        test_loss = test_loss / self.val_len
        return test_loss

    def run(self, epoch_start: Optional[int] = 0) -> None:
        """The function to control training"""
        writer = SummaryWriter()

        best_test_loss = 1e16
        for epoch in range(epoch_start, self.epochs):
            print('-' * 50)

            train_loss = self.train_model(epoch)
            test_loss = self.evaluate_model()

            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Test", test_loss, epoch + 1)

            if test_loss <= best_test_loss:
                # Saving the best model
                best_test_loss = test_loss
                print('The best model is saved at {:.3f}'.format(best_test_loss))
                with open('best_result/best_result.txt', 'w') as fo:
                    fo.write("f1: {}".format(best_test_loss))
                self.save_model()
                
        writer.close()

    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        checkpoints = self.model.state_dict()
        torch.save(checkpoints, filepath)