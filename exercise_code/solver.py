import numpy as np
import torch

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        self.train_epoch_acc_history = []
        self.val_epoch_acc_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        #######################################################################
        # TODO:                                                               #
        # Write your own personal training method for our solver. In each     #
        # epoch iter_per_epoch shuffled training batches are processed. The   #
        # loss for *each batch* is stored in self.train_loss_history. Every   #
        # log_nth iteration the loss is logged. After one epoch the training  #
        # *accuracy of the last mini batch* is logged and stored in           #
        # self.train_acc_history. We *validate at the end of each epoch*, log #
        # the result and store *the accuracy of the entire validation set* in #
        # self.val_acc_history.                                               #
        #                                                                     #
        # Your logging could like something like:                             #
        #   ...                                                               #
        #   [Iteration 700/4800] TRAIN loss: 1.452                            #
        #   [Iteration 800/4800] TRAIN loss: 1.409                            #
        #   [Iteration 900/4800] TRAIN loss: 1.374                            #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                           #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                           #
        #   ...                                                               #
        #######################################################################
        for epoch in range(1, num_epochs+1):
            # Train for one epoch
            train_out = None
            train_targets = None
            for iter, (train_inputs, train_targets) in enumerate(train_loader, 1):
                train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
                optim.zero_grad()                                 # Clear gradients
                train_out = model(train_inputs)                   # Get predictions
                loss = self.loss_func(train_out, train_targets)   # Calculate loss
                loss.backward()                                   # Backpropagation
                optim.step()                                      # Optimize parameters based on backpropagation
                self.train_loss_history.append(loss.item())       # Store loss for each batch

                # Log nth iteration
                if iter % log_nth == 0:
                    print("[Iteration {}/{}] TRAIN loss: {}".format(iter, iter_per_epoch, loss.item()))
            
            # Calculate & store train accuracy (of the last batch only)
            _, train_preds = torch.max(train_out, 1) # torch.max returns (max, max_indices)
            targets_mask = train_targets >= 0
            acc = np.mean((train_preds == train_targets)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(acc)

            # Validate after training for one epoch
            model.eval()
            for (val_inputs, val_targets) in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_out = model(val_inputs)
                # Calculate & store loss
                loss = self.loss_func(val_out, val_targets)
                self.val_loss_history.append(loss.item())
                # Calculate & store accuracy
                _, val_preds = torch.max(val_out, 1) # torch.max returns (max, max_indices)
                targets_mask = val_targets >= 0
                acc = np.mean((val_preds == val_targets)[targets_mask].data.cpu().numpy())
                self.val_acc_history.append(acc)
            model.train()

            # Log the results at the end of the epoch
            train_loss = np.mean(self.train_loss_history)
            train_acc = self.train_acc_history[-1]
            val_loss = np.mean(self.val_loss_history)
            val_acc = np.mean(self.val_acc_history)
            print("[Epoch {}/{}] TRAIN acc/loss: {}/{}".format(epoch, num_epochs, train_acc, train_loss))
            print("[Epoch {}/{}] VAL   acc/loss: {}/{}".format(epoch, num_epochs, val_acc, val_loss))
            self.train_epoch_acc_history.append(train_acc)
            self.val_epoch_acc_history.append(val_acc)


        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        print('FINISH.')
