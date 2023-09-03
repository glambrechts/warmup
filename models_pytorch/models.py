import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from .cells import LSTM, CHRONO, GRU, MGU, BRC, nBRC, Double


EPSILON_INVERSE = 1e-6


cell_class = {'lstm': LSTM, 'gru': GRU, 'mgu': MGU, 'brc': BRC, 'nbrc': nBRC,
              'chrono': CHRONO}


def _sample_inputs(sample, number, device):
    """
    Sample `number` sequence from a batch/list of sequence and move them to
    `device`.
    """

    num_samples = len(sample) if isinstance(sample, list) else sample.size(1)
    indices = torch.randint(0, num_samples, (number,))

    if isinstance(sample, list):
        return [torch.from_numpy(sample[i]).to(device) for i in indices]
    else:
        return sample[:, indices, :].to(device)


def _vaa(hn, epsilon):
    """
    Returns the truncated VAA as described in the paper, for the set of hidden
    states `hn` and a tolerance of `epsilon`.
    """
    n = hn.size(0)
    vaa = 0.0

    for i in range(n):
        dist_i = torch.norm(hn[:, :] - hn[i, :], dim=1)
        vaa = vaa + 1.0 / (dist_i < epsilon).sum() * (1.0 / n)

    return vaa


def _vaa_star(hn, epsilon):
    """
    Returns the truncated VAA* as described in the paper, for the set of hidden
    states `hn` and a tolerance of `epsilon`.
    """
    n = hn.size(0)
    vaa = torch.tensor(0.0, device=hn.device)
    tanh = torch.tanh(hn)

    for i in range(n):
        dist_i = torch.norm(tanh[:, :] - tanh[i, :], dim=1)
        Ci = 1.0 - F.relu(dist_i - epsilon) / (dist_i + EPSILON_INVERSE)
        vaa = vaa + (1.0 / Ci.sum()) * (1.0 / n)

    return vaa


class Model(nn.Module):
    """
    This class implements an RNN that can be warmed up and trained on a dataset
    of sequences.

     - cell: str
        Underlying cell architecture (gru, lstm, chrono, brc, nbrc, mgu)
     - input_size: int
        The input size
     - hidden_size: int
        The RNN hidden size
     - num_layers: int
        The number of stacked RNNs
     - output_size: int
        The output_size
     - regression: bool
        Problem type (True for MSE loss, False for CE loss)
     - batch_first: bool
        Time axis (True for 0, False for 1)
     - kwargs: dict
        Additional arguments for the underlying cell architecture
    """

    def __init__(self, cell, input_size, hidden_size, num_layers, output_size,
                 regression=True, double=False, **kwargs):
        """
        See class documentation.
        """
        super().__init__()

        if torch.cuda.is_available():
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        if regression:
            self.loss = self._regression_loss
        else:
            self.loss = self._classification_loss

        Cell = cell_class[cell]
        self.num_states = 2 if cell in ('lstm', 'chrono') else 1

        # Initialize recurrent layers
        if double:
            cells = [Double(Cell, input_size, hidden_size, **kwargs)]
            for _ in range(num_layers - 1):
                cells.append(Double(Cell, hidden_size, hidden_size, **kwargs))
        else:
            cells = [Cell(input_size, hidden_size, **kwargs)]
            for _ in range(num_layers - 1):
                cells.append(Cell(hidden_size, hidden_size, **kwargs))
        self.cells = nn.ModuleList(cells)

        # Initialize linear layer
        self.linear = nn.Linear(hidden_size, output_size)

        # Optimizer (uninitialized)
        self.opt = None

        # Put model weights on device
        self.to(self.device)

    def forward(self, x, h0=None):
        """
        Returns model prediction for input sequence `x`, with all hidden
        states..
        """
        x = x.to(self.device)

        hn = torch.empty(self.num_states, self.num_layers, x.size(1),
                         self.hidden_size, device=self.device)

        for i, cell in enumerate(self.cells):
            x, hni = cell(x, h0[:, i, :, :] if h0 is not None else None)
            for state_id, state in enumerate(hni):
                hn[state_id, i, ...] = state

        x = self.linear(x)

        return x, hn

    def predict(self, x, h0=None):
        """
        Returns model prediction for input sequence `x`, without the gradients
        and with only the last hidden states.
        """
        with torch.no_grad():

            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(self.device)
                x = x.unsqueeze(1)

            x, hn = self.forward(x, h0)

            return x.cpu().numpy(), hn

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, state_dict):
        return self.load_state_dict(state_dict)

    def warmup_step(self, input, stabilization_increment, stabilization_max,
                    num_estimates_vaa=10, stabilization_measure=10_000,
                    epsilon=1e-4, vaa_target=0.95, double=False):
        """"
        Perform one gradient step of the warmup procedure.
        """
        # Compute vaa*
        stabilization = min(1 + self.num_warmup_step * stabilization_increment,
                            stabilization_max)
        stabilization = torch.randint(1, stabilization + 1, ())

        # Compute loss
        loss_vaa_star = self.vaa_star(
            input, stabilization, epsilon, double=double)

        if loss_vaa_star.isnan().any():
            print("Error: NaNs")
            import sys
            sys.exit(1)

        target = torch.full_like(loss_vaa_star, vaa_target)
        loss = F.mse_loss(loss_vaa_star, target)

        wandb.log({
            'warmup/step': self.num_warmup_step,
            'warmup/vaa*': loss_vaa_star.mean().item(),
        })

        # Optimize weights
        self.warmup_opt.zero_grad()
        loss.backward()
        self.warmup_opt.step()

        # Count steps
        self.num_warmup_step += 1
        print("#", end='', flush=True)

        return loss.item()

    def warmup(self, inputs, num_steps, learning_rate, batch_size,
               stabilization_increment=10, stabilization_max=200,
               num_estimates_vaa=10, stabilization_measure=10_000,
               epsilon=1e-4, vaa_target=0.95, double=False):
        """
        Perform the warmup procedure.
        """
        self.warmup_opt = optim.Adam(self.parameters(), lr=learning_rate)

        if isinstance(inputs, list):
            num_samples = len(inputs)
        else:
            inputs = torch.from_numpy(inputs)
            num_samples = inputs.size(1)

        self.num_warmup_step = 0
        while True:
            permutation = torch.randperm(num_samples)

            for i in range(0, num_samples, batch_size):

                # Draw minibatch
                if isinstance(inputs, list):
                    input = [inputs[j] for j in permutation[i:i + batch_size]]
                else:
                    input = inputs[:, permutation[i:i + batch_size], :]

                self.warmup_step(
                    input,
                    stabilization_increment=stabilization_increment,
                    stabilization_max=stabilization_max,
                    num_estimates_vaa=num_estimates_vaa,
                    stabilization_measure=stabilization_measure,
                    epsilon=epsilon,
                    vaa_target=vaa_target,
                    double=double
                )

                if self.num_warmup_step >= num_steps:
                    print()
                    return

    def training_step(self, input, target, mask, learning_rate):
        """
        Performs a training gradient step.

        Either input, target, mask are tensors, either they are lists
        of numpy arrays
        """
        if (isinstance(input, list) and isinstance(target, list) and
                isinstance(mask, list)):

            input = [torch.from_numpy(x) for x in input]
            target = [torch.from_numpy(x) for x in target]
            mask = [torch.from_numpy(x) for x in mask]

            input = nn.utils.rnn.pad_sequence(input)
            target = nn.utils.rnn.pad_sequence(target)
            mask = nn.utils.rnn.pad_sequence(mask)

        loss = self.loss(input, target, mask)

        if self.opt is None:
            self.opt = optim.Adam(self.parameters(), lr=learning_rate)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def train(self, dataset, num_epochs, learning_rate, batch_size,
              logger=None, best_model=True, num_estimates_vaa=10,
              stabilization_measure=10_000, epsilon=1e-4):
        """
        Performs the training procedure.
        """
        # Statistics
        num_gradient_steps = 0
        best_loss = float('inf')
        num_epochs_no_improvement = 0
        best_weights = deepcopy(self.state_dict())

        train, valid = dataset.train_test_split(test_ratio=0.2)

        # Transform to PyTorch Tensors
        train.inputs = torch.from_numpy(train.inputs)
        train.targets = torch.from_numpy(train.targets)
        train.masks = torch.from_numpy(train.masks)
        valid.inputs = torch.from_numpy(valid.inputs)
        valid.targets = torch.from_numpy(valid.targets)
        valid.masks = torch.from_numpy(valid.masks)

        with torch.no_grad():
            valid_loss = self.eval(valid, batch_size)
            train_loss = self.eval(train, batch_size)
            if not valid.regression:
                valid_acc = self.eval(valid, batch_size, accuracy=True)
            else:
                valid_acc = None

        if logger is not None:
            logger({
                'train/epoch': 0,
                'train/valid_loss': valid_loss,
                'train/train_loss': train_loss,
                'train/valid_acc': valid_acc,
            })

        for epoch in range(num_epochs):
            permutation = torch.randperm(train.size)
            epoch_loss = 0.0

            for i in range(0, train.size, batch_size):

                print(f'{i}/{train.size}')

                # Prepare minibatch
                indices = permutation[i:i + batch_size]

                input = train.inputs[:, indices, :].to(self.device)
                target = train.targets[:, indices, :].to(self.device)
                mask = train.masks[:, indices, :].to(self.device)

                # Compute and backpropagate loss, optimize weights
                loss = self.training_step(input, target, mask, learning_rate)

                # Update statistics
                epoch_loss += loss
                num_gradient_steps += 1

                print('\033[F\033[K', end='')

            train_loss = epoch_loss / int(train.size / batch_size)

            with torch.no_grad():
                valid_loss = self.eval(valid, batch_size)
                train_loss = self.eval(train, batch_size)
                if not valid.regression:
                    valid_acc = self.eval(valid, batch_size, accuracy=True)
                else:
                    valid_acc = None

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_weights = deepcopy(self.state_dict())
                num_epochs_no_improvement = 0
            else:
                num_epochs_no_improvement += 1

            mean_vaa = 0.0
            for _ in range(num_estimates_vaa):
                indices = torch.randint(
                    low=0, high=train.size, size=(batch_size,))
                input = train.inputs[:, indices, :]
                vaa = self.vaa(input, stabilization_measure, epsilon)
                mean_vaa += vaa / num_estimates_vaa

            print(f'Epoch {epoch:04d}')
            print(f'\tTrain: {train_loss:.4f}, Valid: {valid_loss:.4f}, '
                  f'Mean VAA: {mean_vaa:.4f}')

            if logger is not None:
                logger({
                    'train/epoch': epoch + 1,
                    'train/train_loss': train_loss,
                    'train/valid_loss': valid_loss,
                    'train/valid_acc': valid_acc,
                    'train/mean_vaa': mean_vaa,
                })

        if best_model:
            self.load_state_dict(best_weights)

    def _regression_loss(self, inputs, targets, masks):
        """
        Compute MSE loss on the time steps indicated by the mask.
        """
        outputs, _ = self.forward(inputs.to(self.device))

        ordered_masks = masks.transpose(0, 1).to(outputs.device)
        return F.mse_loss(outputs.transpose(0, 1)[ordered_masks],
                          targets.transpose(0, 1).flatten().to(outputs.device))

    def accuracy(self, inputs, targets, masks):
        outputs, _ = self.forward(inputs.to(self.device))

        num_classes = masks.size(-1)

        ordered_masks = masks.transpose(0, 1).to(self.device)
        outputs = outputs.transpose(0, 1)[ordered_masks]
        outputs = outputs.view(-1, num_classes)
        targets = targets.transpose(0, 1)[:, 0, 0].to(outputs.device)

        return (outputs.argmax(dim=-1) == targets).float().mean()

    def _classification_loss(self, inputs, targets, masks):
        """
        Compute CE loss on the time steps indicated by the mask.
        """
        outputs, _ = self.forward(inputs.to(self.device))

        num_classes = masks.size(-1)

        ordered_masks = masks.transpose(0, 1).to(self.device)
        outputs = outputs.transpose(0, 1)[ordered_masks]
        outputs = outputs.view(-1, num_classes)
        targets = targets.transpose(0, 1)[:, 0, 0].to(outputs.device)

        return F.cross_entropy(outputs, targets)

    def eval(self, dataset, batch_size, num_samples=None, accuracy=False):
        """
        Evaluate the model on the sequences from `dataset`.
        """
        if num_samples is not None:
            permutation = np.random.choice(dataset.size, size=num_samples,
                                           replace=False)
        else:
            permutation = range(dataset.size)
            num_samples = dataset.size

        # Transform to PyTorch Tensors
        if isinstance(dataset.inputs, np.ndarray):
            dataset.inputs = torch.from_numpy(dataset.inputs)
            dataset.targets = torch.from_numpy(dataset.targets)
            dataset.masks = torch.from_numpy(dataset.masks)

        with torch.no_grad():
            total_loss = 0.0

            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i + batch_size]

                input = dataset.inputs[:, indices, :]
                target = dataset.targets[:, indices, :]
                mask = dataset.masks[:, indices, :]

                if accuracy:
                    loss = self.accuracy(input, target, mask)
                else:
                    loss = self.loss(input, target, mask)
                total_loss += loss.item()

        return total_loss / int(num_samples / batch_size)

    def sample_hidden_states(self, sequences):
        """
        Sample random hidden states from the batch of sequences.
        """
        if isinstance(sequences, list):
            h0 = torch.empty(self.num_states, self.num_layers, len(sequences),
                             self.hidden_size, device=self.device)

            for i, seq in enumerate(sequences):
                _, h0i = self.forward(seq.unsqueeze(1))
                for state_id, state in enumerate(h0i):
                    h0[state_id, :, i:i+1, :] = state
        else:

            h0 = torch.empty(self.num_states, self.num_layers,
                             sequences.size(1), self.hidden_size,
                             device=self.device)

            for i in range(sequences.size(1)):

                c = torch.randint(1, sequences.size(0) + 1, ())
                _, h0i = self.forward(sequences[:c, i:i+1, :])

                for state_id, state in enumerate(h0i):
                    h0[state_id, :, i:i+1, :] = state

        return h0

    def vaa_star(self, inputs, stabilization, epsilon=1e-4, num_samples=None, double=False):
        """
        Estimates the VAA* on all layers of the RNN by sampling a batch of
        input sequences and a random perturbation.
        """
        if num_samples is not None:
            samples = _sample_inputs(inputs, num_samples, self.device)
        else:
            if isinstance(inputs, list):
                num_samples = len(inputs)
                samples = [torch.from_numpy(i).to(self.device) for i in inputs]
            else:
                num_samples = inputs.size(1)
                samples = inputs.to(self.device)

        h0 = self.sample_hidden_states(samples)

        vaas = torch.empty(self.num_layers)
        for i, cell in enumerate(self.cells):
            u = torch.randn(1, 1, cell.input_size, device=self.device)
            u = u.expand(stabilization, num_samples, -1)
            if double:
                h0i_warmup = h0[:, i, :, :int(
                    self.hidden_size / 2)].contiguous()
                _, hn = cell.cell_1(u, h0i_warmup)
            else:
                _, hn = cell(u, h0[:, i, ...])
            hn = torch.cat(list(hn), dim=-1)
            vaas[i] = _vaa_star(hn, epsilon=epsilon)

        return vaas

    def mean_vaa_star(self, *args, **kwargs):
        with torch.no_grad():
            return self.vaa_star(*args, **kwargs).mean().item()

    def vaa(self, inputs, stabilization, epsilon=1e-4, num_samples=None):
        """
        Estimates the VAA on of the RNN by sampling a batch of input sequences
        and a random perturbation.
        """
        if num_samples is not None:
            samples = _sample_inputs(inputs, num_samples, self.device)
        else:
            if isinstance(inputs, list):
                num_samples = len(inputs)
                samples = [torch.from_numpy(i).to(self.device) for i in inputs]
            else:
                num_samples = inputs.size(1)
                samples = inputs.to(self.device)

        with torch.no_grad():

            h0 = self.sample_hidden_states(samples)

            # Update hidden state with stable input in stacked recurrent cells
            u = torch.randn(1, 1, self.input_size, device=self.device)
            u = u.expand(stabilization, num_samples, -1)
            _, hn = self.forward(u, h0)

            # Compute next hidden state
            _, hnp1 = self.forward(u[0:1, :, :], hn)

            # Concatenate all hidden states from all layers
            hn = torch.cat(list(hn), dim=-1)
            hn = torch.cat([h for h in hn], dim=-1)

            # Compute variability among attractors in final hidden states
            return _vaa(hn, epsilon)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
