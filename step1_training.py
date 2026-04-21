"""
Step 1: Distributed Deep Learning Methods Implementation with Model Saving

Implements four distributed learning paradigms:
1. Federated Learning (FL) - McMahan et al., AISTATS 2017
2. Split Learning (SL) - Vepakomma et al., NeurIPS 2018 Workshop
3. Parallel Split Learning (PSL) - Jeon & Kim, ICOIN 2020
4. SplitFed Learning (SFL) - Thapa et al., AAAI 2022
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from abc import ABC, abstractmethod
import copy
import numpy as np
from typing import List, Dict, Tuple
import os
import argparse

from utils.model_utils import get_models
from utils.dataset_utils import get_dataset
from utils.data_distribution import data_distribution


# ============== Base Class ==============
class DistributedLearningBase(ABC):
    """Abstract base class for distributed learning methods."""

    def __init__(
        self,
        num_clients: int,
        num_classes: int,
        learning_rate: float = 0.01,
        local_epochs: int = 1,
        batch_size: int = 64,
    ):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.lr = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()

    @abstractmethod
    def train_round(self, client_loaders: List[DataLoader]) -> float:
        """Execute one round of distributed training."""
        pass

    @abstractmethod
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the global model on the test set."""
        pass

    @abstractmethod
    def save_models(self, checkpoint_dir: str):
        """Save trained model checkpoints."""
        pass


# ============== Federated Learning ==============
class FederatedLearning(DistributedLearningBase):
    """Federated Averaging (FedAvg) algorithm."""

    def __init__(self, num_clients: int, num_classes: int, **kwargs):
        super().__init__(num_clients, num_classes, **kwargs)
        self.global_model = FullModel(num_classes).to(device)

    def train_round(self, client_loaders: List[DataLoader]) -> float:
        """One round of FedAvg: local training followed by model aggregation."""
        client_weights = []
        client_samples = []
        total_loss = 0.0

        for loader in client_loaders:
            # Copy global model to client
            local_model = copy.deepcopy(self.global_model)
            optimizer = optim.SGD(
                local_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
            )

            # Local training
            local_model.train()
            client_loss = 0.0

            for _ in range(self.local_epochs):
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    client_loss += loss.item() * data.size(0)

            client_weights.append(copy.deepcopy(local_model.state_dict()))
            client_samples.append(len(loader.dataset))
            total_loss += client_loss

        # FedAvg aggregation
        self._aggregate_models(client_weights, client_samples)

        total_samples_processed = sum(client_samples) * self.local_epochs
        return total_loss / total_samples_processed if total_samples_processed > 0 else 0.0

    def _aggregate_models(self, client_weights: List[Dict], client_samples: List[int]):
        """Weighted averaging of client model parameters."""
        total_samples = sum(client_samples)
        global_dict = self.global_model.state_dict()

        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for weights, n_samples in zip(client_weights, client_samples):
                global_dict[key] += (n_samples / total_samples) * weights[key].float()

        self.global_model.load_state_dict(global_dict)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.global_model.eval()
        test_loss, correct = 0.0, 0
        n_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                test_loss += self.criterion(output, target).item() * data.size(0)
                correct += (output.argmax(1) == target).sum().item()
                n_samples += data.size(0)

        return test_loss / n_samples, correct / n_samples

    def save_models(self, checkpoint_dir: str):
        torch.save(
            {"model_state_dict": self.global_model.state_dict()},
            os.path.join(checkpoint_dir, "global_model.pt"),
        )
        print(f"Saved FL global model to {checkpoint_dir}/global_model.pt")


# ============== Split Learning ==============
class SplitLearning(DistributedLearningBase):
    """Vanilla Split Learning (sequential client processing)."""

    def __init__(self, num_clients: int, num_classes: int, **kwargs):
        super().__init__(num_clients, num_classes, **kwargs)
        self.server_model = ServerModel(num_classes).to(device)
        self.client_model = ClientModel().to(device)

    def train_round(self, client_loaders: List[DataLoader]) -> float:
        total_loss = 0.0
        total_samples_processed = 0

        server_optimizer = optim.SGD(
            self.server_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        client_optimizer = optim.SGD(
            self.client_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )

        for loader in client_loaders:
            for _ in range(self.local_epochs):
                for data, target in loader:
                    data, target = data.to(device), target.to(device)

                    # Client forward pass
                    self.client_model.train()
                    client_optimizer.zero_grad()
                    smashed_data = self.client_model(data)

                    smashed_data_detached = smashed_data.detach().requires_grad_(True)

                    # Server forward and backward
                    self.server_model.train()
                    server_optimizer.zero_grad()
                    output = self.server_model(smashed_data_detached)
                    loss = self.criterion(output, target)
                    loss.backward()
                    server_optimizer.step()

                    # Propagate gradients back to client
                    grad_smashed = smashed_data_detached.grad
                    smashed_data.backward(grad_smashed)
                    client_optimizer.step()

                    total_loss += loss.item() * data.size(0)
                    total_samples_processed += data.size(0)

        return total_loss / total_samples_processed if total_samples_processed > 0 else 0.0

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.client_model.eval()
        self.server_model.eval()
        test_loss, correct = 0.0, 0
        n_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                smashed = self.client_model(data)
                output = self.server_model(smashed)
                test_loss += self.criterion(output, target).item() * data.size(0)
                correct += (output.argmax(1) == target).sum().item()
                n_samples += data.size(0)

        return test_loss / n_samples, correct / n_samples

    def save_models(self, checkpoint_dir: str):
        torch.save(
            {"model_state_dict": self.client_model.state_dict()},
            os.path.join(checkpoint_dir, "client_model.pt"),
        )
        torch.save(
            {"model_state_dict": self.server_model.state_dict()},
            os.path.join(checkpoint_dir, "server_model.pt"),
        )
        print(f"Saved SL models to {checkpoint_dir}/")


# ============== Parallel Split Learning ==============
class ParallelSplitLearning(DistributedLearningBase):
    """
    Parallel Split Learning with client synchronization.

    Key improvements:
    1. Removed double-normalization of gradients to prevent vanishing learning rates.
    2. Added SGD momentum for client models to improve convergence.
    3. Unified learning rate across client and server.
    """

    def __init__(self, num_clients: int, num_classes: int, **kwargs):
        super().__init__(num_clients, num_classes, **kwargs)
        self.server_model = ServerModel(num_classes).to(device)
        # Client models initialized on CPU to reduce VRAM usage
        self.client_models = [ClientModel().to("cpu") for _ in range(num_clients)]
        self._synchronize_client_weights()
        # Chunk size for processing clients in batches
        self.chunk_size = 16

    def _synchronize_client_weights(self):
        """Synchronize all client models to match client 0."""
        base_state_dict = self.client_models[0].state_dict()
        for k in range(1, self.num_clients):
            self.client_models[k].load_state_dict(copy.deepcopy(base_state_dict))

    def train_round(self, client_loaders: List[DataLoader]) -> float:
        assert len(client_loaders) == self.num_clients

        total_loss = 0.0
        total_samples_processed = 0

        # Server optimizer
        server_optimizer = optim.SGD(
            self.server_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )

        # Client master optimizer (attached to client 0, updates propagate via synchronization)
        client_optimizer = optim.SGD(
            self.client_models[0].parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )

        for epoch in range(self.local_epochs):
            client_iters = [iter(loader) for loader in client_loaders]

            while True:
                # Collect active data from all clients
                active_data_cache = []
                for k in range(self.num_clients):
                    try:
                        data, target = next(client_iters[k])
                        active_data_cache.append((k, data, target))
                    except StopIteration:
                        continue

                if len(active_data_cache) == 0:
                    break

                num_active = len(active_data_cache)

                # Zero gradients
                server_optimizer.zero_grad()
                client_optimizer.zero_grad()

                # Dictionary to accumulate gradients on CPU
                total_client_grads = {}

                # Process clients in chunks with gradient accumulation
                for i in range(0, num_active, self.chunk_size):
                    chunk_batch = active_data_cache[i : i + self.chunk_size]
                    chunk_smashed = []
                    chunk_targets = []
                    chunk_indices = []

                    # Client forward pass
                    for k, data, target in chunk_batch:
                        self.client_models[k].to(device)
                        self.client_models[k].train()
                        data, target = data.to(device), target.to(device)

                        smashed = self.client_models[k](data)

                        chunk_smashed.append(smashed)
                        chunk_targets.append(target)
                        chunk_indices.append(k)

                    if not chunk_smashed:
                        continue

                    # Server forward pass
                    combined_smashed = torch.cat(chunk_smashed, dim=0)
                    combined_targets = torch.cat(chunk_targets, dim=0)
                    smashed_detached = combined_smashed.detach().requires_grad_(True)

                    self.server_model.train()
                    output = self.server_model(smashed_detached)
                    loss = self.criterion(output, combined_targets)

                    # Scale loss to ensure gradient accumulation equals global mean gradient
                    loss_scale = len(chunk_batch) / num_active
                    (loss * loss_scale).backward()

                    total_loss += loss.item() * len(chunk_batch)

                    # Extract server gradients
                    if smashed_detached.grad is not None:
                        grad_smashed = smashed_detached.grad.clone()
                    else:
                        grad_smashed = torch.zeros_like(smashed_detached)

                    # Client backward pass and gradient accumulation
                    start_idx = 0
                    for idx, k in enumerate(chunk_indices):
                        batch_len = chunk_smashed[idx].size(0)
                        grad_portion = grad_smashed[start_idx : start_idx + batch_len]
                        start_idx += batch_len

                        self.client_models[k].zero_grad()
                        chunk_smashed[idx].backward(grad_portion)

                        # Accumulate gradients to CPU dictionary
                        with torch.no_grad():
                            for name, param in self.client_models[k].named_parameters():
                                if param.grad is not None:
                                    if name not in total_client_grads:
                                        total_client_grads[name] = torch.zeros_like(
                                            param.data, device="cpu"
                                        )
                                    total_client_grads[name] += param.grad.to("cpu")

                        # Offload client model back to CPU
                        self.client_models[k].to("cpu")

                    # Release GPU memory
                    del (
                        chunk_smashed,
                        combined_smashed,
                        smashed_detached,
                        grad_smashed,
                        output,
                        loss,
                    )
                    torch.cuda.empty_cache()

                total_samples_processed += sum(len(x[1]) for x in active_data_cache)

                # Server optimization step with gradient clipping
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), max_norm=5.0)
                server_optimizer.step()

                # Client optimization step: apply accumulated gradients to master client
                with torch.no_grad():
                    for name, param in self.client_models[0].named_parameters():
                        if name in total_client_grads:
                            param.grad = total_client_grads[name].to(param.device)

                # Clip client gradients before optimization step
                torch.nn.utils.clip_grad_norm_(self.client_models[0].parameters(), max_norm=5.0)
                client_optimizer.step()

                # Synchronize updated weights from master client to all other clients
                self._synchronize_client_weights()

        return total_loss / total_samples_processed if total_samples_processed > 0 else 0.0

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.client_models[0].to(device)
        self.client_models[0].eval()
        self.server_model.eval()
        test_loss, correct = 0.0, 0
        n_samples = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                smashed = self.client_models[0](data)
                output = self.server_model(smashed)
                test_loss += self.criterion(output, target).item() * data.size(0)
                correct += (output.argmax(1) == target).sum().item()
                n_samples += data.size(0)
        self.client_models[0].to("cpu")
        return test_loss / n_samples, correct / n_samples

    def save_models(self, checkpoint_dir: str):
        for k in range(self.num_clients):
            torch.save(
                {"model_state_dict": self.client_models[k].state_dict()},
                os.path.join(checkpoint_dir, f"client_{k}_model.pt"),
            )
        torch.save(
            {"model_state_dict": self.server_model.state_dict()},
            os.path.join(checkpoint_dir, "server_model.pt"),
        )
        print(f"Saved PSL models ({self.num_clients} clients + 1 server) to {checkpoint_dir}/")


# ============== SplitFed Learning ==============
class SplitFedLearning(DistributedLearningBase):
    """
    SplitFed Learning (SFLV2 - Sequential Server Update).

    The server model updates sequentially per client for faster convergence,
    while client models are aggregated using FedAvg.
    """

    def __init__(self, num_clients: int, num_classes: int, **kwargs):
        super().__init__(num_clients, num_classes, **kwargs)
        self.global_server_model = ServerModel(num_classes).to(device)
        self.global_client_model = ClientModel().to(device)

    def train_round(self, client_loaders: List[DataLoader]) -> float:
        client_model_weights = []
        client_samples_count = []
        total_loss = 0.0

        # Differential learning rates for server and client
        server_lr = self.lr
        client_lr = self.lr * 0.1

        # Server optimizer persists within the round for sequential updates
        server_optimizer = optim.SGD(
            self.global_server_model.parameters(), lr=server_lr, momentum=0.9, weight_decay=5e-4
        )

        # Iterate through clients with sequential server updates
        for loader in client_loaders:
            # Client starts with global client weights
            local_client = copy.deepcopy(self.global_client_model)
            client_optimizer = optim.SGD(
                local_client.parameters(), lr=client_lr, momentum=0.9, weight_decay=5e-4
            )

            client_loss = 0.0
            n_samples = 0

            for _ in range(self.local_epochs):
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    batch_size = data.size(0)

                    # Client forward pass
                    local_client.train()
                    client_optimizer.zero_grad()
                    smashed = local_client(data)
                    smashed_detached = smashed.detach().requires_grad_(True)

                    # Server forward pass using current global weights
                    self.global_server_model.train()
                    server_optimizer.zero_grad()
                    output = self.global_server_model(smashed_detached)
                    loss = self.criterion(output, target)
                    loss.backward()

                    # Server update with gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.global_server_model.parameters(), max_norm=5.0
                    )
                    server_optimizer.step()

                    # Client backward pass
                    if smashed_detached.grad is not None:
                        grad_smashed = smashed_detached.grad.clone()
                        # Clamp gradients for numerical stability
                        grad_smashed = torch.clamp(grad_smashed, -5.0, 5.0)
                        smashed.backward(grad_smashed)

                        # Clip client gradients
                        torch.nn.utils.clip_grad_norm_(local_client.parameters(), max_norm=5.0)
                        client_optimizer.step()

                    client_loss += loss.item() * batch_size
                    n_samples += batch_size

            # Store client weights for aggregation
            client_model_weights.append(copy.deepcopy(local_client.state_dict()))
            client_samples_count.append(n_samples)
            total_loss += client_loss

        # FedAvg aggregation for client models
        self._fedavg_client_models(client_model_weights, client_samples_count)

        return total_loss / sum(client_samples_count) if sum(client_samples_count) > 0 else 0.0

    def _fedavg_client_models(self, client_weights: List[Dict], client_samples: List[int]):
        """Standard FedAvg aggregation for client-side models."""
        total_samples = sum(client_samples)
        global_dict = self.global_client_model.state_dict()

        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for weights, n_samples in zip(client_weights, client_samples):
                global_dict[key] += (n_samples / total_samples) * weights[key].float()

        self.global_client_model.load_state_dict(global_dict)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.global_client_model.eval()
        self.global_server_model.eval()
        test_loss, correct = 0.0, 0
        n_samples = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                smashed = self.global_client_model(data)
                output = self.global_server_model(smashed)
                test_loss += self.criterion(output, target).item() * data.size(0)
                correct += (output.argmax(1) == target).sum().item()
                n_samples += data.size(0)
        return test_loss / n_samples, correct / n_samples

    def save_models(self, checkpoint_dir: str):
        torch.save(
            {"model_state_dict": self.global_client_model.state_dict()},
            os.path.join(checkpoint_dir, "global_client_model.pt"),
        )
        torch.save(
            {"model_state_dict": self.global_server_model.state_dict()},
            os.path.join(checkpoint_dir, "global_server_model.pt"),
        )
        print(f"Saved SFL global models to {checkpoint_dir}/")


def create_client_loaders(dataset, num_clients, num_classes, batch_size, mode):
    """
    Split the dataset into client-specific DataLoaders using logic from data_distribution.py.
    The 'mode' parameter supports: "equal_size_equal_class" "unequal_size_equal_class" "equal_size_unequal_class" "unequal_size_unequal_class".
    """
    # 1. Get indices for each client from the utility module
    client_indices = data_distribution(dataset, num_clients, num_classes, mode)
    
    loaders = []
    for indices in client_indices:
        # 2. Create a Subset for each client based on their indices
        subset = Subset(dataset, indices)
        # 3. Wrap the subset in a DataLoader
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True))
        
    return loaders


def train_and_evaluate(
    method_name: str,
    method: DistributedLearningBase,
    client_loaders: List[DataLoader],
    test_loader: DataLoader,
    num_rounds: int = 10,
):
    """Train and evaluate a distributed learning method."""
    print(f"\n{'='*50}")
    print(f"Training: {method_name}")
    print(f"{'='*50}")

    for round_idx in range(num_rounds):
        train_loss = method.train_round(client_loaders)
        test_loss, accuracy = method.evaluate(test_loader)

        print(
            f"Round {round_idx+1:2d} | Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | Accuracy: {accuracy*100:.2f}%"
        )

    return accuracy

def get_num_classes(dataset):
    """
    Robustly detect the number of classes for different types of Torchvision datasets.
    """
    # Most datasets (MNIST, CIFAR10, etc.) have 'targets'
    if hasattr(dataset, 'targets'):
        labels = torch.tensor(dataset.targets)
        return len(torch.unique(labels))
    
    # Some datasets have a 'classes' attribute (list of class names)
    elif hasattr(dataset, 'classes'):
        return len(dataset.classes)
    
    # Specific handling for GTSRB (stores data in _samples as (path, label))
    elif hasattr(dataset, '_samples'):
        labels = torch.tensor([s[1] for s in dataset._samples])
        return len(torch.unique(labels))
    
    # Final fallback: sample the dataset to infer the number of unique labels
    else:
        try:
            # We sample all labels from the dataset
            all_labels = [dataset[j][1] for j in range(len(dataset))]
            return len(set(all_labels))
        except Exception as e:
            raise AttributeError(f"Could not determine the number of classes. Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Distributed Deep Learning Framework")
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--experiments_dir", type=str, default="./experiments")
    parser.add_argument("--data_dir", type=str, default="./data")    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--iid", choices=["true", "false"], default="true")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_distribution", type=str, default="equal_size_equal_class", choices=["equal_size_equal_class","unequal_size_equal_class","equal_size_unequal_class","unequal_size_unequal_class"])
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Configuration for {os.path.basename(__file__)}")
    print(f"{'='*50}")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"{'='*50}")

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset, num_classes = get_dataset(args.dataset, args.data_dir)
    num_classes = get_num_classes(train_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    client_loaders = create_client_loaders(
        train_dataset, args.num_clients, num_classes, args.batch_size, args.data_distribution
    )

    global ClientModel, ServerModel, FullModel
    ClientModel, ServerModel, FullModel = get_models(args.model)

    strategies = [
        ("FL", "Federated Learning (FedAvg)", FederatedLearning),
        ("SL", "Split Learning (Sequential)", SplitLearning),
        ("PSL", "Parallel Split Learning", ParallelSplitLearning),
        ("SFL", "SplitFed Learning (SFLV2)", SplitFedLearning),
    ]

    results = {}

    for key, name, StrategyClass in strategies:
        learner = StrategyClass(
            num_clients=args.num_clients,
            num_classes=num_classes,
            learning_rate=args.learning_rate,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
        )

        acc = train_and_evaluate(name, learner, client_loaders, test_loader, args.num_rounds)
        results[key] = acc

        checkpoint_dir = os.path.join(args.experiments_dir, "checkpoints", key)
        os.makedirs(checkpoint_dir, exist_ok=True)
        learner.save_models(checkpoint_dir)

    print(f"\n{'='*60}")
    print("Final Results Summary")
    print(f"{'='*60}")
    for method, acc in results.items():
        print(f"  {method:4s}: {acc*100:.2f}%")

    print(f"\nAll models saved to: {args.experiments_dir}/checkpoints/")


if __name__ == "__main__":
    main()
