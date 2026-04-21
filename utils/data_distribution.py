"""Splits a dataset into multiple client subsets under different distribution settings (IID and various non-IID scenarios) for federated learning experiments."""
import numpy as np
from collections import defaultdict

# ------------------------------------------------------------
# Extracts labels from various dataset formats in a unified way
# ------------------------------------------------------------
def get_labels(dataset):
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    elif hasattr(dataset, '_samples'): 
        return np.array([s[1] for s in dataset._samples])
    else:
        return np.array([dataset[i][1] for i in range(len(dataset))])

# ------------------------------------------------------------
# Generate groups with equal sample size and equal class number
# ------------------------------------------------------------
def partition_equal_size_equal_class(dataset, num_clients, min_samples=50):
    labels = get_labels(dataset)
    num_samples = len(labels)

    # Check if total samples satisfy minimum requirement
    if num_samples < num_clients * min_samples:
        raise ValueError(f"Total samples ({num_samples}) < {num_clients} * {min_samples}. "
                         f"Reduce num_clients or increase dataset size.")

    indices = np.random.permutation(num_samples)
    split_size = num_samples // num_clients

    client_indices = []
    for i in range(num_clients):
        start = i * split_size
        if i == num_clients - 1:
            end = num_samples   # assign remaining samples
        else:
            end = (i + 1) * split_size
        client_indices.append(indices[start:end].tolist())

    # Double check: ensure each client has at least min_samples
    for i, idxs in enumerate(client_indices):
        if len(idxs) < min_samples:
            raise RuntimeError(f"Client {i} has only {len(idxs)} samples after IID split, min_samples={min_samples}")

    return client_indices


# ------------------------------------------------------------
# Generate groups with unequal sample size but equal class number
# ------------------------------------------------------------
def partition_unequal_size_equal_class(dataset, num_clients, alpha=0.5, min_samples=50):
    labels = get_labels(dataset)
    num_samples = len(labels)

    # Check if total samples are sufficient
    if num_samples < num_clients * min_samples:
        raise ValueError(f"Total samples ({num_samples}) < {num_clients} * {min_samples}. "
                         f"Reduce num_clients or increase dataset size.")

    # 1. Generate proportions
    proportions = np.random.dirichlet(alpha * np.ones(num_clients))
    
    # 2. Initial size allocation
    sizes = (proportions * num_samples).astype(int)
    
    # 3. Ensure each client has at least min_samples
    for i in range(num_clients):
        if sizes[i] < min_samples:
            sizes[i] = min_samples
    
    # 4. Rebalance to match total sample count
    total_allocated = sizes.sum()
    if total_allocated > num_samples:
        # If exceeded: reduce from largest clients (but keep >= min_samples)
        excess = total_allocated - num_samples
        sorted_indices = np.argsort(sizes)[::-1]  # descending order
        for idx in sorted_indices:
            if excess <= 0:
                break
            deduct = min(sizes[idx] - min_samples, excess)
            sizes[idx] -= deduct
            excess -= deduct
        if excess > 0:
            raise RuntimeError("Unable to balance sample counts. Increase min_samples or total samples.")
    elif total_allocated < num_samples:
        # If insufficient: add to clients with largest proportions
        shortage = num_samples - total_allocated
        while shortage > 0:
            max_prop_idx = np.argmax(proportions)
            sizes[max_prop_idx] += 1
            shortage -= 1

    # 5. Shuffle and assign indices
    indices = np.random.permutation(num_samples)
    client_indices = []
    start = 0
    for size in sizes:
        end = start + size
        client_indices.append(indices[start:end].tolist())
        start = end

    # Final check
    for i, idxs in enumerate(client_indices):
        if len(idxs) < min_samples:
            raise RuntimeError(f"Client {i} has only {len(idxs)} samples after quantity_skew split, min_samples={min_samples}")

    return client_indices


# ------------------------------------------------------------
# Generate groups with equal sample size but unequal class number
# ------------------------------------------------------------
def partition_equal_size_unequal_class(dataset, num_clients, shards_per_client=2, min_samples=50):
    labels = get_labels(dataset)
    num_samples = len(labels)

    # Check if total samples satisfy minimum requirement
    if num_samples < num_clients * min_samples:
        raise ValueError(f"Total samples ({num_samples}) < {num_clients} * {min_samples}. "
                         f"Reduce num_clients or increase dataset size.")

    # 1. Sort indices by label
    indices = np.argsort(labels)
    
    # 2. Compute total number of shards
    total_shards = num_clients * shards_per_client
    shard_size = num_samples // total_shards
    
    # 3. Keep valid indices (discard remainder to ensure equal quantity)
    total_used_samples = total_shards * shard_size
    indices = indices[:total_used_samples]
    
    # 4. Split into shards and shuffle
    shards = [indices[i:i + shard_size] for i in range(0, len(indices), shard_size)]
    np.random.shuffle(shards)

    # 5. Assign shards to clients
    client_indices = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        for j in range(shards_per_client):
            client_indices[i].extend(shards[i * shards_per_client + j])

    # Ensure each client has at least min_samples
    for i, idxs in enumerate(client_indices):
        if len(idxs) < min_samples:
            raise RuntimeError(f"Client {i} has only {len(idxs)} samples after label_skew split, min_samples={min_samples}")

    return client_indices


# ------------------------------------------------------------
# Generate groups with unequal sample size and unequal class number
# ------------------------------------------------------------
def partition_unequal_size_unequal_class(dataset, num_clients, num_classes, alpha=0.5, min_samples=50):
    labels = get_labels(dataset)
    num_samples = len(labels)

    if num_samples < num_clients * min_samples:
        raise ValueError(f"Total samples ({num_samples}) < {num_clients} * {min_samples}. "
                         f"Reduce num_clients or increase dataset size.")

    client_indices = [[] for _ in range(num_clients)]

    # Allocate data per class
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        split = np.split(class_idx, proportions)

        for i in range(num_clients):
            client_indices[i].extend(split[i])

    # Post-processing: ensure each client has at least min_samples
    for i in range(num_clients):
        if len(client_indices[i]) < min_samples:
            max_client = max(range(num_clients), key=lambda j: len(client_indices[j]))
            needed = min_samples - len(client_indices[i])
            for _ in range(needed):
                sample = client_indices[max_client].pop()
                client_indices[i].append(sample)

    # Final check
    for i, idxs in enumerate(client_indices):
        if len(idxs) < min_samples:
            raise RuntimeError(f"Client {i} still has only {len(idxs)} samples after non_iid adjustment, min_samples={min_samples}")

    return client_indices


# ------------------------------------------------------------
# Main dispatcher function
# ------------------------------------------------------------
def data_distribution(dataset, num_clients, num_classes, mode, min_samples=50):
    """
    Split dataset among clients
    :param dataset: original training dataset (torchvision Dataset)
    :param num_clients: number of clients
    :param num_classes: number of classes (used in non_iid mode)
    :param mode: split mode ('equal_size_equal_class', 'unequal_size_equal_class', 'equal_size_unequal_class', 'unequal_size_unequal_class')
    :param min_samples: minimum samples per client (default=50, ensures watermark_size >= 1)
    :return: client_indices: List[List[int]]
    """
    if mode == "equal_size_equal_class":
        return partition_equal_size_equal_class(dataset, num_clients, min_samples)
    elif mode == "unequal_size_equal_class":
        return partition_unequal_size_equal_class(dataset, num_clients, alpha=0.5, min_samples=min_samples)
    elif mode == "equal_size_unequal_class":
        return partition_equal_size_unequal_class(dataset, num_clients, shards_per_client=2, min_samples=min_samples)
    elif mode == "unequal_size_unequal_class":
        return partition_unequal_size_unequal_class(dataset, num_clients, num_classes, alpha=0.5, min_samples=min_samples)
    else:
        raise ValueError(f"Invalid mode: {mode}")



