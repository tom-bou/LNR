"""
PyTorch Dataset classes for KEEL datasets.
Follows the same pattern as CIFAR10_LT and CIFAR100_LT for consistency.
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

from .keel_parser import load_keel_dataset
from .sampler import ClassAwareSampler


class KEELDataset(Dataset):
    """
    PyTorch Dataset class for KEEL datasets.
    Similar structure to IMBALANCECIFAR10/IMBALANCECIFAR100.
    """
    
    def __init__(self, data, targets, transform=None, target_transform=None):
        """
        Initialize KEEL dataset.
        
        Args:
            data: numpy array of shape (n_samples, n_features)
            targets: numpy array of shape (n_samples,) with class labels
            transform: Optional transform to apply to features
            target_transform: Optional transform to apply to targets
        """
        self.data = data.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.transform = transform
        self.target_transform = target_transform
        
        # Calculate class distribution
        unique_classes = np.unique(self.targets)
        self.cls_num = len(unique_classes)
        
        # Create class data structure for class-aware sampling
        self.class_data = [[] for _ in range(self.cls_num)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)
        
        # Calculate cls_num_list (number of samples per class)
        self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(self.cls_num)]
        
        # Create num_per_cls_dict for compatibility
        self.num_per_cls_dict = {i: count for i, count in enumerate(self.cls_num_list)}
    
    def get_cls_num_list(self):
        """Get list of number of samples per class."""
        return self.cls_num_list
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample
            
        Returns:
            tuple: (index, features, target) to match existing pattern
        """
        features = self.data[index]
        target = self.targets[index]
        
        # Convert to tensor
        features = torch.from_numpy(features).float()
        
        # Apply transforms if provided
        if self.transform is not None:
            features = self.transform(features)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return index, features, target


class KEEL_LT(object):
    """
    KEEL Long-Tail dataset loader.
    Similar to CIFAR10_LT and CIFAR100_LT.
    """
    
    def __init__(self, distributed, dataset_name, root='./data/keel', 
                 batch_size=128, num_works=4, config=None):
        """
        Initialize KEEL dataset loader.
        
        Args:
            distributed: Whether to use distributed sampling
            dataset_name: Name of the dataset (e.g., 'yeast1', 'yeast3')
            root: Root directory for KEEL datasets
            batch_size: Batch size for DataLoaders
            num_works: Number of workers for DataLoaders
            config: Optional config object (for compatibility, not used for KEEL)
        """
        self.dataset_name = dataset_name
        self.root = Path(root)
        dataset_path = self.root / dataset_name
        
        # Load train and test data
        X_train, y_train, X_test, y_test, class_names = load_keel_dataset(dataset_path)
        
        if X_train is None or y_train is None:
            raise ValueError(f"Could not load training data for {dataset_name}")
        
        self.num_features = X_train.shape[1]
        
        # Normalize features to zero mean and unit variance (as per LNR paper)
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)
        self.std[self.std == 0] = 1.0  # Avoid division by zero
        
        X_train = (X_train - self.mean) / self.std
        if X_test is not None:
            X_test = (X_test - self.mean) / self.std
        
        # Create datasets
        train_dataset = KEELDataset(X_train, y_train)
        
        if X_test is not None and y_test is not None:
            eval_dataset = KEELDataset(X_test, y_test)
        else:
            # If no test set, use train set for evaluation (not ideal but works)
            print(f"Warning: No test set found for {dataset_name}, using train set for evaluation")
            eval_dataset = KEELDataset(X_train, y_train)
        
        # For validation, we can use a subset of training data or the test set
        # Following CIFAR pattern, create a validation dataset
        # For now, use test set as validation if available, otherwise use train
        if X_test is not None and y_test is not None:
            val_dataset = KEELDataset(X_test, y_test)
        else:
            # Use a subset of training data for validation
            n_val = int(0.1 * len(X_train))
            val_dataset = KEELDataset(X_train[:n_val], y_train[:n_val])
        
        self.cls_num_list = train_dataset.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        self.class_names = class_names
        
        print(f"Dataset {dataset_name} loaded:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Test samples: {len(eval_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Class distribution: {self.cls_num_list}")
        
        # Create distributed sampler if needed
        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        
        # Create DataLoaders
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True if not distributed else False,
            num_workers=num_works,
            pin_memory=True,
            sampler=self.dist_sampler
        )
        
        # Balanced sampler for training
        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_works,
            pin_memory=True,
            sampler=balance_sampler
        )
        
        # All training data without shuffling (for evaluation)
        self.train_all = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_works,
            pin_memory=True
        )
        
        # Evaluation DataLoader
        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_works,
            pin_memory=True
        )
        
        # Validation dataset (as dataset object, not DataLoader, to match CIFAR pattern)
        self.val = val_dataset
        self.train_dataset = train_dataset
        
        print(f'Train class distribution: {self.cls_num_list}')


def get_keel_dataset(dataset_name, root='./data/keel'):
    """
    Convenience function to get a KEEL dataset.
    
    Args:
        dataset_name: Name of the dataset
        root: Root directory for KEEL datasets
        
    Returns:
        KEEL_LT: Dataset loader object
    """
    return KEEL_LT(
        distributed=False,
        dataset_name=dataset_name,
        root=root,
        batch_size=128,
        num_works=4
    )


