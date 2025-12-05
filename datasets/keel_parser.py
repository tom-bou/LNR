"""
Parser for KEEL .dat files (similar to ARFF format).
Handles parsing of KEEL dataset files and extraction of features and labels.
"""

import numpy as np
from pathlib import Path
from collections import Counter

try:
    from scipy.io import arff
    HAS_SCIPY_ARFF = True
except ImportError:
    HAS_SCIPY_ARFF = False
    print("Warning: scipy.io.arff not available. Using custom parser.")


def parse_arff_file(file_path):
    """
    Parse an ARFF/.dat file and return data and metadata.
    
    Args:
        file_path: Path to the .dat or .arff file
        
    Returns:
        tuple: (data, attributes, relation_name)
            - data: numpy array of shape (n_samples, n_features+1) where last column is class
            - attributes: list of attribute names
            - relation_name: name of the relation/dataset
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try using scipy.io.arff first
    if HAS_SCIPY_ARFF:
        try:
            data, meta = arff.loadarff(str(file_path))
            # Convert structured array to regular array
            data_array = np.array([list(row) for row in data])
            attributes = [attr[0] for attr in meta.names()]
            relation_name = meta.name
            return data_array, attributes, relation_name
        except Exception as e:
            print(f"scipy.io.arff failed, using custom parser: {e}")
    
    # Custom parser for KEEL .dat format
    return parse_keel_dat_file(file_path)


def parse_keel_dat_file(file_path):
    """
    Custom parser for KEEL .dat files.
    KEEL .dat format is similar to ARFF but may have slight differences.
    
    Args:
        file_path: Path to the .dat file
        
    Returns:
        tuple: (data, attributes, relation_name)
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    relation_name = None
    attributes = []
    attribute_types = []
    data_start_idx = None
    
    # Parse header
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        
        line_lower = line.lower()
        
        # Parse relation name
        if line_lower.startswith('@relation'):
            relation_name = line.split(maxsplit=1)[1].strip().strip("'\"")
        
        # Parse attributes
        elif line_lower.startswith('@attribute'):
            parts = line.split()
            if len(parts) >= 3:
                attr_name = parts[1].strip("'\"")
                attr_type = parts[2].strip("'\"")
                attributes.append(attr_name)
                attribute_types.append(attr_type)
        
        # Find data section
        elif line_lower.startswith('@data'):
            data_start_idx = i + 1
            break
    
    if data_start_idx is None:
        raise ValueError("No @data section found in file")
    
    # Parse data
    data_rows = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        
        # Split by comma, handling quoted strings
        values = []
        current_value = ""
        in_quotes = False
        
        for char in line:
            if char == '"' or char == "'":
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                values.append(current_value.strip().strip("'\""))
                current_value = ""
            else:
                current_value += char
        
        if current_value:
            values.append(current_value.strip().strip("'\""))
        
        if len(values) == len(attributes):
            data_rows.append(values)
    
    # Convert to numpy array, handling numeric conversion
    data_array = []
    for row in data_rows:
        converted_row = []
        for i, val in enumerate(row):
            if i < len(attribute_types) and attribute_types[i].lower() in ['numeric', 'real', 'integer']:
                try:
                    converted_row.append(float(val))
                except (ValueError, TypeError):
                    converted_row.append(np.nan)
            else:
                converted_row.append(val)
        data_array.append(converted_row)
    
    if not data_array:
        raise ValueError("No data rows found in file")
    
    data_array = np.array(data_array, dtype=object)
    
    return data_array, attributes, relation_name or "unknown"


def load_keel_dataset(dataset_path, train_file=None, test_file=None):
    """
    Load a KEEL dataset from file(s).
    
    Args:
        dataset_path: Path to dataset directory or single .dat file
        train_file: Optional path to training file (if separate from test)
        test_file: Optional path to test file
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, class_names)
            If only one file provided, returns (X, y, None, None, class_names)
    """
    dataset_path = Path(dataset_path)
    
    # If single file provided
    if dataset_path.is_file():
        data, attributes, _ = parse_arff_file(dataset_path)
        X = data[:, :-1].astype(float)
        y_str = data[:, -1]
        
        # Convert string labels to integers
        unique_labels = np.unique(y_str)
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_to_int[label] for label in y_str], dtype=np.int64)
        
        return X, y, None, None, unique_labels.tolist()
    
    # If directory, look for train/test files
    if dataset_path.is_dir():
        # Look for common file patterns
        dat_files = list(dataset_path.glob('*.dat'))
        
        if train_file or test_file:
            # Use specified files
            if train_file:
                train_data, train_attrs, _ = parse_arff_file(train_file)
            else:
                train_data, train_attrs = None, None
            
            if test_file:
                test_data, test_attrs, _ = parse_arff_file(test_file)
            else:
                test_data, test_attrs = None, None
        else:
            # Auto-detect train/test files
            train_files = [f for f in dat_files if 'train' in f.name.lower() or 'tra' in f.name.lower()]
            test_files = [f for f in dat_files if 'test' in f.name.lower() or 'tst' in f.name.lower()]
            
            if train_files and test_files:
                train_data, train_attrs, _ = parse_arff_file(train_files[0])
                test_data, test_attrs, _ = parse_arff_file(test_files[0])
            elif len(dat_files) == 1:
                # Single file, split manually (80/20) with stratified shuffle
                data, attributes, _ = parse_arff_file(dat_files[0])
                
                # Extract labels for stratified split
                y_labels = data[:, -1]
                unique_labels = np.unique(y_labels)
                
                # Stratified shuffle split to maintain class distribution
                np.random.seed(42)  # For reproducibility
                train_indices = []
                test_indices = []
                
                for label in unique_labels:
                    label_indices = np.where(y_labels == label)[0]
                    np.random.shuffle(label_indices)
                    n_train_label = int(0.8 * len(label_indices))
                    train_indices.extend(label_indices[:n_train_label])
                    test_indices.extend(label_indices[n_train_label:])
                
                # Shuffle the indices
                np.random.shuffle(train_indices)
                np.random.shuffle(test_indices)
                
                train_data = data[train_indices]
                test_data = data[test_indices]
                train_attrs = test_attrs = attributes
            else:
                raise ValueError(f"Could not determine train/test files in {dataset_path}")
        
        # Process training data
        if train_data is not None:
            X_train = train_data[:, :-1].astype(float)
            y_train_str = train_data[:, -1]
        else:
            X_train, y_train_str = None, None
        
        # Process test data
        if test_data is not None:
            X_test = test_data[:, :-1].astype(float)
            y_test_str = test_data[:, -1]
        else:
            X_test, y_test_str = None, None
        
        # Get all unique labels from both sets
        all_labels = []
        if y_train_str is not None:
            all_labels.extend(y_train_str.tolist())
        if y_test_str is not None:
            all_labels.extend(y_test_str.tolist())
        
        unique_labels = np.unique(all_labels)
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert labels to integers
        if y_train_str is not None:
            y_train = np.array([label_to_int[label] for label in y_train_str], dtype=np.int64)
        else:
            y_train = None
        
        if y_test_str is not None:
            y_test = np.array([label_to_int[label] for label in y_test_str], dtype=np.int64)
        else:
            y_test = None
        
        return X_train, y_train, X_test, y_test, unique_labels.tolist()
    
    raise ValueError(f"Invalid dataset path: {dataset_path}")


def get_class_distribution(y):
    """Get class distribution from labels."""
    counter = Counter(y)
    return dict(counter)

