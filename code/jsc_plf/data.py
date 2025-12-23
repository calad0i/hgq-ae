from pathlib import Path

import h5py as h5
import numpy as np


def get_data(data_path: Path | str, n_constituents: int, ptetaphi: bool, val_size: float = 0.1):
    data_path = Path(data_path)

    with h5.File(data_path / '150c-train.h5') as f:
        X_train_val = np.array(f['feature'])
        y_train_val = np.array(f['label'])
    with h5.File(data_path / '150c-test.h5') as f:
        X_test = np.array(f['feature'])
        y_test = np.array(f['label'])

    X_train_val = X_train_val[:, :n_constituents]
    y_train_val = y_train_val.astype(np.int32)
    X_test = X_test[:, :n_constituents]
    y_test = y_test.astype(np.int32)

    scale = np.std(X_train_val.astype(np.float32), axis=(0, 1), keepdims=True)
    shift = np.mean(X_train_val.astype(np.float32), axis=(0, 1), keepdims=True)

    if ptetaphi:
        X_train_val = X_train_val[..., [5, 8, 11]]
        X_test = X_test[..., [5, 8, 11]]

    X_train_val = (X_train_val - shift) / scale
    X_test = (X_test - shift) / scale

    order = np.arange(len(X_train_val))
    np.random.shuffle(order)

    X_train_val, y_train_val = X_train_val[order], y_train_val[order]

    n_train = int((1 - val_size) * len(X_train_val))
    X_train, X_val = X_train_val[:n_train], X_train_val[n_train:]
    y_train, y_val = y_train_val[:n_train], y_train_val[n_train:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
