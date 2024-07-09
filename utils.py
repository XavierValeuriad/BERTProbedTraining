import logging, os


def create_folder_if_not_exists(folder_path: str):
    if not os.path.exists(folder_path):
        logging.info(f'Creating folder `{folder_path}`.')
        os.makedirs(folder_path)


def create_all_subfolders_if_not_exists(folder_path: str):
    logging.info(f'Checking if `{folder_path}` exist, and creating it if not.')
    if folder_path:
        path = os.path.normpath(folder_path)
        splitted_path = path.split(os.sep)
        if len(splitted_path) == 1:
            if '.' not in path:
                create_folder_if_not_exists(splitted_path[0])
        elif len(splitted_path) > 1:
            subpath = os.path.join(path[0], splitted_path[1])
            if '.' not in subpath:
                create_folder_if_not_exists(subpath)
            for i, _directory in enumerate(splitted_path[2:]):
                if '.' not in _directory:
                    subpath = os.path.join(subpath, _directory)
                    create_folder_if_not_exists(subpath)
                else:
                    logging.warning(f"Only directories, which means with names not containing a dot '.', are created. Thus, it is assumed that {os.path.join(splitted_path[i:])} is a file.")
                    break


import torch
from torch import Tensor


MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64


def hash_tensor(x: Tensor) -> Tensor:
    assert x.dtype == torch.int64
    while x.ndim > 0:
        x = _reduce_last_axis(x)
    return x


@torch.no_grad()
def _reduce_last_axis(x: Tensor) -> Tensor:
    assert x.dtype == torch.int64
    acc = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        acc *= MULTIPLIER
        acc += INCREMENT
        acc += x[..., i]
        # acc %= MODULUS  # Not really necessary.
    return acc