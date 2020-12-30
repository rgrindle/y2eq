from torch.utils.data import Dataset


class TensorDatasetGPU(Dataset):
    r"""Dataset wrapping tensors.

    GPU version

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index].cuda() for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class TensorDatasetCPU(Dataset):
    r"""Dataset wrapping tensors.

    CPU version

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
