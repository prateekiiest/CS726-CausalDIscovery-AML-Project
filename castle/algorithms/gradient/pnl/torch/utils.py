
import torch
from torch.utils.data import Dataset, DataLoader


class SampleDataset(Dataset):
    """
    construct class for DataLoader

    Parameters
    ----------
    data: sequential array
        if data contains more than one samples set,
        the number of samples in all data must be equal.
    """

    def __init__(self, *data):
        super(SampleDataset, self).__init__()
        if len(set([x.shape[0] for x in data])) != 1:
            raise ValueError("The number of samples in all data must be equal.")
        self.data = data
        self.n_samples = data[0].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        return [d[index] for d in self.data]


def batch_loader(*x, batch_size=64, **kwargs):

    dataset = SampleDataset(*x)
    loader = DataLoader(dataset, batch_size=batch_size, **kwargs)

    return loader


def compute_jacobian(func, inputs):
    """
    Function that computes the Jacobian of a given function.

    See Also
    --------
    torch.autograd.functional.jacobian
    """

    return torch.autograd.functional.jacobian(func, inputs, create_graph=True)


def compute_entropy(x):
    """Computation information entropy of x"""

    distr = torch.distributions.Normal(loc=torch.mean(x),
                                       scale=torch.std(x))
    entropy = distr.entropy()

    return entropy
