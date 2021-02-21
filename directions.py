import torch
import paths

def random_dir(model, device):
    weights = [p.data for p in model.parameters()]
    direction = [torch.randn(w.size(), device=device) for w in weights]

    # normalize
    assert (len(direction) == len(weights))

    for d, w in zip(direction, weights):
        for dire, wei in zip(d, w):
            # print("WEIGHT: ", wei)
            # print("DIR: ", dire)
            dire.mul_(wei.norm() / (dire.norm() + 1e-10))

    return direction

def random_directions(model, device):
    x_dir = random_dir(model, device)
    y_dir = random_dir(model, device)

    return [x_dir, y_dir]

def set_surf_file(file):
    """
    Prepare surface file for random directions experiment
    """
