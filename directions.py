import torch


def random_dir(model):
    weights = [p.data for p in model.parameters()]
    print(weights)
    direction = [torch.randn(w.size()) for w in weights]

    # normalize
    assert (len(direction) == len(weights))

    for d, w in zip(direction, weights):
        for dire, wei in zip(d, w):
            # print("WEIGHT: ", wei)
            # print("DIR: ", dire)
            dire.mul_(wei.norm() / (dire.norm() + 1e-10))

    return direction


def random_directions(model):
    x_dir = random_dir(model)
    y_dir = random_dir(model)

    return [x_dir, y_dir]
