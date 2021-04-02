import net
import math
import torch
import numpy as np
from paths import *


class Parameters:
    def __init__(self, parameters):
        self.parameters = parameters

    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Parameters([self.parameters[i].add(other) for i in range(len(self.parameters))])
        elif isinstance(other, self.__class__):
            return Parameters([self.parameters[i].add(other.parameters[i]) for i in range(len(self.parameters))])
        else:
            raise TypeError(f"Unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")

    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Parameters([self.parameters[i].sub(other) for i in range(len(self.parameters))])
        elif isinstance(other, self.__class__):
            return Parameters([self.parameters[i].sub(other.parameters[i]) for i in range(len(self.parameters))])
        else:
            raise TypeError(f"Unsupported operand type(s) for -: '{self.__class__}' and '{type(other)}'")

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Parameters([self.parameters[i].mul(other) for i in range(len(self.parameters))])
        elif isinstance(other, self.__class__):
            return Parameters([self.parameters[i].mul(other.parameters[i]) for i in range(len(self.parameters))])
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Parameters([self.parameters[i].div(other) for i in range(len(self.parameters))])
        elif isinstance(other, self.__class__):
            return Parameters([self.parameters[i].div(other.parameters[i]) for i in range(len(self.parameters))])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{self.__class__}' and '{type(other)}'")

    def dot_prod(self, other):
        aux = []

        for i in range(len(self.parameters)):
            aux.append((self.parameters[i] * other.parameters[i]).sum().item())

        return sum(aux)

    def model_norm(self, order=2):
        return math.pow(sum([torch.pow(layer, order).sum().item()
                             for layer in self.parameters]),
                        1.0 / order)


def get_rand_like(original, device):
    """
    Function generates random direction with size equivalent to start

    :param start: Defines size of the new direction
    :return: random direction
    """
    r = []

    for p in original.parameters:
        r.append(torch.rand(size=p.size(), dtype=original.parameters[0].dtype).to(device))

    return Parameters(r)


def get_ortogonal_dir(original, device, model):
    """
    Function generates direction orthogonal to point with model-wise normalization

    :param point: original vector
    :param model: model to be normalized to
    :return: new vector
    """
    r = get_rand_like(original, device)
    r = r - (original * r.dot_prod(original)) / math.pow(original.model_norm(), 2)

    return r


def convert_parameters_to_state_dict(params, model):
    keys = model.state_dict().keys()
    updated_dict = {k: t for (k, t) in zip(keys, params.parameters)}

    return updated_dict


def rand_2d(model, device, steps, test_loader):

    start_p = Parameters([p.to(device) for p in model.parameters()])
    distance = 1

    d1 = get_rand_like(start_p, device)
    d2 = get_ortogonal_dir(d1, device, model)

    # directions normalization

    # scale
    aux = start_p.model_norm() * distance / steps
    d1 = d1 * (aux / d1.model_norm())
    d2 = d2 * (aux / d2.model_norm())

    d1 = d1 * (steps / 2)
    d2 = d2 * (steps / 2)

    start_p = start_p - d1
    start_p = start_p - d2

    d1 = d1 / (steps / 2)
    d2 = d2 / (steps / 2)

    result = []

    for i in range(steps):
        column = []
        for j in range(steps):
            if i % 2 == 0:
                start_p = start_p + d2
                model.load_state_dict(convert_parameters_to_state_dict(start_p, model))
                loss, _ = net.test(model, test_loader, device)
                column.append(loss)
            else:
                start_p = start_p - d2
                model.load_state_dict(convert_parameters_to_state_dict(start_p, model))
                loss, _ = net.test(model, test_loader, device)
                column.insert(0, loss)

        result.append(column)
        start_p = start_p + d1

    np.savetxt(Path(os.path.join(random_dirs, f"rnd_dirs_{steps}_{distance}")), np.array(result))
    return np.array(result)
