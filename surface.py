import net
import math
import torch
import numpy as np
from paths import *

logger = logging.getLogger("vis_net")


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

    def model_normalize(self, ref_point, order=2):
        for p in self.parameters:
            p = p * (ref_point.model_norm(order) / self.model_norm(order))


def get_rand_like(original, device):
    """
    Function generates random direction with size equivalent to start

    :param original: Original point
    :param device: device
    :return: random direction
    """
    logger.debug("Getting new random direction")
    r = []

    for p in original.parameters:
        r.append(torch.rand(size=p.size(), dtype=original.parameters[0].dtype).to(device))

    return Parameters(r)


def get_ortogonal_dir(original, device):
    """
    Function generates direction orthogonal to point with model-wise normalization

    :param original: original vector
    :param device: device
    :return: new vector
    """
    logger.debug("Getting new orthogonal direction")
    r = get_rand_like(original, device)
    r = r - (original * r.dot_prod(original)) / math.pow(original.model_norm(), 2)

    return r


def convert_parameters_to_state_dict(params, model):
    """
    Function converts list of tensors to state dict param for PyTorch model

    :param params: list to be converted
    :param model: model for which the parameters are converted
    :return: updated state dict
    """
    keys = model.state_dict().keys()
    updated_dict = {k: t for (k, t) in zip(keys, params.parameters)}

    return updated_dict


def rand_2d(model, device, steps, test_loader):
    """
    Function calculates values of loss function around a point in the parameter space

    :param model: model to be evaluated
    :param device: device
    :param steps: number of steps to take
    :param test_loader: test dataset loader
    :return: matrix of values of loss function around the point
    """
    start_p = Parameters([p.to(device) for p in model.parameters()])
    distance = 1

    d1 = get_rand_like(start_p, device)
    d2 = get_ortogonal_dir(d1, device)

    # directions normalization
    d1.model_normalize(start_p)
    d2.model_normalize(start_p)

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
            logger.debug(f"Evaluating for i: {i}, j: {j}")
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
