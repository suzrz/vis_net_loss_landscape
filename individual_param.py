import torch
import data_load
import numpy as np
import prep
from interpolate import *


def run_single(args):
    # setup
    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    args = prep.parse_arguments()
    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = Interpolator(model, device, alpha, final_state, init_state)

    interpolate.single_acc_vloss(test_loader, args.layer, args.idxs)
