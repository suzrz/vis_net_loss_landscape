import prep
import data_load
from interpolate import *


def run_layers(args):
    """
    Function setups and executes experiment of interpolation of parameters

    :param args: experimental setup
    """
    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = Interpolator(model, device, alpha, final_state, init_state)

    interpolate.vec_acc_vlos(test_loader, args.layer, args.trained)