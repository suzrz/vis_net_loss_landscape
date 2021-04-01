import prep
import data_load
from interpolate import *


logger = logging.getLogger("vis_net")


def run_layers(args):
    """
    Function setups and executes experiment of interpolation of parameters

    :param args: experimental setup
    """
    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)
    logger.debug(f"Interpolation coefficient alpha "
                 f"start: {args.alpha_start}"
                 f"end: {args.alpha_end}"
                 f"steps: {args.alpha_steps}")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.debug(f"Device: {device}")

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = Interpolator(model, device, alpha, final_state, init_state)

    interpolate.vec_acc_vlos(test_loader, args.layer, args.trained)