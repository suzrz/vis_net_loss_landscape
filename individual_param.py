import prep
import data_load
from interpolate import *

logger = logging.getLogger("vis_net")


def run_single(args):
    """
    Function executes experiment with individual parameter

    :param args: command line arguments
    """
    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)  # setup interpolation coefficient
    logger.debug(f"Interpolation coefficient alpha "
                 f"start: {args.alpha_start}"
                 f"end: {args.alpha_end}"
                 f"steps: {args.alpha_steps}")

    use_cuda = not args.no_cuda and torch.cuda.is_available()  # setup device
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.debug(f"Device: {device}")

    train_loader, test_loader = data_load.data_load()  # setup data loaders

    model = prep.get_net(device, train_loader, test_loader, args.epochs)  # setup model

    interpolate = Interpolator(model, device, alpha, final_state, init_state)  # get interpolator instance

    interpolate.single_acc_vloss(test_loader, args.layer, args.idxs)  # execute the experiment
