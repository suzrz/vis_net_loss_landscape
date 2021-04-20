import prep
import data_load
from interpolate import *

logger = logging.getLogger("vis_net")


def run_quadr_interpolation(args):
    logger.info("Running quadratic interpolation of parameters")

    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.debug(f"Device: {device}")

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate_s = Interpolator(model, device, alpha, final_state, init_state)

    interpolate_s.single_acc_vloss_q(test_loader, args.layer, args.idxs)

    alpha_vec = np.linspace(0, 1, args.alpha_steps)
    interpolate_v = Interpolator(model, device, alpha_vec, final_state, init_state)
    interpolate_v.vec_acc_vloss_q(test_loader, args.layer)
