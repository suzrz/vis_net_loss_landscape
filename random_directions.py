import prep
import data_load
import surface
from interpolate import *

logger = logging.getLogger("vis_net")


def run_rand_dirs(args):
    logger.info("Running loss function landscape visualization")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.debug(f"Device: {device}")

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    d = surface.rand_2d(model, device, 20, test_loader)

    plot.plot_surface_contours(d, True)
    plot.surface_3d(d, 20, True)