from lib import examine_surface, data_load, prep
from lib.examine1D import *

logger = logging.getLogger("vis_net")


def run_rand_dirs(args):
    logger.info("Running loss function landscape visualization using random directions")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.debug(f"Device: {device}")

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    dirs = examine_surface.get_directions(model, device)
    examine_surface.calc_loss(model, test_loader, dirs, device)
    plot.surface3d_rand_dirs()
    plot.surface_heatmap_rand_dirs()
