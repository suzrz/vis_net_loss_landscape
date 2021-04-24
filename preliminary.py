import prep
import data_load
from examine1D import *


logger = logging.getLogger("vis_net")


def run_preliminary(args):
    """
    Function executes preliminary experiments

    :param args: command line arguments
    """
    subs_train = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 50000, 60000]
    subs_test = [1000, 1500, 2000, 3000, 4000, 5000, 7000, 8000, 9000, 10000]
    epochs = [2, 5, 10, 15, 17, 20, 22, 25, 27, 30]

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.debug(f"Device: {device}")

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)
    model.load_state_dict(init_state)

    net.pre_train_subset(model, device, subs_train, args.epochs, test_loader)
    net.pre_test_subset(model, device, subs_test)
    net.pre_epochs(model, device, epochs)

    plot.plot_impact(subs_train, train_subs_loss, train_subs_acc, xlabel="Size of training dataset")
    plot.plot_impact(epochs, epochs_loss, epochs_acc, annotate=False, xlabel="Number of epochs")
    plot.plot_box(subs_test, show=True, xlabel="Size of test subset")
