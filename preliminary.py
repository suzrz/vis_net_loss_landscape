import torch
import prep
import nnvis


def run_preliminary(args, device):
    """
    Function executes preliminary experiments

    :param args: command line arguments
    :param device: device to be used
    """
    subs_train = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 50000, 60000]
    subs_test = [1000, 1500, 2000, 3000, 4000, 5000, 7000, 8000, 9000, 10000]
    epochs = [2, 5, 10, 15, 17, 20, 22, 25, 27, 30]

    train_loader, test_loader = nnvis.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)
    model.load_state_dict(torch.load(nnvis.init_state))

    nnvis.pre_train_subset(model, device, subs_train, args.epochs, test_loader)
    nnvis.pre_test_subset(model, device, subs_test)
    nnvis.pre_epochs(model, device, epochs)

    nnvis.plot_impact(subs_train, nnvis.train_subs_loss, nnvis.train_subs_acc, xlabel="Size of training dataset")
    nnvis.plot_impact(epochs, nnvis.epochs_loss, nnvis.epochs_acc, annotate=False, xlabel="Number of epochs")
    nnvis.plot_box(subs_test, show=False, xlabel="Size of test subset")
