import os
import logging
from pathlib import Path


logger = logging.getLogger("vis_net")

# general directories
results = Path(os.path.join(".", "results"))
imgs = Path(os.path.join(".", "imgs"))

checkpoints = Path(os.path.join(".", "model_states"))

init_state = Path(os.path.join(results, "init_state.pt"))
final_state = Path(os.path.join(results, "final_state.pt"))

"""
  ----------------- FILENAMES MEANING -----------------
       first position               second position
    ----------------------      ----------------------
     prefix |   meaning            abbr.  |  meaning
    --------|-------------      ----------|-----------
       s    | single param           f    |   final    
       v    | whole vector          v|t   | val|train
    ----------------------      ----------------------
  ----------------- SUBSET FILENAMES -----------------


  -----------------------------------------------------
"""
# directories for single parameters experiments
single = Path(os.path.join(results, "singleParam"))
single_img = Path(os.path.join(imgs, "singleParam"))

# directories for vector paramaters experiments
vec = Path(os.path.join(results, "vec"))
vec_img = Path(os.path.join(imgs, "vec"))

# directory for preliminary experiments results
prelim = Path(os.path.join(results, "preliminary"))

# directory for random directions experiment
random_dirs = Path(os.path.join(results, "rand_dirs"))
random_dirs_img = Path(os.path.join(imgs, "rand_dirs"))

# directory for PCA directions
pca_dirs = Path(os.path.join(results, "PCA_dirs"))
pca_dirs_img = Path(os.path.join(imgs, "PCA_dirs"))

# actual loss and accuracy progress of the model
actual_loss_path = Path(os.path.join(results, "actual_loss"))
actual_acc_path = Path(os.path.join(results, "actual_acc"))

# final loss and accuracy of the model
sf_loss_path = Path(os.path.join(results, "final_loss"))
sf_acc_path = Path(os.path.join(results, "final_acc"))

# interpolation
loss_path = Path(os.path.join(results), "loss_all")
acc_path = Path(os.path.join(results), "acc_all")

# quadratic interpolation
q_loss_path = Path(os.path.join(results), "q_loss_all")
q_acc_path = Path(os.path.join(results), "q_acc_all")

# single loss experiments paths
svloss_path = Path(os.path.join(single, "svloss"))  # results\svloss
svloss_img_path = Path(os.path.join(single_img, "svloss"))

# single accuracy experiments paths
sacc_path = Path(os.path.join(single, "sacc"))
sacc_img_path = Path(os.path.join(single_img, "sacc"))

# vector loss experiments paths
vvloss_path = Path(os.path.join(vec, "vvloss"))
vvloss_img_path = Path(os.path.join(vec_img, "vvloss"))

# vector accuracy experiments paths
vacc_path = Path(os.path.join(vec, "vacc"))
vacc_img_path = Path(os.path.join(vec_img, "vacc"))

# preliminary experiments paths
train_subs_loss = Path(os.path.join(prelim, "train_subs_loss"))
train_subs_acc = Path(os.path.join(prelim, "train_subs_acc"))

test_subs_loss = Path(os.path.join(prelim, "test_subs_loss"))
test_subs_acc = Path(os.path.join(prelim, "test_subs_acc"))

epochs_loss = Path(os.path.join(prelim, "epochs_loss"))
epochs_acc = Path(os.path.join(prelim, "epochs_acc"))

# random directions experiment surface file
surf = Path(os.path.join(results, "surf_file.h5"))


def init_dirs():
    """
    Function initializes directories
    """
    logger.info("Initializing directories")
    dirs = [results, imgs, checkpoints, single, single_img, vec, vec_img, prelim,
            random_dirs, random_dirs_img, pca_dirs, pca_dirs_img]

    for d in dirs:
        logger.debug(f"Searching for {d}")
        if not d.exists():
            logger.debug(f"{d} not found. Creating...")
            os.makedirs(d)
