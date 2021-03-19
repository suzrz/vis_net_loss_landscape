import os
import logging
from pathlib import Path


# general directories
results = Path(os.path.join(".", "results"))
imgs = Path(os.path.join(".", "imgs"))

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

# final loss and accuracy of the model
sf_loss_path = Path(os.path.join(results, "final_loss"))
sf_acc_path = Path(os.path.join(results, "final_acc"))

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
    dirs = [results, imgs, single, single_img, vec, vec_img, prelim]

    for dir in dirs:
        logging.debug("[paths]: Searching for {}...".format(dir))
        if not dir.exists():
            logging.debug("[paths]: Creating new {} directory...".format(dir))
            os.makedirs(dir)
