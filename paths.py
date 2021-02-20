import os
from pathlib import Path


directory = "results"
imgs = "imgs"

init_state = Path(os.path.join(directory, "init_state.pt"))
final_state = Path(os.path.join(directory, "final_state.pt"))

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
single = Path(os.path.join(directory, "singleParam"))
single_img = Path(os.path.join(imgs, "singleParam"))
if not single.exists():
    os.makedirs(single)
if not single_img.exists():
    os.makedirs(single_img)

# directories for vector paramaters experiments
vec = Path(os.path.join(directory, "vec"))
vec_img = Path(os.path.join(imgs, "vec"))
if not vec.exists():
    os.makedirs(vec)
if not vec_img.exists():
    os.makedirs(vec_img)

# directory for preliminary experiments results
prelim = Path(os.path.join(directory, "preliminary"))
if not prelim.exists():
    os.makedirs(prelim)

# final loss and accuracy of the model
sf_loss_path = Path(os.path.join(directory, "final_loss"))
sf_acc_path = Path(os.path.join(directory, "final_acc"))

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

epochs_loss = Path(os.path.join(directory, "epochs_loss"))
epochs_acc = Path(os.path.join(directory, "epochs_acc"))

# random directions experiment surface file
surf = Path(os.path.join(directory, "surf_file.h5"))
