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
single = Path(os.path.join(directory, "singleParam"))
single_img = Path(os.path.join(imgs, "singleParam"))
if not single.exists():
    os.makedirs(single)
if not single_img.exists():
    os.makedirs(single_img)

sf_loss_path = Path(os.path.join(directory, "sf_loss"))
sf_acc_path = Path(os.path.join(directory, "sf_acc"))
svloss_path = Path(os.path.join(single, "svloss"))  # results\svloss
svloss_img_path = Path(os.path.join(single_img, "svloss"))
#stloss_path = Path(os.path.join(directory, "stloss"))
sacc_path = Path(os.path.join(single, "sacc"))
sacc_img_path = Path(os.path.join(single_img, "sacc"))
vvloss_path = Path(os.path.join(directory, "vvloss"))
vvloss_img_path = Path(os.path.join(imgs, "vvloss"))
#vtloss_path = Path(os.path.join(directory, "vtloss"))
vacc_path = Path(os.path.join(directory, "vacc"))
vacc_img_path = Path(os.path.join(imgs, "vacc"))

train_subs_loss = Path(os.path.join(directory, "train_subs_loss"))
train_subs_acc = Path(os.path.join(directory, "train_subs_acc"))

test_subs_loss = Path(os.path.join(directory, "test_subs_loss"))
test_subs_acc = Path(os.path.join(directory, "test_subs_acc"))

epochs_loss = Path(os.path.join(directory, "epochs_loss"))
epochs_acc = Path(os.path.join(directory, "epochs_acc"))

