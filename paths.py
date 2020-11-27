import os
from pathlib import Path


directory = "results"

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
sf_loss_path = Path(os.path.join(directory, "sf_loss"))
sf_acc_path = Path(os.path.join(directory, "sf_acc"))
svloss_path = Path(os.path.join(directory, "svloss"))
stloss_path = Path(os.path.join(directory, "stloss"))
sacc_path = Path(os.path.join(directory, "sacc"))
vvloss_path = Path(os.path.join(directory, "vvloss"))
vtloss_path = Path(os.path.join(directory, "vtloss"))
vacc_path = Path(os.path.join(directory, "vacc"))

train_subs_loss = Path(os.path.join(directory, "train_subs_loss"))
train_subs_acc = Path(os.path.join(directory, "train_subs_acc"))

test_subs_loss = Path(os.path.join(directory, "test_subs_loss"))
test_subs_acc = Path(os.path.join(directory, "test_subs_acc"))
