import os
from pathlib import Path

directory = "results"
"""
prefix |   meaning
-------|-------------
   s   | single param
   v   | whole vector 
----------------------
    second position
----------------------
 abbr. |   meaning
----------------------
   f   |    final
  v|t  | validation|train
"""
sf_loss_path = Path(os.path.join(directory, "sf_loss"))
sf_acc_path = Path(os.path.join(directory, "sf_acc"))
svloss_path = Path(os.path.join(directory, "svloss"))
stloss_path = Path(os.path.join(directory, "stloss"))
sacc_path = Path(os.path.join(directory, "sacc"))
vvloss_path = Path(os.path.join(directory, "vvloss"))
vtloss_path = Path(os.path.join(directory, "vtloss"))
vacc_path = Path(os.path.join(directory, "vacc"))

subs_loss = Path(os.path.join(directory, "subs_loss"))
subs_acc = Path(os.path.join(directory, "subs_acc"))

stab_loss = Path(os.path.join(directory, "stab_loss"))
stab_acc = Path(os.path.join(directory, "stab_acc"))