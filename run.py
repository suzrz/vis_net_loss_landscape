import os
import random
import subprocess
from pathlib import Path

alpha_start = -1.0
alpha_end = 2.0
alpha_steps = 40
layers_c = ["conv1", "conv2"]
layers_f = ["fc1", "fc2", "fc3"]
#idxs_conv = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
#idxs_conv = [random.sample(range(1, 5), 4) for x in range(0, 20)]
#idxs_fc = [[0, 0], [1, 1], [2, 2]]
#idxs_fc = [random.sample(range(1, 10), 3) for x in range(0, 20)]
idxs_conv = [[1, 1, 1, 1]]
idxs_fc = [[1, 1]]
print(idxs_conv)
print(idxs_fc)

#main = Path(os.path.join("D:", os.path.join("vis_net_loss_landscape", "main.py")))
main = Path(os.path.join(".", "main.py"))
for l in layers_c:
    for i in idxs_conv:
        print((str(i[0])))

        subprocess.call(["py", main,
                         "--alpha-start", "{}".format(alpha_start),
                         "--alpha-end", "{}".format(alpha_end),
                         "--alpha-steps", "{}".format(alpha_steps),
                         "--layer", "{}".format(l),
                         "--idxs", str(i[0]), str(i[1]), str(i[2]), str(i[3]),
                         "--trained"
                         ])

for l in layers_f:
    for i in idxs_fc:
        subprocess.call(["py", main,
                         "--alpha-start", "{}".format(alpha_start),
                         "--alpha-end", "{}".format(alpha_end),
                         "--alpha-steps", "{}".format(alpha_steps),
                         "--layer", "{}".format(l),
                         "--idxs", str(i[0]), str(i[1]),
                         "--trained"
                         ])

