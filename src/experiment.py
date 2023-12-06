import numpy as np
import os

path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "data",
    "FakeNews",
    "gossipcop_0df94731",
    "new_profile_feature.npz",
)
data = np.load(path)
print(data["shape"])
data.close()
