# load_sampler.py

from utils.sampler import Sampler
import time

start = time.time()

# 87k mini training dataset: 3.09 s
s1 = Sampler(
    "/home/pinkychow/biomass-estimation/data/biomass_dataset_train_87k.h5",
    range(1, 87000),
)
s1()

print("Bootstrapping time (s): ", time.time() - start)

# 2e6 full training dataset: 453.27 s (~ 7.5 min)
s2 = Sampler(
    "/home/pinkychow/biomass-estimation/data/biomass_dataset_train_2e6.h5",
    range(1, 2000000),
)
s2()

print("Bootstrapping time (s): ", time.time() - start)
