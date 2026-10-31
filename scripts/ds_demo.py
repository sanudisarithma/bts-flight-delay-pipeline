
import os
import random
import sys
import timeit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from preprocess import config

random.seed(1)

airports = pd.read_csv(config.RAW_CSV, usecols=["airport"])["airport"].dropna().astype(str).tolist()
unique_codes = sorted(set(airports))
print("rows:", len(airports), "  different airport codes:", len(unique_codes))

# 1. is this code one of the known airports?  list against set
as_list = list(unique_codes)
as_set = set(unique_codes)
checks = random.sample(airports, 20000)

t_list = timeit.timeit(lambda: [c in as_list for c in checks], number=3) / 3
t_set = timeit.timeit(lambda: [c in as_set for c in checks], number=3) / 3
print("20,000 membership checks")
print("  list: %.4f s   set: %.4f s   set is %.0f times faster" % (t_list, t_set, t_list / t_set))

# 2. counting rows per (carrier, airport) pair - the key has to be a tuple
pairs = pd.read_csv(config.RAW_CSV, usecols=["carrier", "airport"]).dropna()
counts = {}
for c, a in zip(pairs["carrier"], pairs["airport"]):
    key = (c, a)             # a tuple can be a dict key
    counts[key] = counts.get(key, 0) + 1
print("different carrier-airport pairs:", len(counts))

try:
    bad = {}
    bad[["AA", "JFK"]] = 1   # a list can not be a dict key
except TypeError as e:
    print("a list as a dict key fails:", e)
