import numpy as np

schema = {
    "nodeid": np.uint16,
    "age": np.int32,
    "dob": np.int32,
    "dod": np.int32,
    "susceptibility": np.uint8,
    "ri_timer": np.uint16,
    "etimer": np.uint8,
    "itimer": np.uint8,
    "susceptibility_timer": np.uint16,
    "accessibility": np.uint8,
    # below are for line-list in-place reporting experiment
    #"age_at_infection": np.int32,
    #"time_at_infection": np.int16,
    #"node_at_infection": np.uint16
}

