from laser_core.demographics.kmestimator import KaplanMeierEstimator as KME

# Derived from table 1 of "National Vital Statistics Reports Volume 54, Number 14 United States Life Tables, 2003" (see README.md)
cumulative_deaths = [
    0,
    687,
    733,
    767,
    792,
    811,
    829,
    844,
    859,
    872,
    884,
    895,
    906,
    922,
    945,
    978,
    1024,
    1081,
    1149,
    1225,
    1307,
    1395,
    1489,
    1587,
    1685,
    1781,
    1876,
    1968,
    2060,
    2153,
    2248,
    2346,
    2449,
    2556,
    2669,
    2790,
    2920,
    3060,
    3212,
    3377,
    3558,
    3755,
    3967,
    4197,
    4445,
    4715,
    5007,
    5322,
    5662,
    6026,
    6416,
    6833,
    7281,
    7759,
    8272,
    8819,
    9405,
    10032,
    10707,
    11435,
    12226,
    13089,
    14030,
    15051,
    16146,
    17312,
    18552,
    19877,
    21295,
    22816,
    24445,
    26179,
    28018,
    29972,
    32058,
    34283,
    36638,
    39108,
    41696,
    44411,
    47257,
    50228,
    53306,
    56474,
    59717,
    63019,
    66336,
    69636,
    72887,
    76054,
    79102,
    81998,
    84711,
    87212,
    89481,
    91501,
    93266,
    94775,
    96036,
    97065,
    97882,
    100000,
]

class ExtendedKaplanMeierEstimator(KME):
    DEFAULT_FILE = "USA-pyramid-2023.csv"  # Replace with actual path

    def __init__(self, source=cumulative_deaths):
        """
        Initializes the ExtendedKaplanMeierEstimator.
        If no source is provided, the default dataset is loaded.
        """
        if source is None:
            # If no source is provided, load the default file
            source = Path(self.DEFAULT_FILE)
            if not source.exists() or not source.is_file():
                raise FileNotFoundError(f"Default file not found: {self.DEFAULT_FILE}")
            with source.open("r") as file:
                source = np.loadtxt(file, delimiter=",", usecols=1).astype(np.uint32)

        super().__init__(source)


