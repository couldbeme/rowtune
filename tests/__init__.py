import warnings

warnings.simplefilter("ignore", ResourceWarning)  # tests read small files with one-line open().read()
