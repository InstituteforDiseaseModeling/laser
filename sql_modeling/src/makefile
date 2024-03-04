# Define variables
PYTHON := python3
CC := g++
CFLAGS := -shared -fPIC -O3 -march=native -flto -fpermissive# -mfma
# ALSO NO: -finline-functions
# NO: -floop-parallelize-all
# neutral: -funroll-loops
# NO: -ffast-math
#CFLAGS := -shared -fPIC -O0 -g
DB_SCRIPT := python3 -m laser_numpy_model.csv_to_db
PREPROC_SCRIPT := ./eula_preproc.sh
CREATE_POP_SCRIPT := python3 -m laser_numpy_model.create_pop_as_csv
POP_SEEDED_CSV := pop_seeded.csv
EULA_DB := eula.db
UPDATE_AGES_SO := update_ages.so
MODELED_POP_CSV := modeled_pop.csv
EULA_POP_CSV := eula_pop.csv

# Default target
all: $(EULA_DB) $(MODELED_POP_CSV)

# Rule to build update_ages.so from update_ages.c
$(UPDATE_AGES_SO): update_ages.cpp
	$(CC) $(CFLAGS) -o $@ $<
	

# Rule to create eula.db from eula_pop.csv
$(EULA_DB): eula_pop.csv
	rm -f $(EULA_DB)
	$(DB_SCRIPT) $<

$(POP_SEEDED_CSV): settings.py
	$(CREATE_POP_SCRIPT)

# Rule to create modeled_pop.csv and pop_seeded.csv
$(MODELED_POP_CSV) $(EULA_POP_CSV): $(POP_SEEDED_CSV)
	$(PREPROC_SCRIPT) $(POP_SEEDED_CSV) 5

eula_pops_20yr_noaging.csv: eula_binned.csv
	$(PYTHON) utils/eula_precalc.py eula_pops_20yr_noaging.csv

fits.npy: eula_pops_20yr_noaging.csv
	$(PYTHON) utils/fit_eula_pops.py eula_pops_20yr_noaging.csv

run:
	python3 tlc.py

# Clean target to remove generated files
clean:
	rm -f $(UPDATE_AGES_SO) $(EULA_DB) $(POP_SEEDED_CSV) $(MODELED_POP_CSV)

# Phony targets
.PHONY: all clean

