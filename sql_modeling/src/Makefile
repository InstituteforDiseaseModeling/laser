all: tlc.py

pop_1M_25nodes_seeded.csv:
	python3 create_pop_as_csv.py

update_ages.so: update_ages.c
	gcc -O3 -march=native -flto -o update_ages.so -shared -fPIC update_ages.c

tlc.py: pop_1M_25nodes_seeded.csv update_ages.so
	python3 tlc.py

age_gt_25.csv:
	./eula_preproc.sh pop_1M_25nodes_seeded.csv 25

eula.db:
	python3 csv_to_db.py age_gt_25.csv

run:
	python3 tlc.py

clean:
	rm -f pop_input.csv

test:
	python3 tests/test_calc_new_inf.py
