init:
	pip3 install opencv-python
	pip3 install tensorflow
	pip3 install mpi4py

old:
	python3 classical.py

pass_o:
	python3 pass_cl.py

run_py:
	mpiexec -n 5 python main.py > res.txt

pass:
	mpiexec -n 7 python pass.py

img_gen:
	python3 img_gen.py

cpass: pass.c
	mpic++ pass.c -o cpass

main: main.c
	mpic++ main.c -o main

cnob: no_bias.c
	mpic++ no_bias.c -o no_bias

cpass_run:
	mpirun -np 7 ./cpass

run:
	mpirun -np 15 ./main 100 50 30 10

nob:
	mpirun -np 7 ./no_bias

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o cpass main img/* log/*
