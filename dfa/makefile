init:
	pip3 install opencv-python
	pip3 install tensorflow
	pip3 install mpi4py

old:
	python3 classical.py

pass_o:
	python3 pass_cl.py

run:
	mpiexec -n 3 python main.py

pass:
	mpiexec -n 3 python pass.py

.PHONY: clean
clean:
#	rm -f 
