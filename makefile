all: t1 t2 t3 t4 t5 t6 e1

t1: v1-1.cpp
	mpic++ v1-1.cpp -o t1
t2: v1-2.cpp
	mpic++ v1-2.cpp -o t2
t3: v1-3.cpp
	mpic++ v1-3.cpp -o t3
t4: v1-4.cpp
	mpic++ v1-4.cpp -o t4
t5: v1-5.cpp
	g++ -fopenmp -Wall v1-5.cpp -o t5
t6: v1-6.cpp
	g++ -fopenmp -Wall v1-6.cpp -o t6
e1: paral.cpp
	g++ -fopenmp -Wall paral.cpp -o e1

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o t1 t2 t3 t4 t5 t6 e1
	
run:
	mpirun -np 4 ./t1
	mpirun -np 4 ./t2
	mpirun -np 4 ./t3
	mpirun -np 4 ./t4 vector.txt matrix.txt
	./t5 6
	./t6 6
	./e1 6
