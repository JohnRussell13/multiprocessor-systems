t1: v1-1.cpp
	mpicc v1-1.cpp -o t1
t2: v1-2.cpp
	mpicc v1-2.cpp -o t2
t3: v1-3.cpp
	mpicc v1-3.cpp -o t3
t4: v1-4.cpp
	mpicc v1-4.cpp -o t4
t5: v1-5.cpp
	g++ -fopenmp -Wall v1-5.cpp -o t5
t6: v1-6.cpp
	g++ -fopenmp -Wall v1-6.cpp -o t6
e1: paral.cpp
	g++ -fopenmp -Wall paral.cpp -o e1

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o t1 t2 t3 t4 t5 t6 e1