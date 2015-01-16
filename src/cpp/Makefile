all:
	g++ -Wall -fopenmp -std=c++0x -I eigen -O4 train.cc -o train

run:
	./train --dim=10 --wpid_fname=./work/benchmark.wpids --patid_fname=./work/benchmark.patids --matrix_fname=./work/benchmark.ppmi

clean:
	rm full 
