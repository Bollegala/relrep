all:
	g++ -Wall -fopenmp -std=c++0x -I eigen -O4 train.cc -o train

run:
	./train --dim=100 --wpid_fname=./work/benchmark.wpids --patid_fname=./work/benchmark.patids --matrix_fname=./work/benchmark.ppmi --pos=./work/unsup.pos --neg=./work/unsup.neg --epohs=100 --output=./work/wreps.txt

clean:
	rm full 
