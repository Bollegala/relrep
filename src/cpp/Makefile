all:
	g++ -Wall -fopenmp -std=c++0x -I eigen -O4 train.cc -o train

run:
	./train --dim=300 --wpid_fname=./work/benchmark.wpids --patid_fname=./work/benchmark.patids \
	--matrix_fname=./work/benchmark.ppmi --pos=./work/unsup.pos --neg=./work/usup.neg \
	--epohs=1 --alpha=0.1 --prefile=./work/glove.pretrained --output=./work/wreps.txt

clean:
	rm -f train ../work/wreps.txt
