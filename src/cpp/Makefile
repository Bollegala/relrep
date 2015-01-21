all:
	g++ -Ofast -march=native -O4 -Wall -fopenmp -std=c++0x -I eigen train.cc -o train
	

run:
	./train --dim=300 --wpid_fname=./work/benchmark.wpids --patid_fname=./work/benchmark.patids \
	--matrix_fname=./work/benchmark.ppmi --pos=./work/small.pos --neg=./work/small.neg \
	--epohs=2 --alpha=0.01  \
	--prefile=./work/glove840B.pretrained --output=./work/small_models/models_840B_tanh --loss=tanh

clean:
	rm -f train ../work/wreps.txt 


# g++ -Ofast -march=native -Wall -fopenmp -std=c++0x -I eigen train.cc -o train