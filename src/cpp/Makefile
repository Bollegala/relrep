all:
	g++ -Ofast -march=native -O4 -Wall -fopenmp -std=c++0x -I eigen train.cc -o train
	

run:
	./train --dim=100 --wpid_fname=./work/benchmark.wpids --patid_fname=./work/benchmark.patids \
	--matrix_fname=./work/benchmark.ppmi --pos=./work/unsup.pos --neg=./work/unsup.neg \
	--epohs=1 --alpha=0.01  \
	--prefile=./work/glove100.pretrained --output=./work/large_models/models_dims/ --loss=tanh

clean:
	rm -f train ../work/wreps.txt 

batch:
	./train --dim=$(DIM) --wpid_fname=./work/benchmark.wpids --patid_fname=./work/benchmark.patids \
	--matrix_fname=./work/benchmark.ppmi --pos=./work/unsup.pos --neg=./work/unsup.neg \
	--epohs=1 --alpha=0.01  \
	--prefile=./work/glove$(DIM).pretrained --output=./work/models/$(DIM) --loss=tanh

# g++ -Ofast -march=native -Wall -fopenmp -std=c++0x -I eigen train.cc -o train