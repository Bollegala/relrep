# README #

# About #
This module implements the attribute factorisation method that considers semantic relations between words when learning representations for words. See the paper (under review) for the details of the method.

# Organisation of the source code #
## ./preprocess ##
This directory contains source code (./src) required to preprocess data.

### ukWaCParser.py ###
This script reads the MaltParser parsed ukWaC corpus and provides a wrapper class can be used to easily read one sentence at a time from this corpus. It avoids the trouble of parsing the ukWaC corpus manually.

### genPairs.py ###
This script can be used to select sentences from the ukWaC corpus that contain at least one (or both) words in word pairs in the train/test datasets.
It can also be used to merge or convert various benchmark datasets into a format that can be processed by the attrrfac module.

### filter.py ###
This script can be used to count the frequency of patterns extracted for all the train word pairs from the corpus. It can be used to select top frequent patterns.

## ./dataset ##
The train subdirectory contains word pairs that are used for training.
The test subdirectory contains benchmark word pairs.

## ./analyze ##
The ./src subdirectory contains various scrips that be used for evaluation purposes.
### coupledBaseline.py ###
Implements the coupled baseline where we represent a word u by the combination of patterns l and other words v that co-occur with u.

### Label_Baseline.py ###
This is the SVD+LEX/POS/DEP methods described in the AAAI 2015 paper. It uses only the labels to represent a word.

### eval.py ###
All evaluations are done by this script.

### read_vects.py ###
This program reads the binary format produced by the word2vec.

There are several .pyx files (cython files) that implements the ALS optimisation method. But these are not used in the experiments in the paper as they were very slow. We use the Eigen library-based C++ implementations for this purpose.

## ./src ##
This directory contains the c++ files that implements the proposed method.

### diagonal.cc ###
Diagonalised matrix version of G(l)

### full.cc ###
G(l) is set to a full RxR matrix where R is the dimensionality of the word representation.

### diagonal_init.cc ###
This is a version of diagonal.cc where we initialise the word representations to the label only representations for them.
This did not perform well during the preliminary experiments and was not perused further.


# Steps involved and scripts to use #
1. Select words that occur more than a certain number of times in ukWaC (genPairs.py)
2. Select word-pairs by pairing two words u and v selected from (1) and compute their total occurrence in ukWaC. (genPairs.py)
3. Extract patterns for the word pairs selected in (2) (ukWaCParser.py).
4. Create co-occurrence matrices between word-pairs and patterns (ukWaCParser.py implements several pattern weighting methods such as RAW co-occurrences, PPMI, LMI, logarithm of the co-occurrences (LOG), and the entropy-based pattern weighting method (ENT) proposed in Turney 2006.
5. Run diagonal or full methods (C++ executables) on the created matrices in step 4.
6. Run analyze/eval.py to evaluate the created word representations.

All the very best!