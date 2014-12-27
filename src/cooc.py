#!/usr/bin/python -u
"""
This tool can be used to perform following tasks.
    1. Count the total occurrences of words in a corpus and write the sorted counts to a file.
    2. Find the co-occurring word pairs where each word is in the list of words (vocabulary) selected in (2).
    3. Find lexical patterns for the word pairs in (3) and get their counts.
    4. Select top occurring lexical patterns (patterns) from (5) and assign ids.
    5. Generate co-occurrence matrix between co-occrring pairs and patterns. 

The program can be run as a pipeline using intermediate files generated from previous stages.
"""

__author__ = "Danushka Bollegala"
__licence__ = "BSD"
__version__ = "1.0"

from collections import defaultdict, OrderedDict
import string
import scipy.sparse 
import numpy
import svmlight_loader


class COOC:

    def __init__(self):
        """
        Various parameters
        """
        self.VOCAB_MIN_COUNT = 5  # we ignore words that occur less than this in the corpus.
        self.MIN_WORD_LENGTH = 2  # a word must have at least this much of characters to be considered.
        self.WINDOW_SIZE = 5  # consider co-occurrences within this number of tokens between two words.
        self.MIN_COOC_COUNT = 50  # do not consider word-pairs that co-occur less than this value.
        self.MIN_PATTERN_COUNT = 10  # do not consider patterns less than this
        self.N = 11159025  # no. of sentences in the corpus.
        self.stop_words = self.load_stop_words("../data/stopWords.txt")
        pass


    def load_stop_words(self, fname):
        """
        Load the stop words from the specified file. A word per line. 
        """
        with open(fname) as F:
            return set(map(string.strip, F.readlines()))

    def get_vocabulary(self, corpus_fname, vocab_fname):
        """
        Count the total occurrences of words in a corpus and write the sorted counts to a file.
        """
        total_tokens = 0
        total_sentences = 0
        corpus_file = open(corpus_fname)
        h = defaultdict(int)
        for line in corpus_file:
            total_sentences += 1
            if total_sentences % 1000 == 0:
                print "\rCompleted = %d (%f)" % (total_sentences, (100 * float(total_sentences)) / float(self.N)),
            for w in line.lower().split():
                if (len(w) >= self.MIN_WORD_LENGTH) and (w not in self.stop_words):
                    h[w] += 1
                    total_tokens += 1
        corpus_file.close()
        # remove words that occur less than VOCAB_MIN_COUNT and sort
        selwords = []
        for (w, freq) in h.iteritems():
            if freq >= self.VOCAB_MIN_COUNT:
                selwords.append((w, freq))
        selwords.sort(lambda x, y: -1 if x[1] > y[1] else 1)
        vocab_file = open(vocab_fname, 'w')
        for (w, freq) in selwords:
            vocab_file.write("%s\t%d\n" % (w, freq))
        vocab_file.close()
        print "Total tokens in the corpus    =", total_tokens 
        print "Total sentences in the corpus =", total_sentences
        print "Size of the vocabulary        =", len(selwords)
        pass


    def get_coocurrences(self, corpus_fname, vocab_fname, cooc_pairs_fname):
        """
        Find the co-occurring word pairs where each word is in the vocabulary.
        """
        vocab = set()
        vocab_file = open(vocab_fname)
        for line in vocab_file:
            vocab.add(line.strip().split('\t')[0])
        vocab_file.close()
        print "Size of the vocabulary =", len(vocab)
        if "in" in vocab:
            print "Found"
        corpus_file = open(corpus_fname)
        pairs = defaultdict(int)
        total_sentences = 0
        for line in corpus_file:
            total_sentences += 1
            if total_sentences % 1000 == 0:
                print "\rCompleted = %d (%f)" % (total_sentences, (100 * float(total_sentences)) / float(self.N)),
            L = line.lower().strip().split()
            for i in range(0, len(L)):
                for j in range(i + 1, len(L)):
                    if (L[i] in vocab) and (L[j] in vocab) and ((j - i - 1) <= self.WINDOW_SIZE) and (L[i] != L[j]):
                        pairs[(L[i], L[j])] += 1
        corpus_file.close()
        sel_pairs = []
        for (pair, freq) in pairs.iteritems():
            if freq >= self.MIN_COOC_COUNT:
                sel_pairs.append((pair, freq))
        sel_pairs.sort(lambda x, y: -1 if x[1] > y[1] else 1)
        print "Total no. of pairs before selection =", len(pairs)
        print "Total no. of selected pairs         =", len(sel_pairs)
        pairs_file = open(cooc_pairs_fname, 'w')
        for ((first, second), freq) in sel_pairs:
            pairs_file.write("%s\t%s\t%d\n" % (first, second, freq))
        pairs_file.close()


    def extract_patterns(self, s):
        """
        Extract unigrams and bigrams as patterns. 
        """
        pats = set(s)
        for i in range(0, len(s) - 1):
            pats.add("%s+%s" % (s[i], s[i+1]))
        return pats


    def get_patterns(self, corpus_fname, cooc_pairs_fname, patterns_fname):
        """
        Extract patterns for the word pairs and count their frequencies. 
        """
        pairs = set()
        pairs_file = open(cooc_pairs_fname)
        for line in pairs_file:
            p = line.strip().split('\t')
            pairs.add((p[0], p[1]))
        pairs_file.close()
        print "Total no. of word-pairs =", len(pairs)
        corpus_file = open(corpus_fname)
        total_sentences = 0
        patterns = defaultdict(int)
        for line in corpus_file:
            total_sentences += 1
            if total_sentences % 1000 == 0:
                print "\rCompleted = %d (%f)" % (total_sentences, (100 * float(total_sentences)) / float(self.N)),
            L = line.lower().strip().split()
            for i in range(0, len(L)):
                for j in range(i + 1, len(L)):
                    if (j - i - 1) <= self.WINDOW_SIZE and (L[i], L[j]) in pairs:
                        for pattern in self.extract_patterns(L[i+1:j]):
                            patterns[pattern] += 1
        corpus_file.close()
        print "Total no. of patterns before filtering =", len(patterns)
        pat_list = patterns.items()
        pat_list.sort(lambda x, y: -1 if x[1] > y[1] else 1)
        count = 0
        patterns_file = open(patterns_fname, 'w')
        for (pat, freq) in pat_list:
            if freq >= self.MIN_PATTERN_COUNT:
                patterns_file.write("%s\t%d\n" % (pat, freq))
                count += 1
            else:
                break
        print "Total no. of patterns after filtering =", count
        patterns_file.close()
        pass


    def create_matrix(self, corpus_fname, pairs_fname, patterns_fname, prefix, debug=False):
        """
        We will read the selected word pairs from the pairs_fname and selected patterns from the 
        patterns_fname. We will then find the co-occurrences between word pairs and patterns 
        in the corpus_fname. We will create three files as follows:
        prefix.mat = co-occurrence matrix in the libsvm format 
        prefix.patids = pattern ids (pattern\tID)
        prefix.wpids = word pair ids (firstWord\tsecondWord\tID)
        """
        # Load word pairs.
        wpairs = []
        with open(pairs_fname) as wpairs_file:
            for line in wpairs_file:
                p = map(string.strip, line.strip().split('\t'))
                wpair = (p[0], p[1])
                #assert(wpair not in wpairs)
                wpairs.append(wpair)   
        wpairs_h = dict([(wpair, i) for (i, wpair) in enumerate(wpairs)])             
        print "Total no. of word pairs =", len(wpairs)

        # Load patterns. 
        patterns = []
        with open(patterns_fname) as patterns_file:
            for line in patterns_file:
                p =  map(string.strip, line.strip().split('\t'))
                #assert(p[0] not in patterns)
                patterns.append(p[0])        
        patterns_h = dict([(pattern, i) for (i, pattern) in enumerate(patterns)])        
        print "Total no. of patterns =", len(patterns)

        # Create matrix.
        M = scipy.sparse.lil_matrix((len(wpairs), len(patterns)), dtype=int)
        with open(corpus_fname) as corpus_file:
            total_sentences = 0
            for line in corpus_file:
                total_sentences += 1
                if total_sentences % 1000 == 0:
                    print "\rCompleted = %d (%f)" % (total_sentences, (100 * float(total_sentences)) / float(self.N)),
                L = line.lower().strip().split()
                for i in range(0, len(L)):
                    for j in range(i + 1, len(L)):
                        wpair = (L[i], L[j])
                        if (wpair in wpairs_h) and ((j - i - 1) <= self.WINDOW_SIZE):
                            for pattern in self.extract_patterns(L[i+1:j]):
                                if pattern in patterns_h:
                                    M[wpairs_h[wpair], patterns_h[pattern]] += 1

        # Delete zero rows and columns from the co-occurrence matrix.
        nnz_rows, nnz_cols = map(numpy.unique, M.nonzero())
        wpairs = [wpairs[i] for i in nnz_rows]
        patterns = [patterns[i] for i in nnz_cols]
        M = M[nnz_rows, :][:, nnz_cols]

        # Write the matrix.
        svmlight_loader.dump_svmlight_file(scipy.sparse.csr_matrix(M), numpy.arange(len(wpairs)), "%s.mat" % prefix, zero_based=True)

        # Write the pattern ids.
        print "Total no. of word pairs remaining =", len(wpairs)
        with open("%s.patids" % prefix, 'w') as patid_file:
            for (patid, pattern) in enumerate(patterns):
                patid_file.write("%d\t%s\n" % (patid, pattern))

        # Write word-pair ids.
        print "Total no. of patterns remaining =", len(patterns)
        with open("%s.wpids" % prefix, 'w') as wpid_file:
            for (wpid, (first, second)) in enumerate(wpairs):
                wpid_file.write("%d\t%s\t%s\n" % (wpid, first, second))

        # Write actual words and patterns for debugging purposes.
        if debug:
            with open("%s.debug" % prefix, 'w') as F:
                print "Writing debug matrix"
                for (wpid, wpair) in enumerate(wpairs):
                    F.write("%s %s " % wpair)
                    for patid in range(M.shape[1]):
                        if M[wpid, patid] != 0:
                            pattern = patterns[patid]
                            F.write("%s:%d " % (pattern, M[wpid, patid]))
                    F.write("\n")
        pass
    pass



def conv_corpus():
    """
    Convert all words into lower case. This is what is being used by word2vec 
    """
    import string
    table = string.maketrans("", "")
    input_file = open("../work/ukwac.corp.mixed-case")
    output_file = open("../work/ukwac.corp", "w")
    for line in input_file:
        output_file.write(line.lower().translate(table, string.punctuation))
    input_file.close()
    output_file.close()
    pass


def process():
    """
    Call each stage of the pipeline
    """
    corpus_fname = "../work/ukwac.corp"

    #vocab_fname = "../work/ukwac.vocab"
    #cooc_pairs_fname = "../work/ukwac.cooc_pairs"

    vocab_fname = "../work/benchmark-vocabulary.txt"
    cooc_pairs_fname = "../work/benchmark_cooc_pairs"
    patterns_fname = "../work/benchmark_patterns.10000"
    prefix = "../work/benchmark"

    C = COOC()
    #C.get_vocabulary(corpus_fname, vocab_fname)
    #C.get_coocurrences(corpus_fname, vocab_fname, cooc_pairs_fname)
    #C.get_patterns(corpus_fname, cooc_pairs_fname, patterns_fname)
    C.create_matrix(corpus_fname, cooc_pairs_fname, patterns_fname, prefix, debug=False)
    pass


def get_benchmark_words():
    """
    Create a list of words in benchmarks. 
    """
    vocab = set()
    # get words in Google dataset.
    with open("../data/benchmarks/analogy_pairs.txt") as F:
        for line in F:
            if line.startswith(":"):
                continue
            for word in line.lower().split():
                vocab.add(word)
    # get words from SAT.
    from sat import SAT
    S = SAT()
    questions = S.getQuestions()
    for Q in questions:
        (q_first, q_second) = Q['QUESTION']
        vocab.add(q_first['word']) 
        vocab.add(q_second['word'])
        for (c_first, c_second) in Q["CHOICES"]:
            vocab.add(c_first['word'])
            vocab.add(c_second['word'])
    # get SemEval words. 
    from semeval import SemEval
    S = SemEval("../data/benchmarks/semeval")
    for Q in S.data:
        for (first, second) in Q["wpairs"]:
            vocab.add(first)
            vocab.add(second)
        for (first, second) in Q["paradigms"]:
            vocab.add(first)
            vocab.add(second)
    print "Total no. of words in the benchmark datasets =", len(vocab)
    with open("../work/benchmark-vocabulary.txt", 'w') as bench_file:
        for w in vocab:
            bench_file.write("%s\t1\n" % w)
    pass


if __name__ == "__main__":
    #conv_corpus()
    #get_benchmark_words()
    process()





