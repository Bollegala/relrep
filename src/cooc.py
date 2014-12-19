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

from collections import defaultdict

class COOC:

    def __init__(self):
        """
        Various parameters
        """
        self.VOCAB_MIN_COUNT = 5 # we ignore words that occur less than this in the corpus.
        self.MIN_WORD_LENGTH = 2 # a word must have at least this much of characters to be considered.
        self.WINDOW_SIZE = 5 # consider co-occurrences within this number of tokens between two words.
        self.MIN_COOC_COUNT = 5 # do not consider word-pairs that co-occur less than this value.
        self.MIN_PATTERN_COUNT = 10 # do not consider patterns less than this
        self.N = 11159025 # no. of sentences in the corpus.
        pass


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
                if len(w) >= self.MIN_WORD_LENGTH:
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
                    if (j - i - 1) <= self.WINDOW_SIZE:
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
                        for pattern in self.extract_patterns(L[i:j]):
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

    pass


def process():
    """
    Call each stage of the pipeline
    """
    corpus_fname = "../work/ukwac.corp"
    vocab_fname = "../work/ukwac.vocab"
    cooc_pairs_fname = "../work/ukwac.cooc_pairs"
    C = COOC()
    #C.get_vocabulary(corpus_fname, vocab_fname)
    C.get_coocurrences(corpus_fname, vocab_fname, cooc_pairs_fname)
    pass


if __name__ == "__main__":
    process()





