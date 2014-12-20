"""
Perform evaluations of the word representations using three analogy datasets:
Mikolov (Google + MSRA), SAT, and SemEval.

"""

__author__ = "Danushka Bollegala"
__licence__ = "BSD"
__version__ = "1.0"

import numpy
import sys
import collections


class WordReps:

    def __init__(self):
        self.fname = None 
        self.vocab = None 
        self.vectors = None 
        self.vocab_size = None 
        self.vector_size = None
        pass


    def read_w2v_model(self, fname):
        """
        Given a model file (fname) produced by word2vect, read the vocabulary list 
        and the vectors for each word. We will return a dictionary of the form
        h[word] = numpy.array of dimensionality.
        """
        F = open(fname, 'rb')
        header = F.readline()
        vocab_size, vector_size = map(int, header.split())
        vocab = []
        print "Vocabulary size =", vocab_size
        print "Vector size =", vector_size
        vectors = numpy.empty((vocab_size, vector_size), dtype=numpy.float)
        binary_len = numpy.dtype(numpy.float32).itemsize * vector_size
        for line_number in xrange(vocab_size):
            # mixed text and binary: read text first, then binary
            word = ''
            while True:
                ch = F.read(1)
                if ch == ' ':
                    break
                word += ch
            vocab.append(word)
            vector = numpy.fromstring(F.read(binary_len), numpy.float32)
            vectors[line_number] = vector
            F.read(1)  # newline
            #print line_number
        #vocab = numpy.array(vocab)
        F.close()
        self.vocab = vocab
        self.vectors = vectors
        self.vocab_size = vocab_size
        self.vector_size = vector_size
        print "actual vectors", len(vectors)
        pass


    def get_vect(self, word):
        if word not in self.vocab:
            return numpy.zeros(self.vector_size, float)
        wind = self.vocab.index(word)
        print wind
        return self.vectors[wind,:]


    def test_model(self):
        A = self.get_vect("man")
        B = self.get_vect("king")
        C = self.get_vect("woman")
        D = self.get_vect("queen")
        A = A / numpy.linalg.norm(A)
        B = B / numpy.linalg.norm(B)
        C = C / numpy.linalg.norm(C)
        D = D / numpy.linalg.norm(D)
        x = B - A + C
        print cosine(x, D)
        pass   


def cosine(x, y):
    """
    Compute the cosine similarity between two vectors x and y. 
    """
    val = numpy.dot(x,y.T)
    x_norm = numpy.linalg.norm(x)
    y_norm = numpy.linalg.norm(y)
    if val == 0:
        return 0
    else:
        val = float(val) / float(x_norm * y_norm)
        return val 
    pass


def normalize(y):
    """
    L2 normalize vector x. 
    """
    x = numpy.copy(y)
    norm_x = numpy.linalg.norm(x)
    if norm_x != 0:
        x = x / norm_x 
    return x


def mult_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    log(cos(vb, vd)) + log(cos(vc,vd)) - log(cos(va,vd))
    """
    first = (1.0 + cosine(vb, vd)) / 2.0
    second = (1.0 + cosine(vc, vd)) / 2.0
    third = (1.0 + cosine(va, vd)) / 2.0
    score = (first * second) / (third + 1e-5)
    return score


def add_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(vb - va + vc, vd)
    """
    x = vb - va + vc
    return cosine(x, vd)


def eval_Google_Analogies(vects, res_fname):
    """
    Evaluate the accuracy of the learnt vectors on the analogy task. 
    We consider the set of fourth words in the test dataset as the
    candidate space for the correct answer.
    """
    res_file = open(res_fname, 'w')
    analogy_file = open("../data/benchmarks/analogy_pairs.txt")
    cands = []
    questions = collections.OrderedDict()
    while 1:
        line = analogy_file.readline()
        if len(line) == 0:
            break
        if line.startswith(':'): #This is a label 
            label = line.split(':')[1].strip()
            questions[label] = []
        else:
            p = line.lower().strip().split()
            if (p[0] in vects) and (p[1] in vects) and (p[2] in vects) and (p[3] in vects):
                if p[3] not in cands:
                    cands.append(p[3])
                questions[label].append((p[0], p[1], p[2], p[3]))
    analogy_file.close()
    print "Total no. of question types =", len(questions) 
    # print the number of instances for each label type. 
    corrects = {}
    syntactic_total = semantic_total = n = 0
    for label in questions:
        corrects[label] = 0.0
        #print label, len(questions[label]) # show the total instances for a relation.
        n += len(questions[label])
        if label.startswith("gram"):
            syntactic_total += len(questions[label])
        else:
            semantic_total += len(questions[label])
    # predict the fourth word for each question.
    syntactic_corrects = 0
    semantic_corrects = 0
    total_corrects = 0
    for label in questions:
        for (a,b,c,d) in questions[label]:
            va = normalize(vects[a])
            vb = normalize(vects[b])
            vc = normalize(vects[c])
            # set of candidates for the current question are the fourth
            # words in all questions, except the three words for the current question.
            scores = []
            for cand in cands:
                if cand not in [a,b,c]:
                    vd = normalize(vects[d])
                    scores.append((cand, scoring_formula(va, vb, vc, vd)))
            scores.sort(lambda p, q: -1 if p[1] > q[1] else 1)
            #print scores[:5]
            if scores[0][0] == d:
                corrects[label] = corrects[label] + 1
                total_corrects += 1
                if label.startswith("gram"):
                    syntactic_corrects += 1
                else:
                    semantic_corrects += 1
    # Compute accuracy
    res_file.write("Dataset, Accuracy, Corrects, Total\n")
    for label in questions:
        val = corrects[label]
        if len(questions[label]) == 0:
            corrects[label] = 0
        else:
            corrects[label] = float(100 * corrects[label]) / float(len(questions[label]))
        res_file.write("%s, %f, %d, %d\n" % (label, corrects[label], val, len(questions[label])))
        print label, corrects[label]
    semantic_accuracy = float(100 * semantic_corrects) / float(semantic_total)
    syntactic_accuracy = float(100 * syntactic_corrects) / float(syntactic_total)
    accuracy = float(100 * total_corrects) / float(n)
    print "Semantic Accuracy =", semantic_accuracy 
    res_file.write("Semantic, %f, %d, %d\n" % (semantic_accuracy, semantic_corrects, semantic_total))
    print "Syntactic Accuracy =", syntactic_accuracy
    res_file.write("Syntactic, %f, %d, %d\n" % (syntactic_accuracy, syntactic_corrects, syntactic_total))
    print "Total accuracy =", accuracy
    res_file.write("Total, %f, %d, %d\n" % (accuracy, total_corrects, n))
    res_file.close()
    return {"semantic": semantic_accuracy, "syntactic":syntactic_accuracy, "total":accuracy}



def eval_word2vec():
    """
    Evaluate the performance of the word2vec vectors.
    """
    w2v = WordReps()
    model = "../data/word-vects/w2v.neg.300d.bin"
    #model = "../data/word-vects/skip_100.bin"
    w2v.read_w2v_model(model)
    vects = {}
    for i in range(0, w2v.vocab_size):
        vects[w2v.vocab[i]] = w2v.vectors[i,:]
    eval_Google_Analogies(vects, "../work/Google.csv")
    pass


######## CAHNGE THE SCORING FORMULA HERE #######################
def scoring_formula(self, va, vb, vc, vd):
    """
    Call different scoring formulas. 
    """
    return mult_cos(va, vb, vc, vd)
################################################################


def process():
    eval_word2vec()
    pass


if __name__ == "__main__":
    process()
