#! /usr/bin/python -u
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
        self.vocab = None 
        self.vects = None 
        self.vector_size = None
        pass


    def read_model(self, fname, dim, HEADER=False):
        """
        Read the word vectors where the first token is the word.
        """
        res = {}
        F = open(fname)
        if HEADER:
            res["method"] = F.readline().split('=')[1].strip()
            res["input"] = F.readline().split('=')[1].strip()
            res["rank"] = int(F.readline().split('=')[1])
            res["itermax"] = int(F.readline().split('=')[1])
            res["vertices"] = int(F.readline().split('=')[1])
            res["edges"] = int(F.readline().split('=')[1])
            res["labels"] = int(F.readline().split('=')[1])
            R = res["rank"]
        R = dim
        # read the vectors.
        vects = {}
        vocab = []
        line = F.readline()
        while len(line) != 0:
            p = line.split()
            word = p[0]
            v = numpy.zeros(R, float)
            for i in range(0, R):
                v[i] = float(p[i+1])
            vects[word] = normalize(v)
            vocab.append(word)
            line = F.readline()
        F.close()
        self.vocab = vocab
        self.vects = vects
        self.vector_size = R
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
        vects = {}
        print "Vocabulary size =", vocab_size
        print "Vector size =", vector_size
        binary_len = numpy.dtype(numpy.float32).itemsize * vector_size
        for line_number in xrange(vocab_size):
            # mixed text and binary: read text first, then binary
            word = ''
            while True:
                ch = F.read(1)
                if ch == ' ':
                    break
                if ch != '\n':
                    word += ch
            word = word.lower()
            vocab.append(word)
            vector = numpy.fromstring(F.read(binary_len), numpy.float32)
            # If you do not want to normalize the vectors, then do not call the normalize function.
            vects[word] = normalize(vector)            
        F.close()
        self.vocab = vocab
        self.vects = vects
        self.vector_size = vector_size
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
        x = B - A + C
        print cosine(x, D)
        pass   


def cosine(x, y):
    """
    Compute the cosine similarity between two vectors x and y. 
    """
    return numpy.dot(x,y.T)


def normalize(x):
    """
    L2 normalize vector x. 
    """
    norm_x = numpy.linalg.norm(x)
    return x if norm_x == 0 else (x / norm_x)


def eval_SAT_Analogies(vects, method):
    """
    Solve SAT word analogy questions using the vectors. 
    """
    from sat import SAT
    S = SAT()
    questions = S.getQuestions()
    corrects = total = skipped = 0
    for Q in questions:
        total += 1
        (q_first, q_second) = Q['QUESTION']
        if q_first['word'] in vects and q_second['word'] in vects:
            va = vects[q_first['word']]
            vb = vects[q_second['word']]
            max_sim = -100
            max_cand = -100
            for (i, (c_first, c_second)) in enumerate(Q["CHOICES"]):
                sim = 0
                if c_first['word'] in vects and c_second['word'] in vects:
                    vc = vects[c_first['word']]
                    vd = vects[c_second['word']]
                    sim = scoring_formula(va, vb, vc, vd, method)
                    #print q_first['word'], q_second['word'], c_first['word'], c_second['word'], sim
                    #sim = numpy.random.random()
                    if max_sim < sim:
                        max_sim = sim 
                        max_cand = i
            if max_cand == Q['ANS']:
                corrects += 1
                #print "CORRECT:", q_first['word'], q_second['word'], c_first['word'], c_second['word'], sim
        else:
            skipped += 1
    acc = float(100 * corrects) / float(total)
    coverage = 100.0 - (float(100 * skipped) / float(total))
    print "SAT Accuracy = %f (%d / %d)" % (acc, corrects, total)
    print "Qustion coverage = %f (skipped = %d)" % (coverage, skipped) 
    return {"acc":acc, "coverage":coverage}


def eval_Google_Analogies(vects, res_fname, method):
    """
    Evaluate the accuracy of the learnt vectors on the analogy task. 
    We consider the set of fourth words in the test dataset as the
    candidate space for the correct answer.
    """
    res_file = open(res_fname, 'w')
    analogy_file = open("../data/benchmarks/analogy_pairs.txt")
    cands = []
    questions = collections.OrderedDict()
    total_questions = {}
    corrects = {}
    while 1:
        line = analogy_file.readline().lower()
        if len(line) == 0:
            break
        if line.startswith(':'):  # This is a label 
            label = line.split(':')[1].strip()
            questions[label] = []
            total_questions[label] = 0
            corrects[label] = 0
        else:
            p = line.strip().split()
            total_questions[label] += 1
            if (p[0] in vects) and (p[1] in vects) and (p[2] in vects) and (p[3] in vects):
                questions[label].append((p[0], p[1], p[2], p[3]))
            if (p[3] in vects) and (p[3] not in cands):
                cands.append(p[3])
    analogy_file.close()
    valid_questions = sum([len(questions[label]) for label in questions])
    print "Total no. of question types =", len(questions) 
    print "Total no. of candidates =", len(cands)
    print "Valid questions =", valid_questions
    
    # predict the fourth word for each question.
    count = 1
    for label in questions:
        for (a,b,c,d) in questions[label]:
            if count % 100 == 0:
                print "%d%% (%d / %d)" % ((100 * count) / float(valid_questions), count, valid_questions), "\r", 
            count += 1
            # set of candidates for the current question are the fourth
            # words in all questions, except the three words for the current question.
            scores = []
            va = vects[a]
            vb = vects[b]
            vc = vects[c]
            for cand in cands:
                if cand not in [a,b,c]:
                    y = vects[cand]
                    scores.append((cand, scoring_formula(va, vb, vc, y, method)))

            scores.sort(lambda p, q: -1 if p[1] > q[1] else 1)
            if scores[0][0] == d:
                corrects[label] += 1
    
    # Compute accuracy
    n = semantic_total = syntactic_total = semantic_corrects = syntactic_corrects = 0
    for label in total_questions:
        n += total_questions[label]
        if label.startswith("gram"):
            syntactic_total += total_questions[label]
            syntactic_corrects += corrects[label]
        else:
            semantic_total += total_questions[label]
            semantic_corrects += corrects[label]
    print "Percentage of questions attempted = %f (%d / %d)" % ((100 * valid_questions) /float(n), valid_questions, n)

    res_file.write("Dataset, Accuracy, Corrects, Attempted, Total\n")
    for label in questions:
        acc = float(100 * corrects[label]) / float(total_questions[label])
        res_file.write("%s, %f, %d, %d, %d\n" % (label, acc, corrects[label], len(questions[label]), total_questions[label]))
        print "%s = %f (correct = %d, attempted = %d, total = %d)" % (
            label, acc, corrects[label], len(questions[label]), total_questions[label])
    semantic_accuracy = float(100 * semantic_corrects) / float(semantic_total)
    syntactic_accuracy = float(100 * syntactic_corrects) / float(syntactic_total)
    total_corrects = semantic_corrects + syntactic_corrects
    accuracy = float(100 * total_corrects) / float(n)
    print "Semantic Accuracy =", semantic_accuracy 
    res_file.write("Semantic, %f, %d, %d\n" % (semantic_accuracy, semantic_corrects, semantic_total))
    print "Syntactic Accuracy =", syntactic_accuracy
    res_file.write("Syntactic, %f, %d, %d\n" % (syntactic_accuracy, syntactic_corrects, syntactic_total))
    print "Total accuracy =", accuracy
    res_file.write("Total, %f, %d, %d\n" % (accuracy, total_corrects, n))
    res_file.close()
    return {"semantic": semantic_accuracy, "syntactic":syntactic_accuracy, "total":accuracy}


def eval_word2vec(method):
    """
    Evaluate the performance of the word2vec vectors.
    """
    w2v = WordReps()
    model = "../data/word-vects/w2v.neg.300d.bin"
    #model = "../data/word-vects/skip-100.bin"
    print "Model file name =", model
    w2v.read_w2v_model(model)
    #eval_Google_Analogies(w2v.vects, "../work/Google.csv", method)
    eval_SAT_Analogies(w2v.vects, method)
    pass


def eval_glove(method):
    """
    Evaluate the performance of the models trained by Glove. 
    """
    glove = WordReps()
    model = "../data/word-vects/glove.42B.300d.txt"
    dim = 300
    print "Model file name =", model
    glove.read_model(model, dim)
    eval_Google_Analogies(glove.vects, "../work/Google.csv", method)
    eval_SAT_Analogies(glove.vects, method)
    pass


############### SCORING FORMULAS ###################################################
def scoring_formula(va, vb, vc, vd, method):
    """
    Call different scoring formulas. 
    """
    if method == "subt_cos":
        return subt_cos(va, vb, vc, vd)
    elif method == "PairDirection":
        return PairDirection(va, vb, vc, vd)
    elif method == "mult_cos":
        return mult_cos(va, vb, vc, vd)
    elif method == "add_cos":
        return add_cos(va, vb, vc, vd)
    elif method == "domain_funct":
        return domain_funct(va, vb, vc, vd)
    else:
        raise ValueError


def mult_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    log(cos(vb, vd)) + log(cos(vc,vd)) - log(cos(va,vd))
    """
    first = (1.0 + cosine(vb, vd)) / 2.0
    second = (1.0 + cosine(vc, vd)) / 2.0
    third = (1.0 + cosine(va, vd)) / 2.0
    score = numpy.log(first) + numpy.log(second) - numpy.log(third)
    return score


def add_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(vb - va + vc, vd)
    """
    x = normalize(vb - va + vc)
    return cosine(x, vd)


def domain_funct(va, vb, vc, vd):
    """
    Uses the Formula proposed by Turney in Domain and Function paper.
    """
    return numpy.sqrt((1.0 + cosine(va, vc))/2.0 * (1.0 + cosine(vb, vd))/2.0)


def subt_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(va - vc, vb - vd)
    """
    return cosine(normalize(va - vc), normalize(vb - vd))


def PairDirection(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(vd - vc, vb - va)
    """
    return cosine(normalize(vd - vc), normalize(vb - va))
####################################################################################

def process():
    method = "add_cos"
    #eval_word2vec(method)
    eval_glove(method)
    pass


if __name__ == "__main__":
    process()