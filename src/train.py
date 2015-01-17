#! /usr/bin/python -u

"""
Implements IJCAI 2015 Learning Word Representations for Proportional Analogies

Danushka Bollegala 
15th Jan 2015
"""

from cooc import load_cooc
from collections import defaultdict
import numpy
import random
import sys


class RepLearn:

    def __init__(self, D):
        self.D = D  # Dimensionality of the representations
        pass        

    def load_data(self, wpid_fname, patid_fname, matrix_fname, pos_fname, neg_fname):
        """
        Load all related files. 
        """
        self.wpids, self.patids, self.M, self.M_rows = load_cooc(matrix_fname, wpid_fname, patid_fname)
        self.pos_pairs = self.load_train_instances(pos_fname)
        self.neg_pairs = self.load_train_instances(neg_fname)
        print "Positive training pattern pairs =", len(self.pos_pairs)
        print "Negative training pattern pairs =", len(self.neg_pairs)
        self.vocab = defaultdict(lambda: len(self.vocab))
        for (first, second) in self.wpids.values():
            self.vocab[first]
            self.vocab[second]
        print "Total number of unique words =", len(self.vocab)
        print "Total number of unique patterns =", len(self.patids)

        # initialize word representations        
        self.random_initialization()

        # Index patterns and the word-pairs in which those patterns occur.
        self.R = {}  # pattern sets. List of tuples (u,v,f(p,u,v)).
        self.R_total = defaultdict(float)  # |R(p)|
        self.p = {}  # pattern representations
        self.H = {}  # index from patterns to words. H[p][x] contains the value H(p,x).

        n = len(self.patids)
        for j in range(0, len(self.patids)):
            self.R[j] = []
            self.H[j] = defaultdict(float)
            p = self.M[:,j]
            nnz = p.nonzero()[0]
            p = p.toarray().T[0]
            self.p[j] = numpy.zeros(self.D, dtype=numpy.float64)
            print "\r%d of %d %d" % (j, n, len(nnz)),
            for i in nnz:
                val = p[i]
                (first, second) = self.wpids[i]
                u = self.vocab[first]
                v = self.vocab[second]
                self.R[j].append((u, v, val))
                self.R_total[j] += val
                self.p[j] += self.x[u] - self.x[v]
                self.H[j][u] += val 
                self.H[j][v] -= val
                pass
        pass

    def load_train_instances(self, fname):
        """
        Load training pattern pairs. 
        """
        L = []
        with open(fname) as F:
            for line in F:
                p = line.strip().split()
                L.append((int(p[2]), int(p[3]))) 
        return L 

    def update_pattern_reps(self):
        """
        Update the pattern representations
        """
        for p in self.R:
            self.p[p] = numpy.zeros(self.D, dtype=numpy.float64)
            for (u, v, val) in self.R[p]:
                self.p[p] += self.x[u] - self.x[v]
        pass

    def random_initialization(self):
        """
        Initialize word representations randomly
        uniform(-sigma*sqrt(3), sigma*sqrt(3))
        sigma = 1/sqrt(D)
        """
        a = numpy.sqrt(3.0) / numpy.sqrt(self.D)
        self.x = {}
        for w in self.vocab:
            wid = self.vocab[w]
            self.x[wid] = numpy.random.uniform(low=-a, high=a, size=self.D)
        pass

    def train(self, epohs=10):
        """
        Perform training
        """
        # Randomly shuffle positive and negative training instances
        data = [(p1, p2, 1) for (p1, p2) in self.pos_pairs]
        data.extend([(p1, p2, -1) for (p1, p2) in self.neg_pairs])
        random.shuffle(data)
        alpha = 1.7159
        beta = 2.0 / 3.0
        N = len(data)
        s_grad = numpy.zeros(self.D, dtype=numpy.float64)  # AdaGrad
        uvec = numpy.ones(self.D, dtype=numpy.float64)  # one vector

        for epoh in range(epohs):
            loss = 0
            loss_grad_norm = 0
            for (i, (p1, p2, t)) in enumerate(data):
                print "\rEpoh: %d (%d of %d) %f Complete" % (epoh, i, N, float(100 * i) / float(N)),
                sys.stdout.flush()
                # Compute the part of the loss that depends on p1, p2
                score = numpy.dot(self.p[p1], self.p[p2])
                y = numpy.tanh(beta * score)
                y_prime = alpha * beta * (1 - y ** 2)
                y = alpha * y 
                loss += (y - t) ** 2
                l = y_prime * (y - t)
                # get the set of words involved
                cand_words = set(self.H[p1].keys()).union(set(self.H[p2].keys()))

                # update the word representations
                for w in cand_words:
                    g = l * ((self.H[p1][w] / self.R_total[p1]) * self.p[p2] - (self.H[p2][w] / self.R_total[p2]) * self.p[p1])
                    loss_grad_norm += numpy.linalg.norm(g)
                    s_grad = g * g
                    self.x[w] -= (1.0 / numpy.sqrt(s_grad + uvec)) * g

            self.update_pattern_reps()      
            print "\n Loss = %f, |d(Loss)| = %f" % (numpy.sqrt(loss) / len(data)), loss_grad_norm / len(data)
            sys.stdout.flush()
        pass


def initialize_model():
    D = 10
    wpid_fname = "../work/benchmark.wpids"
    patid_fname = "../work/benchmark.patids"
    matrix_fname = "../work/benchmark.ppmi"
    pos_fname = "../work/small.pos"
    neg_fname = "../work/small.neg"
    RL = RepLearn(D)
    RL.load_data(wpid_fname, patid_fname, matrix_fname, pos_fname, neg_fname)
    return RL


def process(RE):
    RE.train(2)
    pass

if __name__ == '__main__':
    RE = initialize_model()
    #process()
