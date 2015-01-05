"""
The following code is obsolete. It selects train instances for word pairs.
"""


def compute_wpair_similarity():
    """
    Compute pairwise similarity between word-pairs
    """
    prefix = "../work/benchmark"
    matrix_fname = "%s.LLR" % prefix
    wpids, patids, M, labels = load_cooc(matrix_fname, "%s.wpids" % prefix, "%s.patids" % prefix)
    pairs = get_benchmark_pairs()
    wpids_h = dict([(wpair, wpid) for (wpid, wpair) in wpids.items()])
    selected_pairs = []
    for wpair in pairs:
        if wpair in wpids_h:
            selected_pairs.append(wpair)

    H = numpy.zeros((len(selected_pairs), len(patids)), dtype=numpy.float)    
    for (i, wpair) in enumerate(selected_pairs):
        wpid = wpids_h[wpair]
        row_no = numpy.where(labels == wpid)[0]
        assert(row_no == wpid)
        H[i, :] = M[row_no,:].todense()
    normalize(H)
    S = numpy.dot(H, H.T)
    L = []
    for i in range(0, S.shape[0]):
        for j in range(i+1, S.shape[0]):
            val = S[i,j]
            if val != 0:
                L.append((i, j, val))
    L.sort(lambda x, y: -1 if x[2] > y[2] else 1)
    with open("%s.sim_scores.LLR" % prefix, 'w') as sim_file:
        for (i, j, val) in L:
            (A, B) = selected_pairs[i]
            (C, D) = selected_pairs[j]
            sim_file.write("%s %s %d %s %s %d %f\n" % (A, B, wpids_h[(A,B)], C, D, wpids_h[(C,D)], val))
    pass


def select_train_data_by_total_patterns():
    """
    Select positive and negative training word pairs. 
    """
    prefix = "../work/benchmark"
    wpids, patids, M, labels = load_cooc(prefix)
    wp_counts = {}
    M = M.todense()
    for i in range(0, M.shape[0]):
        print "\r%d" % i,
        wpid = labels[i]
        wp_counts[wpids[wpid]] = numpy.count_nonzero(M[i,:])
    L = wp_counts.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    with open("../work/counts", 'w') as F:
        for ((first, second), count) in L:
            F.write("%s\t%s\t%d\n" % (first, second, count))
    pass


def select_train_data():
    """
    Select Google train pairs as positive instances. Select top similar word pairs 
    according to relational similarity that do not appear in the positive train 
    instances as the negative train instances.
    """
    prefix = "../work/benchmark"
    positives = set()
    # get pairs in Google dataset.
    with open("../data/benchmarks/analogy_pairs.txt") as F:
        for line in F:
            if line.startswith(":"):
                continue
            (A, B, C, D) = line.lower().split()
            positives.add((A, B, C, D))
            positives.add((B, A, D, C))
            positives.add((A, C, B, D))
            positives.add((C, A, D, B))
    positives = list(positives)
    print "Generated positive instances =", len(positives)

    # Load similarity scores
    sim_fname = "%s.sim_scores" % prefix
    scores = []
    with open(sim_fname) as sim_file:
        for line in sim_file:
            p = line.strip().split()
            A, B, C, D = map(string.strip, (p[0], p[1], p[3], p[4]))
            if A in (C, D) or B in (C, D):
                continue
            if (A, B) == (B, A):
                continue
            scores.append((A, B, C, D))

    # Load word pair ids.
    matrix_fname = "%s.ppmi" % prefix
    wpids, patids, M, labels = load_cooc(matrix_fname, "%s.wpids" % prefix, "%s.patids" % prefix)

    # Select negative training instances
    negatives = set()
    n = 0
    for (A, B, C, D) in scores:
        if (A, B, C, D) not in positives:
            negatives.add((A, B, C, D))
            n += 1
            if n == 10000:
                break 
    negatives = list(negatives)

    # write the training instances to files
    wpids_h = dict([(wpair, wpid) for (wpid, wpair) in wpids.items()])
    pos_count = neg_count = 0
    with open("%s.train.pos" % prefix, 'w') as pos_file:
        for (A, B, C, D) in positives:
            if (A, B) in wpids_h and (C, D) in wpids_h:
                pos_file.write("%s %s %d %s %s %d\n" % (A, B, wpids_h[(A, B)], C, D, wpids_h[(C, D)]))
                pos_count += 1
    with open("%s.train.neg" % prefix, 'w') as neg_file:
        for (A, B, C, D) in negatives:
            neg_file.write("%s %s %d %s %s %d\n" % (A, B, wpids_h[(A, B)], C, D, wpids_h[(C, D)]))
            neg_count += 1
    print "Total no. of positive instances =", pos_count
    print "Total no. of negative instances =", neg_count   
    pass