#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <map>
#include <cmath>
#include <cstring>
#include <functional>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <fstream>

#include <Eigen/Dense>

#include "parse_args.hh"
#include "dictionary.hh"

#include "omp.h"

using namespace std;
using namespace Eigen;

struct edge{
    string u;
    string v;
    double weight;
};

vector<pair<string,string>> wpids; // word pair ids
vector<string> patids; // pattern ids
vector<string> vocab; // vocabulary
int D; // Dimensionality of the word vectors
unordered_map<string, VectorXd> x; // word representations
unordered_map<int, VectorXd> p; // pattern representations
unordered_map<int, vector<edge>> R; // Set R(p)
unordered_map<int, double> R_totals; // /R(p)


void load_data(string wpid_fname, string patid_fname){
    // Read word pairs ids
    ifstream wpid_file(wpid_fname.c_str());
    string first, second;
    int wpid;
    while (wpid_file >> wpid >> first >> second){
        pair<string, string> word_pair;
        word_pair.first = first;
        word_pair.second = second;
        wpids.insert(wpids.begin() + wpid, word_pair);
        if (find(vocab.begin(), vocab.end(), first) == vocab.end()){
            vocab.push_back(first);
        }
        if (find(vocab.begin(), vocab.end(), second) == vocab.end()){
            vocab.push_back(second);
        }
    }
    wpid_file.close();
    fprintf(stderr, "No. of unique words = %d\n", (int) vocab.size());
    fprintf(stderr, "No. of word pairs = %d\n", (int) wpids.size());

    // Reading pattern ids
    ifstream patid_file(patid_fname.c_str());
    string pattern;
    int patid;
    while (patid_file >> patid >> pattern){
        pair<string, string> word_pair;
        patids.insert(patids.begin() + patid, pattern);
    }
    patid_file.close();
    fprintf(stderr, "No. of patterns = %d\n", (int) patids.size());
}


void initialize_random_word_vectors(){
    // Randomly initialize word vectors
    fprintf(stderr, "Dimensionality of the word vectors = %d\n", D);
    double factor = sqrt(3) / sqrt(D);
    for (int i = 0; i < (int) vocab.size(); i++){
        x[vocab[i]] = factor * VectorXd::Random(D);
    }
    
}


void compute_pattern_reps(list<int> selected_patids){
    // Use word representations to compute pattern representations
    for (int i = 0; i < (int) selected_patids.size(); i++){
        patid = selected_patids[i];
        p[patid] = VectorXd::Zero(D);
        fprintf(stderr, "\rPattern %d of %d completed.", patid, (int) patids.size());
        R_totals[patid] = 0;
        for (int j = 0; j < (int) R[patid].size(); j++){
            edge e = R[patid][j];
            p[patid] += x[e.u] - x[e.v];
            R_totals[patid] += e.weight;
        }
    }
}

void load_matrix(string matrix_fname){
    FILE *fp= fopen(matrix_fname.c_str(), "r");
    int count = 0;
    for (char buf[65536]; fgets(buf, sizeof(buf), fp); ) {
        int wpid;
        count += 1;
        if (count % 1000 == 0)
            fprintf(stderr, "\r%d Completed", count);
        sscanf(buf, "%d %[^\n]", &wpid, buf);
    while (1) {
        int c;
        double w;
        bool end = (sscanf(buf, "%d:%lf %[^\n]", &c, &w, buf) != 3);
        edge e;
        e.u = wpids[wpid].first;
        e.v = wpids[wpid].second;
        e.weight = w;
        R[c].push_back(e);
        if (end) break;
        }
    }
    fclose(fp);
}



int main(int argc, char *argv[]) { 
  if (argc == 1) {
    fprintf(stderr, "usage: ./train --dim=dimensionality --wpid_fname=wpids --patid_fname=patids --matrix_fname=matrix --pos=positives.txt --neg=negatives.txt >\n"); 
    return 0;
  }
  int no_threads = 10;
  omp_set_num_threads(no_threads);
  setNbThreads(no_threads);
  initParallel(); 
  parse_args::init(argc, argv); 
  load_data(parse_args::get<string>("--wpid_fname"), parse_args::get<string>("--patid_fname"));
  D = parse_args::get<int>("--dim");
  initialize_random_word_vectors();
  load_matrix(parse_args::get<string>("--matrix_fname"));
  # initial computation of pattern representations.
  list<int> selected_patids;
  for (int i = 0; i < (int) patids.size(); i++)
    selected_patids.push_back(i);
  compute_pattern_reps(selected_patids);
}