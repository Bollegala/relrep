#include <iostream>
#include <vector>
#include <list>
#include <set>
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
#include <stdio.h>

#include <Eigen/Dense>

#include "parse_args.hh"
#include "dictionary.hh"

#include "omp.h"

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

using namespace std;
using namespace Eigen;

struct edge{
    string u;
    string v;
    double weight;
};

struct instance{
    int p1;
    int p2;
    int label;
};

vector<pair<string,string>> wpids; // word pair ids
vector<string> patids; // pattern ids
vector<string> vocab; // vocabulary
int D; // Dimensionality of the word vectors
unordered_map<string, VectorXd> x; // word representations
unordered_map<string, VectorXd> s_grad; // Squared gradient for AdaGrad
unordered_map<int, VectorXd> p; // pattern representations
unordered_map<int, vector<edge>> R; // Set R(p)
unordered_map<int, double> R_totals; // /R(p)
vector<unordered_map<string, double>> H; // H[patid][word] is the differences of values
vector<instance> train_data; // training instances


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
    for (int i = 0; i < (int) vocab.size(); ++i){
        x[vocab[i]] = factor * VectorXd::Random(D);
    }    
}

void scale_word_vectors(){
    // scale the word vectors at the start of the training
    VectorXd mean = VectorXd::Zero(D);
    VectorXd squared_mean = VectorXd::Zero(D);
    for (auto w = x.begin(); w != x.end(); ++w){
        mean += w->second;
        squared_mean += (w->second).cwiseProduct(w->second);
    }
    mean = mean / ((double) x.size());
    VectorXd sd = squared_mean - mean.cwiseProduct(mean);
    for (int i = 0; i < D; ++i){
        sd[i] = sqrt(sd[i]);
    }
    for (auto w = x.begin(); w != x.end(); ++w){
        VectorXd tmp = VectorXd::Zero(D);
        for (int i = 0; i < D; ++i){
            tmp[i] = (w->second)[i] - mean[i];
            if (sd[i] != 0) 
                tmp[i] /= sd[i];
        }
        w->second = tmp;
    }
}

void load_word_vectors(string vects_fname){
    // Initialize word vectors using pre-trained word vectors
    fprintf(stderr, "%sDimensionality of the word vectors = %d%s\n", KRED, D, KNRM);
    fprintf(stderr, "%sReading pre-trained vectors from %s%s\n", KRED, vects_fname.c_str(), KNRM);
    FILE *fp= fopen(vects_fname.c_str(), "r");
    int count = 0;
    for (char buf[65536]; fgets(buf, sizeof(buf), fp); ) {
        char curword[1024];
        sscanf(buf, "%s %[^\n]", curword, buf);
        string w(curword);
        //fprintf(stderr, "%s\n", w.c_str());
        x[w] = VectorXd::Zero(D);
        count = 0;
        while (1) {
            double fval;
            bool end = (sscanf(buf, "%lf %[^\n]", &fval, buf) != 2);
            x[w][count] = fval;
            count += 1;
        if (end) break;
        }
        assert(count == D);
    }
}

void compute_pattern_reps(){
    // Use word representations to compute pattern representations
    VectorXd mean = VectorXd::Zero(D);
    for (size_t patid = 0; patid < patids.size(); ++patid){
        p[patid] = VectorXd::Zero(D);
        for (size_t j = 0; j < R[patid].size(); ++j){
            edge e = R[patid][j];
            p[patid] += (e.weight * (x[e.u] - x[e.v]));
        }
        p[patid] /= R_totals[patid];
        mean += p[patid];
    }

    // scaling
    mean = (1.0 / ((double) patids.size())) * mean;
    for (size_t patid = 0; patid < patids.size(); ++patid)
        p[patid] -= mean;
}

void initialize_pattern_reps(){
    // Use word representations to compute pattern representations
    H.resize(patids.size());
    for (int patid = 0; patid < (int) patids.size(); ++patid){
        p[patid] = VectorXd::Zero(D);
        fprintf(stderr, "\rPattern %d of %d completed.", patid, (int) patids.size());
        R_totals[patid] = 0;
        for (size_t j = 0; j < R[patid].size(); ++j){
            edge e = R[patid][j];
            p[patid] += (e.weight * (x[e.u] - x[e.v]));
            R_totals[patid] += e.weight;
            H[patid][e.u] += e.weight;
            H[patid][e.v] -= e.weight;
        }
        p[patid] /= R_totals[patid];
        p[patid] /= p[patid].norm();
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


void load_train_instances(string train_fname, int label){
    // Load and suffle positive and negative train instances

    ifstream train_file(train_fname.c_str());
    int p1, p2;
    double sim;
    string first, second;
    while (train_file >> first >> second >> p1 >> p2 >> sim){
        instance I;
        I.p1 = p1;
        I.p2 = p2;
        I.label = label;
        train_data.push_back(I);
        }
    train_file.close();
}

void save_word_reps(string fname){
    // write the word representations to a file
    ofstream reps_file;
    reps_file.open(fname);
    for (auto w = x.begin(); w != x.end(); ++w){
        reps_file << w->first << " ";
        for (int i = 0; i < D; ++i){
            reps_file << w->second[i] << " ";
        }
        reps_file << endl;
    }
    reps_file.close();
}

void train(int epohs, double init_alpha){
    // Training 
    fprintf(stderr, "\nTotal no of train instances = %d\n", (int) train_data.size());
    fprintf(stderr, "Total ephos to train = %d\n", epohs);
    fprintf(stderr, "Initial learning rate = %f\n", init_alpha);
    // Randomly shuffle train instances
    random_shuffle(train_data.begin(), train_data.end());
    int p1, p2, label, count, update_count;
    double loss, loss_grad_norm, weight_norm, score, y, y_prime, lmda, errors;
    const double alpha = 1.7159;
    const double beta = 2.0 / 3.0;

    // Initialize squared gradient counts for AdaGrad
    for (auto w = x.begin(); w != x.end(); ++w)
        s_grad[w->first] = VectorXd::Zero(D);

    for (int t = 0; t < epohs; ++t){
        loss = 0;
        loss_grad_norm = 0;
        weight_norm = 0;
        count = 0;
        update_count = 0;
        errors = 0;
        for (auto inst = train_data.begin(); inst != train_data.end(); ++inst){
            count += 1;
            p1 = inst->p1;
            p2 = inst->p2;
            label = inst->label;
            score = p[p1].adjoint() * p[p2];

            y = tanh(beta * score);
            y_prime = alpha * beta * (1 - (y * y));
            y = alpha * y;            

            loss += (y - label) * (y - label);

            if (label * score < 0)
                errors += 1.0;

            lmda = y_prime * (y - label);

            // get the candidate words
            set<string> cand_words;
            /*for (auto w = H[p1].begin(); w != H[p1].end(); ++w)
                cand_words.insert(w->first);
            for (auto w = H[p2].begin(); w != H[p2].end(); ++w)
                cand_words.insert(w->first);*/

            for (auto w = H[p1].begin(); w != H[p1].end(); ++w){
                if (H[p2].find(w->first) != H[p2].end())
                    cand_words.insert(w->first);
            }            

            // update word representations
            for (auto w = cand_words.begin(); w != cand_words.end(); ++w){
                VectorXd grad = lmda * (((H[p1][(*w)] / R_totals[p1]) * p[p2]) - ((H[p2][(*w)] / R_totals[p2]) * p[p1]));
                loss_grad_norm += grad.norm();
                update_count += 1;
                s_grad[(*w)] += grad.cwiseProduct(grad);
                weight_norm += x[(*w)].norm();
                for (int i = 0; i < D; ++i)
                    x[(*w)][i] -= (init_alpha * grad[i]) / sqrt(1.0 + s_grad[(*w)][i]);                  
            }

            if ((count % 1000) == 0){
                //scale_word_vectors();
                compute_pattern_reps();
            }

            fprintf(stderr, "\rEpoh = %d: instance = %d, Loss(MSRE) = %f, ||gradLoss|| = %E, ratio = %E, Err = %f, CandWords = %lu", 
                t, count, sqrt(loss / count),  
                (loss_grad_norm / update_count), (loss_grad_norm / weight_norm),  (errors / count), cand_words.size());
        }
        //compute_pattern_reps();
        fprintf(stderr, "%s\n Epoh: %d Loss(MSRE) = %f, ||gradLoss|| = %E%s\n", 
            KYEL, t, sqrt(loss / (double) count), loss_grad_norm / (double) update_count, KNRM);
    }
}


int main(int argc, char *argv[]) { 
  if (argc == 1) {
    fprintf(stderr, "usage: ./train --dim=dimensionality --wpid_fname=wpids --patid_fname=patids \
                            --matrix_fname=matrix --pos=positives.txt --neg=negatives.txt \
                            --ephos=rounds --alpha=initial_learning_rate\n"); 
    return 0;
  }
  int no_threads = 100;
  omp_set_num_threads(no_threads);
  setNbThreads(no_threads);
  initParallel(); 
  parse_args::init(argc, argv); 
  load_data(parse_args::get<string>("--wpid_fname"), parse_args::get<string>("--patid_fname"));
  D = parse_args::get<int>("--dim");

  bool rand_mode = parse_args::get<bool>("--random");
  if (rand_mode){
    // Random initialization
    initialize_random_word_vectors();
  }
  else{
    // Load from pre-trained file
    load_word_vectors(parse_args::get<string>("--prefile"));
  }
  scale_word_vectors();

  load_matrix(parse_args::get<string>("--matrix_fname"));
  initialize_pattern_reps();

  load_train_instances(parse_args::get<string>("--pos"), 1);
  load_train_instances(parse_args::get<string>("--neg"), -1);

  //train(parse_args::get<int>("--epohs"), parse_args::get<double>("--alpha"));
  save_word_reps(parse_args::get<string>("--output"));
  return 0;
}