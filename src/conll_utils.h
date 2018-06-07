#ifndef CONLL_UTILS_H
#define CONLL_UTILS_H

#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "utils.h"
#include "random_utils.h"

using std::string;
using std::cout;
using std::cerr;
using std::endl;

//struct Morph{
//    unordered_map<int, int> content;
//    void add(int k, int v){
//        content[k] = v;
//    }
//};

namespace ConllU{enum{ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC};}

class ConllTreebank;

struct Pair{
    int first;
    int second;
    Pair(int first, int second);
};

struct Output{

    static const int BASE;

    string code;
    bool xpos;
    bool morph;
    int n_feats;

    bool n_chars;
    int max_chars;
    vector<int> n_labels;

    bool bigram_left;
    unordered_map<int, int> bigrams_left;

    bool bigram_right;
    unordered_map<int, int> bigrams_right;

    bool trigram;
    unordered_map<int, int> trigrams;

    bool skipgram;
    unordered_map<int, int> skipgrams;

    bool experts;
    vector<Pair> expert_classes;

    Output(string s);
    void initialize(string s);

    void get_size_(unordered_map<int, int> &map);
    void get_output_sizes();

    void export_model(string output_dir);
    void import_model(string output_dir);

    void update_encoder(unordered_map<int, int> &map, int pair_id);
    void update_bigrams(ConllTreebank &treebank);

    int get_code_bigram_left(int first, int second);
    int get_code_bigram_right(int second, int third);
    int get_code_trigram(int first, int second, int third);
    int get_code_skipgram(int first, int third);

    int get_code(unordered_map<int, int> &map, int pair_id);

    bool add_expert(int l1, int l2);
};


class ConllToken{

    int _position;
    String _form;
    int _iform;
    int _cpos;
    int _fpos;

    int head;
    int label;

    vector<int> _morpho;

public:
    //ConllToken(int position, String _form, int _iform);
    ConllToken(int position, String _form, int _iform, int _cpos, int _fpos, vector<int> morpho);

    int i();
    int form();
    int cpos();
    int fpos();

    void cpos(int new_cpos);
    void fpos(int new_fpos);

    void print_morphology(ostream &os);
    bool has_morpho();
    int get_morpho(int type);
    void set_morpho(int type, int val);

    int len_form();

    friend ostream & operator<<(ostream &os, ConllToken &ct);
};

class ConllTree{
    vector<ConllToken> tokens;

public:
    ConllTree(vector<ConllToken> &tokens);
    ConllToken* operator[](int i);
    int size();

    void to_training_example(vector<STRCODE> &X, vector<vector<int>> &Y, Output &output);

    void assign_tags(vector<vector<int>> &Y, Output &output);

    friend ostream & operator<<(ostream &os, ConllTree &ct);
};

class ConllTreebank{
    vector<ConllTree> trees;
    //vector<int> voc_sizes;
    unordered_map<int, int> frequencies;
public:
    ConllTreebank();
    void add_tree(ConllTree &tree);
    ConllTree* operator[](int i);
    int size();

    void update_vocsize_and_frequencies();

    void shuffle();

    void subset(ConllTreebank &other, int n);

    unordered_map<int, int>* get_frequencies_dict();

    friend ostream & operator<<(ostream &os, ConllTreebank &ct);
};

void parse_morphology(String &s, vector<int> &morph, bool train);

void read_conll_corpus(std::string &filename,
                       ConllTreebank &treebank,
                       bool train);



#endif // CONLL_UTILS_H

