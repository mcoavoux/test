#ifndef CONLL_UTILS_H
#define CONLL_UTILS_H

#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>

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

class ConllToken{

    int _position;
    String _form;
    int _iform;
    int _cpos;
    int _fpos;

    int head;
    int label;

public:
    //ConllToken(int position, String _form, int _iform);
    ConllToken(int position, String _form, int _iform, int _cpos, int _fpos);
    int i();
    int form();
    int cpos();
    int fpos();

    void cpos(int new_cpos);
    void fpos(int new_fpos);

    friend ostream & operator<<(ostream &os, ConllToken &ct);
};

class ConllTree{
    vector<ConllToken> tokens;

public:
    ConllTree(vector<ConllToken> &tokens);
    ConllToken* operator[](int i);
    int size();

    void to_training_example(vector<STRCODE> &X, vector<vector<int>> &Y);

    void assign_tags(vector<vector<int>> &Y);

    friend ostream & operator<<(ostream &os, ConllTree &ct);
};

class ConllTreebank{
    vector<ConllTree> trees;
    vector<int> voc_sizes;
    unordered_map<int, int> frequencies;
public:
    ConllTreebank();
    void add_tree(ConllTree &tree);
    ConllTree* operator[](int i);
    int size();

    void update_vocsize_and_frequencies();

    void shuffle();

    void subset(ConllTreebank &other, int n);

    friend ostream & operator<<(ostream &os, ConllTreebank &ct);
};

void read_conll_corpus(std::string &filename,
                       ConllTreebank &treebank,
                       bool train);



#endif // CONLL_UTILS_H

