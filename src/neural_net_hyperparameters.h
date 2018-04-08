#ifndef NEURAL_NET_HYPERPARAMETERS_H
#define NEURAL_NET_HYPERPARAMETERS_H


#include "layers.h"

struct NetTopology{
    int n_hidden_layers;
    int size_hidden_layers;
    vector<int> embedding_size_type;
    NetTopology();
};

struct CharRnnParameters{
    int dim_char;
    int dim_char_based_embeddings;
    int crnn;
    CharRnnParameters();
};

struct RnnParameters{
    int cell_type;
    int depth; // 1 forward rnn, 2, bi-rnn, etc
    int hidden_size;
    int features; // number of features to consider for bi-rnn: if 2 -> (word,tag) if 3 -> (word,tag,morph1) etc..

    CharRnnParameters crnn;
    //int char_rnn_feature_extractor;  // make this an int ?

//    bool auxiliary_task;
//    int auxiliary_task_max_target;  // predict from features + 1 to auxiliary_task_max

    RnnParameters();
};


struct NeuralNetParameters{
    NetTopology topology;
    RnnParameters rnn;
    double learning_rate;
    double decrease_constant;
    double clip_value;
    double gaussian_noise_eta;

    bool gaussian_noise;
    bool gradient_clipping;
    bool soft_clipping;
    //bool rnn_feature_extractor;

    vector<string> header;
    vector<int> voc_sizes;
    NeuralNetParameters();

    void print(ostream &os);

    static void read_option_file(const string &filename, NeuralNetParameters &p);
};


#endif // NEURAL_NET_HYPERPARAMETERS_H

