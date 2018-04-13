#ifndef BILSTM_TAGGER_H
#define BILSTM_TAGGER_H

#include <vector>
#include <memory>

#include "layers.h"
#include "neural_encoder.h"


using std::vector;

class BiLstmTagger{

    int n_updates_;
    int T_;
    vector<int> n_classes_;
    int hidden_size;
    int n_hidden;
    int voc_size;

    NeuralNetParameters params_;

    LookupTable lu;

    BiRnnFeatureExtractor rnn;

    vector<vector<shared_ptr<Layer>>> layers;
    vector<shared_ptr<Parameter>> parameters;

    vector<NodeMatrix> output_nodes;


    // Place holders for computations
//    vector<vector<Vec*>> t_edata;
//    vector<vector<Vec*>> t_edata_grad;

//    vector<vector<vector<Vec>>> t_states;
//    vector<vector<vector<Vec>>> t_dstates;

public:

    BiLstmTagger(int vocsize, vector<int> &n_classes, NeuralNetParameters &params);


    double get_learning_rate();

    void train_one(vector<STRCODE> &X, vector<vector<int>> &Y);
    void predict_one(vector<STRCODE> &X, vector<vector<int>> &Y);
    void eval_one(vector<STRCODE> &X, vector<vector<int>> &Y, vector<float> &losses, vector<float> &accuracies);
    void fprop(vector<STRCODE> &X);
    void get_losses(vector<float> &losses, vector<vector<int> > &targets);
    void get_predictions(vector<vector<int>> &predictions);

    void bprop(vector<vector<int>> &targets);
    void update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);

    void assign_parameters(BiLstmTagger *other);

    BiLstmTagger* copy();

    void average_parameters();

    void export_model(string &output_dir);
    void import_model(string &output_dir);
};


#endif // BILSTM_TAGGER_H

