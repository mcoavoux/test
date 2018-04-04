#ifndef NEURAL_ENCODER_H
#define NEURAL_ENCODER_H

#include <limits>
#include <unordered_map>
#include <fstream>
#include <math.h>
#include <memory>
#include <vector>
#include <iomanip>
#include "layers.h"
#include "neural_net_hyperparameters.h"

using std::ifstream;
using std::ofstream;
using std::cerr;
using std::endl;
using std::unordered_map;
using std::shared_ptr;


const double MINUS_INFINITY = - std::numeric_limits<double>::infinity();

typedef vector<vector<shared_ptr<AbstractNeuralNode>>> NodeMatrix;


/*
struct AuxiliaryTaskEvaluator{
    vector<int> good;
    float total;
    float complete_match;

    friend ostream& operator<<(ostream &os, AuxiliaryTaskEvaluator &ev){
        os << "{";
        for (int i = 0; i < ev.good.size(); i++){
            os << " " << i << "="<< std::setprecision(4) << 100.0*(ev.good[i]/ev.total);
        }
        os << " cm=" << std::setprecision(4) << 100.0*(ev.complete_match/ev.total) << " }";
        return os;
    }
};
*/

class CharBiRnnFeatureExtractor{
    vector<shared_ptr<RecurrentLayerWrapper>> layers;// 0: forward, 1: backward, 2: forward, 3:backward, etc...

    // Computation nodes
    vector<shared_ptr<AbstractNeuralNode>> init_nodes;
    vector<NodeMatrix> states; // states[word][depth][char]

    // input (lookup) nodes
    vector<NodeMatrix> input; // input[word][depth][char]

    // hyperparameters and lookup tables
    LookupTable lu;
    CharRnnParameters *params;

    // parameters
    vector<shared_ptr<Parameter>> parameters;
    SequenceEncoder encoder;

    vector<vector<Vec>> precomputed_embeddings;

public:
    CharBiRnnFeatureExtractor();
    CharBiRnnFeatureExtractor(CharRnnParameters *nn_parameters);
    ~CharBiRnnFeatureExtractor();

    void precompute_lstm_char();
    bool has_precomputed();
    void init_encoders();
    void build_computation_graph(vector<STRCODE> &buffer);
    void add_init_node(int depth);
    void fprop();
    void bprop();
    void update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);
    double gradient_squared_norm();
    void scale_gradient(double scale);
    void operator()(int i, vector<shared_ptr<AbstractNeuralNode>> &output);
    int size();
    void copy_encoders(CharBiRnnFeatureExtractor &other);
    void assign_parameters(CharBiRnnFeatureExtractor &other);
    void average_weights(int T);
    void get_parameters(vector<shared_ptr<Parameter>> &weights);
    void export_model(const string &outdir);
    void load_parameters(const string &outdir);
    void reset_gradient_history();
};




class BiRnnFeatureExtractor{

    vector<shared_ptr<RecurrentLayerWrapper>> layers;// 0: forward, 1: backward, 2: forward, 3:backward, etc...

    // Computation nodes
    vector<shared_ptr<AbstractNeuralNode>> init_nodes;
    //vector<vector<shared_ptr<AbstractNeuralNode>>> states; // idem
    NodeMatrix states;

    // input (lookup) nodes
    //vector<vector<shared_ptr<AbstractNeuralNode>>> input;
    NodeMatrix input;

    // hyperparameters and lookup tables
    LookupTable *lu;
    NeuralNetParameters *params;

    // parameters
    vector<shared_ptr<Parameter>> parameters;

//    Vec out_of_bounds;
//    Vec out_of_bounds_d;

    CharBiRnnFeatureExtractor char_rnn;

    // Auxiliary task
    //static const int AUX_HIDDEN_LAYER_SIZE = 32;
    /*
    vector<vector<shared_ptr<Layer>>> auxiliary_layers;
    vector<shared_ptr<Parameter>> auxiliary_parameters;
    vector<int> aux_output_sizes;
    vector<vector<int>> aux_targets;
    vector<NodeMatrix> auxiliary_output_nodes;
    int aux_start;
    int aux_end;
    */

    bool train_time;

    bool parse_time;

public:
    BiRnnFeatureExtractor();
    BiRnnFeatureExtractor(NeuralNetParameters *nn_parameters, LookupTable *lookup);

    ~BiRnnFeatureExtractor();

    void precompute_char_lstm();

    void build_computation_graph(vector<STRCODE> &buffer, bool aux_task=false);

    void add_init_node(int depth);

    AbstractNeuralNode* get_recurrent_node(shared_ptr<AbstractNeuralNode> &pred,
                                           vector<shared_ptr<AbstractNeuralNode>> &input_nodes,
                                           RecurrentLayerWrapper &l);

    void fprop();

    void bprop();

    void update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);
    double gradient_squared_norm();
    void scale_gradient(double scale);

    //void operator()(int i, vector<Vec*> &data, vector<Vec*> &data_grad);
    void operator()(int i, vector<shared_ptr<AbstractNeuralNode>> &output);

    int size();

    void assign_parameters(BiRnnFeatureExtractor &other);
    void copy_char_birnn(BiRnnFeatureExtractor &other);

    void average_weights(int T);

    void get_parameters(vector<shared_ptr<Parameter>> &weights);

    void export_model(const string &outdir);

    void load_parameters(const string &outdir);

    /*
    void auxiliary_task_summary(ostream &os);
    void add_aux_graph(vector<STRCODE> &buffer, vector<vector<int>> &targets, bool aux_only);
    void fprop_aux();
    void bprop_aux();
    void update_aux(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);
    void eval_aux(AuxiliaryTaskEvaluator &evaluator);

    void assign_deplabels(vector<shared_ptr<Node>> &buffer, int deplabel_id);
    void assign_tags(vector<shared_ptr<Node>> &buffer);
    void assign_morphological_features(vector<shared_ptr<Node>> &buffer, int deplabel_id);

    void auxiliary_gradient_check(vector<shared_ptr<Node>> &buffer, double epsilon);
    double aux_loss();
    double full_fprop_aux(vector<shared_ptr<Node>> &buffer);
    //int n_aux_tasks();
    void aux_reset_gradient_history();
    */

    void set_train_time(bool b);
};






#endif // NEURAL_ENCODER_H

