

#include "bilstm_tagger.h"


BiLstmTagger::BiLstmTagger(int vocsize, vector<int> &n_classes, NeuralNetParameters &params):
    n_updates_(0), T_(0), n_classes_(n_classes), voc_size(vocsize), params_(params){

    hidden_size = params_.topology.size_hidden_layers;
    n_hidden = params_.topology.n_hidden_layers;

    lu = LookupTable(vocsize, params_.topology.embedding_size_type[enc::TOK]);

    vector<int> input_sizes{params_.rnn.hidden_size, params_.rnn.hidden_size};

    for (int i = 0; i < n_classes_.size(); i++){
        int output_size = n_classes_[i];
        vector<shared_ptr<Layer>> task_i;

        if (n_hidden > 0){
            task_i.push_back(shared_ptr<Layer>(new MultipleLinearLayer(2, input_sizes, hidden_size)));
            task_i.push_back(shared_ptr<Layer>(new ReLU()));

            for (int l = 1; l < n_hidden; l++){
                task_i.push_back(shared_ptr<Layer>(new AffineLayer(hidden_size, hidden_size)));
                task_i.push_back(shared_ptr<Layer>(new ReLU()));
            }
            task_i.push_back(shared_ptr<Layer>(new AffineLayer(hidden_size, output_size)));
            task_i.push_back(shared_ptr<Layer>(new Softmax()));
        }else{
            task_i.push_back(shared_ptr<Layer>(new MultipleLinearLayer(2, input_sizes, output_size)));
            task_i.push_back(shared_ptr<Layer>(new Softmax()));
        }
        layers.push_back(task_i);
    }

    for (int i = 0; i < layers.size(); i++){
        for (int j = 0; j < layers[i].size(); j++){
            layers[i][j]->get_params(this->parameters);
        }
    }
    rnn = BiRnnFeatureExtractor(&params_, &lu);
}

double BiLstmTagger::get_learning_rate(){
    return params_.learning_rate / (1.0 + T_ * params_.decrease_constant);
}

void BiLstmTagger::train_one(vector<STRCODE> &X, vector<vector<int>> &Y){
    rnn.set_train_time(true);

    this->fprop(X);
    this->bprop(Y);

    double lr = get_learning_rate();

    this->update(lr, T_, params_.clip_value, params_.gradient_clipping, params_.gaussian_noise, params_.gaussian_noise_eta);
    T_ += 1;
}

void BiLstmTagger::predict_one(vector<STRCODE> &X, vector<vector<int>> &Y){
    rnn.set_train_time(false);
    this->fprop(X);
    this->get_predictions(Y);
}

void BiLstmTagger::eval_one(vector<STRCODE> &X, vector<vector<int>> &Y, vector<vector<int>> &predictions, vector<float> &losses){
    rnn.set_train_time(false);
    this->fprop(X);
    this->get_losses(losses, Y);
    this->get_predictions(predictions);
}

void BiLstmTagger::fprop(vector<STRCODE> &X){
    rnn.build_computation_graph(X);
    rnn.fprop();

    output_nodes.resize(X.size());

    for (int i = 0; i < X.size(); i++){
        vector<shared_ptr<AbstractNeuralNode>> input;
        rnn(i, input);
        output_nodes[i].resize(n_classes_.size());
        for (int t = 0; t < layers.size(); t++){
            output_nodes[i][t].clear();
            if (n_hidden > 0){
                output_nodes[i][t].push_back(
                            shared_ptr<AbstractNeuralNode>(
                                new ComplexNode(
                                    this->hidden_size,
                                    layers[t][0].get(),
                                input)));  // size / layer / vector input
            }else{
                output_nodes[i][t].push_back(
                            shared_ptr<AbstractNeuralNode>(
                                new ComplexNode(
                                    this->n_classes_[t],
                                    layers[t][0].get(),
                                input)));  // size / layer / vector input
            }
            int layer_size = this->hidden_size;
            for (int l = 1; l < layers[t].size(); l++){
                if (l >= layers[t].size() - 2){
                    layer_size = n_classes_[t];
                }
                output_nodes[i][t].push_back(
                            shared_ptr<AbstractNeuralNode>(
                                new SimpleNode(layer_size,
                                               layers[t][l].get(),
                                output_nodes[i][t].back())));
            }
        }
    }
    for (int i = 0; i < output_nodes.size(); i++){
        for (int t = 0; t < output_nodes[i].size(); t++){
            for (int l = 0; l < output_nodes[i][t].size(); l++){
                output_nodes[i][t][l]->fprop();
            }
        }
    }
}

void BiLstmTagger::get_losses(vector<float> &losses, vector<vector<int>> &targets){
    assert(losses.size() == n_classes_.size());
    for (int i = 0; i < output_nodes.size(); i++){
        //for (int t = 0; t < output_nodes[i].size(); t++){
        for (int t = 0; t < n_classes_.size(); t++){
            Vec* v = output_nodes[i][t].back()->v();
            losses[t] += - log((*v)[targets[i][t]]);
        }
    }
}

void BiLstmTagger::get_predictions(vector<vector<int>> &predictions){
    predictions.resize(output_nodes.size());
    for (int i = 0; i < output_nodes.size(); i++){
        predictions[i].resize(n_classes_.size());
        //for (int t = 0; t < output_nodes[i].size(); t++){
        assert(output_nodes[i].size() == n_classes_.size());
//        cerr << n_classes_.size() << endl;
//        cerr << output_nodes[i].size() << endl;
        for (int t = 0; t < n_classes_.size(); t++){
            Vec* v = output_nodes[i][t].back()->v();
            int argmax;
            v->maxCoeff(&argmax);
            predictions[i][t] = argmax;
        }
//        for (int t = 0; t < output_nodes[i].size(); t++){
//            cerr << predictions[i][t] << " ";
//        }
//        cerr << endl;
    }
}

void BiLstmTagger::bprop(vector<vector<int>> &targets){
    for (int i = 0; i < output_nodes.size(); i++){
        for (int t = 0; t < output_nodes[i].size(); t++){
            layers[t].back()->target = targets[i][t];
            for (int l = output_nodes[i][t].size() -1; l >= 0; l--){
                output_nodes[i][t][l]->bprop();
            }
        }
    }
    rnn.bprop();
}

void BiLstmTagger::update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    for (shared_ptr<Parameter> &p: parameters){
        p->update(lr, T, clip, clipping, gaussian, gaussian_eta);
    }
    rnn.update(lr, T, clip, clipping, gaussian, gaussian_eta);
    lu.update(lr, T, clip, clipping, gaussian, gaussian_eta);
}

void BiLstmTagger::assign_parameters(BiLstmTagger *other){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->assign(other->parameters[i]);
    }
    rnn.assign_parameters(other->rnn);
    rnn.copy_char_birnn(other->rnn);
}

BiLstmTagger* BiLstmTagger::copy(){
    BiLstmTagger* avg_tagger = new BiLstmTagger(voc_size, n_classes_, params_);
    avg_tagger->n_updates_ = n_updates_;
    avg_tagger->T_ = T_;
    avg_tagger->hidden_size = hidden_size;
    avg_tagger->n_hidden = n_hidden;
    avg_tagger->lu = lu;

    avg_tagger->assign_parameters(this);

    return avg_tagger;
}

void BiLstmTagger::average_parameters(){
    lu.average(T_);
    rnn.average_weights(T_);
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->average(T_);
    }
}

void BiLstmTagger::export_model(string &output_dir){

    enc::export_encoders(output_dir);

    lu.export_model(output_dir + "/lu");
    rnn.export_model(output_dir);

    ofstream out_params(output_dir + "/hyperparameters");
    params_.print(out_params);
    out_params.close();

    ofstream os(output_dir + "/n_classes");
    os << n_classes_.size() << endl;
    for (int i = 0; i < n_classes_.size(); i++){
        os << n_classes_[i] << " ";
    }
    os.close();

    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->export_model(output_dir+"/parameters" + std::to_string(i));
    }
}

void BiLstmTagger::import_model(string &output_dir){

    lu.clear();
    lu.load(output_dir + "/lu");
    rnn.load_parameters(output_dir);

    n_updates_ = 0;
    T_ = 0;

    ifstream in(output_dir + "/n_classes");
    n_classes_.clear();
    int size;
    int tmp;
    in >> size;
    while (in >> tmp){
        n_classes_.push_back(tmp);
    }
    assert(n_classes_.size() == size);

    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->load(output_dir+"/parameters" + std::to_string(i));
    }
    rnn.precompute_char_lstm();
}

void BiLstmTagger::add_expert_classifier(){
    int output_size = 3;
    n_classes_.push_back(output_size);
    vector<shared_ptr<Layer>> new_classifier;
    vector<int> input_sizes{params_.rnn.hidden_size, params_.rnn.hidden_size};

    if (n_hidden > 0){
        new_classifier.push_back(shared_ptr<Layer>(new MultipleLinearLayer(2, input_sizes, hidden_size)));
        new_classifier.push_back(shared_ptr<Layer>(new ReLU()));

        for (int l = 1; l < n_hidden; l++){
            new_classifier.push_back(shared_ptr<Layer>(new AffineLayer(hidden_size, hidden_size)));
            new_classifier.push_back(shared_ptr<Layer>(new ReLU()));
        }
        new_classifier.push_back(shared_ptr<Layer>(new AffineLayer(hidden_size, output_size)));
        new_classifier.push_back(shared_ptr<Layer>(new Softmax()));
    }else{
        new_classifier.push_back(shared_ptr<Layer>(new MultipleLinearLayer(2, input_sizes, output_size)));
        new_classifier.push_back(shared_ptr<Layer>(new Softmax()));
    }
    layers.push_back(new_classifier);

    for (int j = 0; j < new_classifier.size(); j++){
        new_classifier[j]->get_params(this->parameters);
    }
}


