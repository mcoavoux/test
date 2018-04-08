
#include "neural_encoder.h"



CharBiRnnFeatureExtractor::CharBiRnnFeatureExtractor(){}
CharBiRnnFeatureExtractor::CharBiRnnFeatureExtractor(CharRnnParameters *nn_parameters)
    : params(nn_parameters){
    encoder = SequenceEncoder(nn_parameters->crnn);
    vector<int> input_sizes{params->dim_char};

    int cell_type = RecurrentLayerWrapper::LSTM;
    int hidden_size = params->dim_char_based_embeddings;

    // 2 layers only
    layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(cell_type, input_sizes, hidden_size)));
    layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(cell_type, input_sizes, hidden_size)));

    for (int i = 0; i < layers.size(); i++){
        for (int j = 0; j < layers[i]->size(); j++){
            (*layers[i])[j]->get_params(parameters);
        }
    }
}

CharBiRnnFeatureExtractor::~CharBiRnnFeatureExtractor(){}

void CharBiRnnFeatureExtractor::precompute_lstm_char(){
    cerr << "Precomputing char-lstm for known words" << endl;
    vector<STRCODE> fake_buffer; // contain the list of tokens in vocabulary
    for (STRCODE i = 0; i < enc::hodor.size(enc::TOK); i++){
        //const vector<STRCODE> morph{i};
        fake_buffer = {i};

        build_computation_graph(fake_buffer);
        fprop();

        vector<Vec> pair{*(states[0][0].back()->v()), *(states[0][1].front()->v())};
        precomputed_embeddings.push_back(pair);
    }

    cerr << "Precomputing char-lstm for known words: done" << endl;
}

bool CharBiRnnFeatureExtractor::has_precomputed(){
    return precomputed_embeddings.size() > 0;
}

void CharBiRnnFeatureExtractor::init_encoders(){
    encoder.init();
    lu = LookupTable(encoder.char_voc_size(), params->dim_char);
}


void CharBiRnnFeatureExtractor::build_computation_graph(vector<STRCODE> &buffer){

    input = vector<NodeMatrix>(buffer.size());
    states.resize(input.size());

    init_nodes.clear();
    for (int depth = 0; depth < 2; depth++){
        add_init_node(depth);
    }

    for (int w = 0; w < input.size(); w++){
        STRCODE tokcode = buffer[w];

        // If a precomputed vector is available
        if (tokcode < precomputed_embeddings.size()){
            shared_ptr<AbstractNeuralNode> forwardnode(new ConstantNode(&precomputed_embeddings[tokcode][0]));
            shared_ptr<AbstractNeuralNode> backwardnode(new ConstantNode(&precomputed_embeddings[tokcode][1]));
            vector<shared_ptr<AbstractNeuralNode>> forward{forwardnode};
            vector<shared_ptr<AbstractNeuralNode>> backward{backwardnode};
            states[w] = {forward, backward};
        }else{
            vector<int> *sequence = encoder(tokcode);
            for (int c = 0; c < sequence->size(); c++){
                shared_ptr<VecParam> e;
                lu.get(sequence->at(c), e);
                vector<shared_ptr<AbstractNeuralNode>> proxy{shared_ptr<AbstractNeuralNode>(new LookupNode(*e))};
                input[w].push_back(proxy);
            }

            states[w] = {vector<shared_ptr<AbstractNeuralNode>>(sequence->size()),
                         vector<shared_ptr<AbstractNeuralNode>>(sequence->size())};

            int depth = 0;
            states[w][depth][0] = shared_ptr<AbstractNeuralNode>(
                        new LstmNode(params->dim_char_based_embeddings,
                                     init_nodes[depth],
                                     input[w][0],*layers[depth]));

            for (int c = 1; c < sequence->size(); c++){
                states[w][depth][c] = shared_ptr<AbstractNeuralNode>(
                            new LstmNode(params->dim_char_based_embeddings,
                                         states[w][depth][c-1],
                            input[w][c], *layers[depth]));
            }
            depth = 1;
            states[w][depth].back() = shared_ptr<AbstractNeuralNode>(
                        new LstmNode(params->dim_char_based_embeddings,
                                     init_nodes[depth],
                                     input[w].back(), *layers[depth]));

            for (int c = sequence->size()-2; c >= 0; c--){
                states[w][depth][c] = shared_ptr<AbstractNeuralNode>(
                            new LstmNode(params->dim_char_based_embeddings,
                                         states[w][depth][c+1],
                            input[w][c], *layers[depth]));
            }
        }
    }
}


void CharBiRnnFeatureExtractor::add_init_node(int depth){
    shared_ptr<ParamNode> init11(new ParamNode(params->dim_char_based_embeddings, (*layers[depth])[GruNode::INIT2]));
    shared_ptr<AbstractNeuralNode> init1(new MemoryNodeInitial(
                                             params->dim_char_based_embeddings,
                                             (*layers[depth])[GruNode::INIT1],
                                         init11));
    init_nodes.push_back(init1);
}

void CharBiRnnFeatureExtractor::fprop(){
    for (int i = 0; i < init_nodes.size(); i++){
        init_nodes[i]->fprop();
    }
    for (int w = 0; w < states.size(); w++){
        for (int c = 0; c < states[w][0].size(); c++){
            states[w][0][c]->fprop();
        }
        for (int c = states[w][1].size() -1; c >= 0; c--){
            states[w][1][c]->fprop();
        }
    }
}

void CharBiRnnFeatureExtractor::bprop(){
    for (int w = 0; w < states.size(); w++){
        for (int c = states[w][0].size() -1; c >= 0; c--){
            states[w][0][c]->bprop();
        }
        for (int c = 0; c < states[w][1].size(); c++){
            states[w][1][c]->bprop();
        }
    }
    for (int i = 0; i < init_nodes.size(); i++){
        init_nodes[i]->bprop();
    }
}

void CharBiRnnFeatureExtractor::update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->update(lr, T, clip, clipping, gaussian, gaussian_eta);
    }
    lu.update(lr, T, clip, clipping, gaussian, gaussian_eta);
}

double CharBiRnnFeatureExtractor::gradient_squared_norm(){
    double gsn = 0;
    for (int i = 0; i < parameters.size(); i++){
        gsn += parameters[i]->gradient_squared_norm();
    }
    gsn += lu.gradient_squared_norm();
    return gsn;
}

void CharBiRnnFeatureExtractor::scale_gradient(double scale){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->scale_gradient(scale);
    }
    lu.scale_gradient(scale);
}


void CharBiRnnFeatureExtractor::operator()(int i, vector<shared_ptr<AbstractNeuralNode>> &output){
    assert( i >= 0 && i < size() );
    output = {states[i][0].back(), states[i][1].front()};
}

int CharBiRnnFeatureExtractor::size(){
    assert(input.size() == states.size());
    return input.size();
}

void CharBiRnnFeatureExtractor::copy_encoders(CharBiRnnFeatureExtractor &other){
    lu = other.lu;
    encoder = other.encoder;
}

void CharBiRnnFeatureExtractor::assign_parameters(CharBiRnnFeatureExtractor &other){
    assert(parameters.size() == other.parameters.size());
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->assign(other.parameters[i]);
    }
}

void CharBiRnnFeatureExtractor::average_weights(int T){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->average(T);
    }
    lu.average(T);
}
void CharBiRnnFeatureExtractor::get_parameters(vector<shared_ptr<Parameter>> &weights){ // REMINDER: this is used by gradient checker
    weights.insert(weights.end(), parameters.begin(), parameters.end());
    lu.get_active_params(weights);
}

void CharBiRnnFeatureExtractor::export_model(const string &outdir){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->export_model(outdir+"/char_rnn_parameters" + std::to_string(i));
    }
    lu.export_model(outdir+"/lu_char_rnn");
}

void CharBiRnnFeatureExtractor::load_parameters(const string &outdir){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->load(outdir+"/char_rnn_parameters" + std::to_string(i));
    }
    lu.clear();
    lu.load(outdir+"/lu_char_rnn");
}

void CharBiRnnFeatureExtractor::reset_gradient_history(){
    lu.reset_gradient_history();
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->reset_gradient_history();
    }
}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////





BiRnnFeatureExtractor::BiRnnFeatureExtractor():train_time(false), parse_time(false){}
BiRnnFeatureExtractor::BiRnnFeatureExtractor(NeuralNetParameters *nn_parameters,
                      LookupTable *lookup)
    :lu(lookup), params(nn_parameters), train_time(false), parse_time(false){

    vector<int> input_sizes;

    if (params->rnn.crnn.crnn > 0){
        input_sizes.push_back(params->rnn.crnn.dim_char_based_embeddings);
        input_sizes.push_back(params->rnn.crnn.dim_char_based_embeddings);
    }
    for (int i = 0; i < params->rnn.features; i++){
        input_sizes.push_back(params->topology.embedding_size_type[i+1]);
    }

    layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(params->rnn.cell_type, input_sizes, params->rnn.hidden_size)));
    layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(params->rnn.cell_type, input_sizes, params->rnn.hidden_size)));
    for (int i = 2; i < params->rnn.depth; i++){
        vector<int> prec_layer_sizes{params->rnn.hidden_size, params->rnn.hidden_size};
        layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(params->rnn.cell_type, prec_layer_sizes, params->rnn.hidden_size)));
    }

    for (int i = 0; i < layers.size(); i++){
        for (int j = 0; j < layers[i]->size(); j++){
            (*layers[i])[j]->get_params(parameters);
        }
    }

//    out_of_bounds = Vec::Zero(params->rnn.hidden_size);
//    out_of_bounds_d = Vec::Zero(params->rnn.hidden_size);

    if (params->rnn.crnn.crnn > 0){
        char_rnn = CharBiRnnFeatureExtractor(& params->rnn.crnn);
        char_rnn.init_encoders();
    }


//    if (params->rnn.auxiliary_task){
//        aux_start = params->rnn.features;
//        aux_end = params->rnn.auxiliary_task_max_target;
//        auxiliary_layers.resize(aux_end - aux_start);
//        aux_output_sizes.resize(aux_end - aux_start);
//        vector<int> input_sizes{params->rnn.hidden_size, params->rnn.hidden_size};
//        for (int i = aux_start; i < aux_end; i++){
//            int output_size = params->voc_sizes[i+1]; // +1 -> 0 is non terminals
//            aux_output_sizes[i-aux_start] = output_size;
//            auxiliary_layers[i-aux_start] ={
//                //shared_ptr<Layer>(new MultipleLinearLayer(2, input_sizes, AUX_HIDDEN_LAYER_SIZE)),
//                //shared_ptr<Layer>(new ReLU()),
//                //shared_ptr<Layer>(new AffineLayer(AUX_HIDDEN_LAYER_SIZE, output_size)),
//                shared_ptr<Layer>(new MultipleLinearLayer(2, input_sizes, output_size)),
//                shared_ptr<Layer>(new Softmax())
//            };
//            for (int j = 0; j < auxiliary_layers[i-aux_start].size(); j++){
//                auxiliary_layers[i-aux_start][j]->get_params(auxiliary_parameters);
//                //auxiliary_layers[i-aux_start][1]->get_params(auxiliary_parameters);
//            }
//        }
//    }
}


BiRnnFeatureExtractor::~BiRnnFeatureExtractor(){}

void BiRnnFeatureExtractor::precompute_char_lstm(){
    parse_time = true;
    char_rnn.precompute_lstm_char();
}

void BiRnnFeatureExtractor::build_computation_graph(vector<STRCODE> &buffer){

    if (params->rnn.crnn.crnn > 0){
        char_rnn.build_computation_graph(buffer);
    }

    int add_features = (params->rnn.crnn.crnn > 0) ? 2 : 0;

    input = NodeMatrix(
                buffer.size(),
                vector<shared_ptr<AbstractNeuralNode>>(
                    params->rnn.features + add_features,
                    nullptr));  // +2 if char rnn

    for (int i = 0; i < buffer.size(); i++){

        if (params->rnn.crnn.crnn > 0){
            vector<shared_ptr<AbstractNeuralNode>> char_based_embeddings;
            char_rnn(i, char_based_embeddings);
            assert(char_based_embeddings.size() == 2);
            assert(char_based_embeddings[0].get() != NULL);
            assert(char_based_embeddings[1].get() != NULL);
            input[i][0] = char_based_embeddings[0];
            input[i][1] = char_based_embeddings[1];
        }

        shared_ptr<VecParam> e;
        //for (int f = 0; f < params->rnn.features; f++){
        STRCODE word_code = buffer[i];
        if (train_time && word_code != enc::UNDEF){ // 2% unknown words   --> won't work unless prob depends on frequency
            assert(word_code != enc::UNKNOWN);
            double threshold = 0.8375 / (0.8375 + enc::hodor.get_freq(word_code));
            if (rd::random() < threshold){
                word_code = enc::UNKNOWN;
            }
        }

        lu->get(word_code, e);
        input[i][add_features] = shared_ptr<AbstractNeuralNode>(new LookupNode(*e));
        //}
    }

    int depth = params->rnn.depth;

    //states.resize(params->rnn.depth);
    states.resize(depth);
    for (int d = 0; d < states.size(); d++){
        states[d].resize(buffer.size());
    }

    init_nodes.clear();
    for (int i = 0; i < depth; i++){
        add_init_node(i);
    }

    depth = 0;
    states[depth][0]=  shared_ptr<AbstractNeuralNode>(get_recurrent_node(init_nodes[depth], input[0], *layers[depth]));
    for (int i = 1; i < buffer.size(); i++){
        states[depth][i]= shared_ptr<AbstractNeuralNode>(get_recurrent_node(states[depth][i-1], input[i], *layers[depth]));
    }

    depth = 1;
    states[depth].back() = shared_ptr<AbstractNeuralNode>(get_recurrent_node(init_nodes[depth], input.back(), *layers[depth]));
    for (int i = buffer.size()-2; i >=0 ; i--){
        states[depth][i] = shared_ptr<AbstractNeuralNode>(get_recurrent_node(states[depth][i+1], input[i], *layers[depth]));
    }

    for (depth = 2; depth < params->rnn.depth; depth++){
        if (depth % 2 == 0){
            vector<shared_ptr<AbstractNeuralNode>> rnn_in{states[depth-1][0], states[depth-2][0]};
            states[depth][0] = shared_ptr<AbstractNeuralNode>(get_recurrent_node(init_nodes[depth], rnn_in, *layers[depth]));
            for (int i = 1; i < buffer.size(); i++){
                rnn_in = {states[depth-1][i], states[depth-2][i]};
                states[depth][i]= shared_ptr<AbstractNeuralNode>(get_recurrent_node(states[depth][i-1], rnn_in, *layers[depth]));
            }
        }else{
            vector<shared_ptr<AbstractNeuralNode>> rnn_in{states[depth-2].back(), states[depth-3].back()};
            states[depth].back() = shared_ptr<AbstractNeuralNode>(get_recurrent_node(init_nodes[depth], rnn_in, *layers[depth]));
            for (int i = buffer.size()-2; i >=0 ; i--){
                rnn_in = {states[depth-2][i], states[depth-3][i]};
                states[depth][i] = shared_ptr<AbstractNeuralNode>(get_recurrent_node(states[depth][i+1], rnn_in, *layers[depth]));
            }
        }
    }
}



void BiRnnFeatureExtractor::add_init_node(int depth){
    switch(params->rnn.cell_type){
    case RecurrentLayerWrapper::GRU:
    case RecurrentLayerWrapper::LSTM:{
        shared_ptr<ParamNode> init11(new ParamNode(params->rnn.hidden_size, (*layers[depth])[GruNode::INIT2]));
        shared_ptr<AbstractNeuralNode> init1(new MemoryNodeInitial(
                                                 params->rnn.hidden_size,
                                                 (*layers[depth])[GruNode::INIT1],
                                                  init11));
        init_nodes.push_back(init1);
        break;
    }
    case RecurrentLayerWrapper::RNN:{
        shared_ptr<AbstractNeuralNode> init(new ParamNode(params->rnn.hidden_size, (*layers[depth])[RnnNode::INIT]));
        init_nodes.push_back(init);
        break;
    }
    default:
        assert(false && "Not Implemented error or unknown rnn cell type");
    }
}

AbstractNeuralNode* BiRnnFeatureExtractor::get_recurrent_node(
        shared_ptr<AbstractNeuralNode> &pred,
        vector<shared_ptr<AbstractNeuralNode> > &input_nodes,
        RecurrentLayerWrapper &l){

    switch(params->rnn.cell_type){
    case RecurrentLayerWrapper::GRU:
        return new GruNode(params->rnn.hidden_size, pred, input_nodes, l);
    case RecurrentLayerWrapper::RNN:
        return new RnnNode(params->rnn.hidden_size, pred, input_nodes, l);
    case RecurrentLayerWrapper::LSTM:
        return new LstmNode(params->rnn.hidden_size, pred, input_nodes, l);
    default:
        assert(false);
    }
    assert(false);
    return nullptr;
}


void BiRnnFeatureExtractor::fprop(){
    if (params->rnn.crnn.crnn > 0){
        char_rnn.fprop();
    }
    for (int i = 0; i < init_nodes.size(); i++){
        init_nodes[i]->fprop();
    }

    for (int d = 0; d < states.size(); d++){
        if (d % 2 == 0){
            for (int i = 0; i < states[d].size(); i++){
                states[d][i]->fprop();
            }
        }else{
            for (int i = states[d].size() -1; i >= 0; i--){
                states[d][i]->fprop();
            }
        }
    }
}

void BiRnnFeatureExtractor::bprop(){
    for (int d = states.size()-1; d >= 0; d--){
        if (d % 2 == 0){
            for (int i = states[d].size() -1; i >= 0; i--){
                states[d][i]->bprop();
            }
        }else{
            for (int i = 0; i < states[d].size(); i++){
                states[d][i]->bprop();
            }
        }
    }
    for (int i = 0; i < init_nodes.size(); i++){
        init_nodes[i]->bprop();
    }
    if (params->rnn.crnn.crnn > 0){
        char_rnn.bprop();
    }
}

void BiRnnFeatureExtractor::update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->update(lr, T, clip, clipping, gaussian, gaussian_eta);
    }
    if (params->rnn.crnn.crnn > 0){
        char_rnn.update(lr, T, clip, clipping, gaussian, gaussian_eta);
    }
}

double BiRnnFeatureExtractor::gradient_squared_norm(){
    double gsn = 0;
    for (int i = 0; i < parameters.size(); i++){
        gsn += parameters[i]->gradient_squared_norm();
    }
    if (params->rnn.crnn.crnn > 0){
        gsn += char_rnn.gradient_squared_norm();
    }
//    for (int i = 0; i < auxiliary_parameters.size(); i++){
//        gsn += auxiliary_parameters[i]->gradient_squared_norm();
//    }
    return gsn;
}

void BiRnnFeatureExtractor::scale_gradient(double scale){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->scale_gradient(scale);
    }
//    for (int i = 0; i < auxiliary_parameters.size(); i++){
//        auxiliary_parameters[i]->scale_gradient(scale);
//    }
    char_rnn.scale_gradient(scale);
}


//void BiRnnFeatureExtractor::operator()(int i, vector<Vec*> &data, vector<Vec*> &data_grad){
//    if (i >= 0 && i < size()){
//        int j = params->rnn.depth - 2;
//        assert((j+2) == states.size());
//        data.push_back(states[j][i]->v());
//        data_grad.push_back(states[j][i]->d());
//        data.push_back(states[j+1][i]->v());
//        data_grad.push_back(states[j+1][i]->d());
//    }else{
//        // TODO: find cleverer way (use start / stop symbols ??)
//        for (int d = 0; d < 2; d++){
//            data.push_back(&out_of_bounds);
//            data_grad.push_back(&out_of_bounds_d);
//        }
//    }
//}

void BiRnnFeatureExtractor::operator()(int i, vector<shared_ptr<AbstractNeuralNode>> &output){
    if (i >= 0 && i < size()){
        int j = params->rnn.depth - 2;
        assert((j+2) == states.size());
        output.push_back(states[j][i]);
        output.push_back(states[j+1][i]);
    }else{
        assert(false);
//        for (int d = 0; d < 2; d++){
//            data.push_back(&out_of_bounds);
//            data_grad.push_back(&out_of_bounds_d);
//        }
    }
}

int BiRnnFeatureExtractor::size(){
    assert( states.size() > 0 );
    assert(input.size() == states[0].size());
    return input.size();
}

void BiRnnFeatureExtractor::assign_parameters(BiRnnFeatureExtractor &other){
    assert(parameters.size() == other.parameters.size());
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->assign(other.parameters[i]);
    }
//    for (int i = 0; i < auxiliary_parameters.size(); i++){
//        auxiliary_parameters[i]->assign(other.auxiliary_parameters[i]);
//    }
}

void BiRnnFeatureExtractor::copy_char_birnn(BiRnnFeatureExtractor &other){
    if (params->rnn.crnn.crnn > 0){
        char_rnn.copy_encoders(other.char_rnn);
        char_rnn.assign_parameters(other.char_rnn);
    }
}

void BiRnnFeatureExtractor::average_weights(int T){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->average(T);
    }
    if (params->rnn.crnn.crnn > 0){
        char_rnn.average_weights(T);
    }
//    if (params->rnn.auxiliary_task){
//        for (int i = 0; i < auxiliary_parameters.size(); i++){
//            auxiliary_parameters[i]->average(T);
//        }
//    }
}

void BiRnnFeatureExtractor::get_parameters(vector<shared_ptr<Parameter>> &weights){
    weights.insert(weights.end(), parameters.begin(), parameters.end());
    if (params->rnn.crnn.crnn){
        char_rnn.get_parameters(weights);
    }
}

void BiRnnFeatureExtractor::export_model(const string &outdir){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->export_model(outdir+"/rnn_parameters" + std::to_string(i));
    }
    if (params->rnn.crnn.crnn){
        char_rnn.export_model(outdir);
    }
//    for (int i = 0; i < auxiliary_parameters.size(); i++){
//        auxiliary_parameters[i]->export_model(outdir+"/rnn_aux_parameters" + std::to_string(i));
//    }
}

void BiRnnFeatureExtractor::load_parameters(const string &outdir){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->load(outdir+"/rnn_parameters" + std::to_string(i));
    }
    if (params->rnn.crnn.crnn){
        char_rnn.load_parameters(outdir);
    }
//    for (int i = 0; i < auxiliary_parameters.size(); i++){
//        auxiliary_parameters[i]->load(outdir+"/rnn_aux_parameters" + std::to_string(i));
//    }
}

/*
void BiRnnFeatureExtractor::auxiliary_task_summary(ostream &os){
    os << "Auxiliary tasks summary:" << endl;
    for (int i = 0; i < aux_output_sizes.size(); i++){
        os << "    task " << i << ": output size = " << aux_output_sizes[i] << endl;
    }
}


void BiRnnFeatureExtractor::add_aux_graph(vector<STRCODE> &buffer, vector<vector<int> > &targets, bool aux_only=true){
    build_computation_graph(buffer, aux_only);
    assert(input.size() == buffer.size());
    auxiliary_output_nodes.resize(input.size());

    for (int i = 0; i < buffer.size(); i++){
        int depth = 0;  /// params->rnn.depth - 2;  -> supervise lower tasks at lower layers
        vector<shared_ptr<AbstractNeuralNode>> input_nodes = {
                states[depth][i],
                states[depth+1][i]};
        auxiliary_output_nodes[i].resize(aux_end - aux_start);

        for (int j = 0; j < aux_end - aux_start; j++){

            auxiliary_output_nodes[i][j].clear();
            auxiliary_output_nodes[i][j].push_back(
                        shared_ptr<AbstractNeuralNode>(
                            new ComplexNode(aux_output_sizes[j],
                                            auxiliary_layers[j][0].get(), input_nodes)));  // size / layer / vector input
            auxiliary_output_nodes[i][j].push_back(
                        shared_ptr<AbstractNeuralNode>(
                            new SimpleNode(aux_output_sizes[j],
                                           auxiliary_layers[j][1].get(), auxiliary_output_nodes[i][j][0])));
        }
    }

    this->aux_targets = targets;

//    aux_targets = vector<vector<int>>(buffer.size());
//    for (int i = 0; i < buffer.size(); i++){
//        aux_targets[i].resize(aux_end - aux_start);
//        for (int j = 0; j < aux_end - aux_start; j++){
//            if (buffer[i]->n_fields() > (j+aux_start)){
//                aux_targets[i][j] = buffer[i]->get_field(j + aux_start);
//                if (aux_targets[i][j] >= aux_output_sizes[j]){
////                    cerr << (j+aux_start) << endl;
////                    cerr << "unknown:" << enc::hodor.decode(aux_targets[i][j], j + aux_start + 1) << endl;
////                    for (int k = 0; k < buffer[i]->n_fields(); k++){
////                        cerr << enc::hodor.decode(buffer[i]->get_field(k), k+1) << " ";
////                    }cerr << endl;
//                    aux_targets[i][j] = enc::UNKNOWN;
//                }
//            }
//        }
//    }
}

void BiRnnFeatureExtractor::fprop_aux(){
    fprop();
    for (int i = 0; i < auxiliary_output_nodes.size(); i++){
        for (int j = 0; j < auxiliary_output_nodes[i].size(); j++){
            for (int k = 0; k < auxiliary_output_nodes[i][j].size(); k++){
                auxiliary_output_nodes[i][j][k]->fprop();
            }
        }
    }
}

void BiRnnFeatureExtractor::bprop_aux(){
    for (int i = 0; i < auxiliary_output_nodes.size(); i++){
        //int j = rand() % auxiliary_output_nodes[i].size(); // random auxiliary task
        //std::uniform_int_distribution<int> distribution(0,auxiliary_output_nodes[i].size()-1);
        //int j = distribution(Parameter::random);

        for (int j = 0; j < auxiliary_output_nodes[i].size(); j++){
            auxiliary_layers[j].back()->target = aux_targets[i][j];
            for (int k = auxiliary_output_nodes[i][j].size() - 1; k >= 0; k--){
                auxiliary_output_nodes[i][j][k]->bprop();
            }
        }
    }
    bprop();
}

void BiRnnFeatureExtractor::update_aux(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    double learning_rate_aux = lr;// / aux_output_sizes.size();
    for (int i = 0; i < auxiliary_parameters.size(); i++){
        auxiliary_parameters[i]->update(learning_rate_aux, T, clip, clipping, gaussian, gaussian_eta);
    }
    for (int i = 0; i < lu->size(); i++){
        lu->at(i).update(learning_rate_aux, T, clip, clipping, gaussian, gaussian_eta);
    }
    update(learning_rate_aux, T, clip, clipping, gaussian, gaussian_eta);
}

void BiRnnFeatureExtractor::eval_aux(AuxiliaryTaskEvaluator &evaluator){
    evaluator.total += auxiliary_output_nodes.size();
    for (int i = 0; i < auxiliary_output_nodes.size(); i++){
        bool all_good = true;
        for (int k = 0; k < auxiliary_output_nodes[i].size(); k++){
            Vec* output = auxiliary_output_nodes[i][k].back()->v();
            int argmax;
            output->maxCoeff(&argmax);
            //cerr << enc::hodor.decode(argmax, enc::TAG) << " " << enc::hodor.decode(aux_targets[i][k], enc::TAG) << endl;
            assert(argmax < aux_output_sizes[k]);
            if (argmax == aux_targets[i][k]){
                evaluator.good[k] ++;
            }else{
                all_good = false;
            }
        }
        if (all_good){
            evaluator.complete_match ++;
        }
    }
    //cerr << endl;
}

void BiRnnFeatureExtractor::assign_deplabels(vector<shared_ptr<Node>> &buffer, int deplabel_id){
    int task_id = deplabel_id - aux_start;
    for (int i = 0; i < buffer.size(); i++){
        Vec* output = auxiliary_output_nodes[i][task_id].back()->v();
        int argmax;
        output->maxCoeff(&argmax);
        //cerr << "Assigning label " << argmax << "   " << enc::hodor.decode(argmax, deplabel_id+1) << endl;
        buffer[i]->set_dlabel(argmax);
    }
}

void BiRnnFeatureExtractor::assign_tags(vector<shared_ptr<Node>> &buffer){
    for (int i = 0; i < buffer.size(); i++){
        Vec* output = auxiliary_output_nodes[i][0].back()->v();  // task 0 is necessarily tag
        int argmax;
        output->maxCoeff(&argmax);
//        if (argmax == enc::UNDEF || argmax == enc::UNKNOWN){
//            cerr << "Assigning " << argmax << "  as tag" << endl;
//            cerr << *buffer[i] << endl;
//        }
        buffer[i]->set_label(enc::hodor.code(enc::hodor.decode(argmax, enc::TAG), enc::CAT));
    }
}

void BiRnnFeatureExtractor::assign_morphological_features(vector<shared_ptr<Node>> &buffer, int deplabel_id){
    for (int i = 0; i < buffer.size(); i++){
        for (int task = 1; task < auxiliary_output_nodes[i].size(); task++){
            if (task != deplabel_id){
                Vec* output = auxiliary_output_nodes[i][task].back()->v();
                int argmax;
                output->maxCoeff(&argmax);
                buffer[i]->set_pred_field(task, argmax);
            }
        }
    }
}

void BiRnnFeatureExtractor::auxiliary_gradient_check(vector<shared_ptr<Node>> &buffer, double epsilon){
    cerr << "Gradient Checking auxiliary task" << endl;

    add_aux_graph(buffer);
    fprop_aux();
    bprop_aux();

    for (int i = 0; i < lu->size(); i++){
        (*lu)[i].get_active_params(auxiliary_parameters);
    }
    get_parameters(auxiliary_parameters);
//    if (params->rnn.char_rnn_feature_extractor){
//        char_rnn.get_parameters(auxiliary_parameters);
//    }
//    auxiliary_parameters.insert(auxiliary_parameters.end(), parameters.begin(), parameters.end());

    for (int i = 0; i < auxiliary_parameters.size(); i++){
        for (int k = 0; k < auxiliary_parameters[i]->size(); k++){
            auxiliary_parameters[i]->add_epsilon(k, epsilon);
            double a = full_fprop_aux(buffer);
            auxiliary_parameters[i]->add_epsilon(k, -epsilon);
            auxiliary_parameters[i]->add_epsilon(k, -epsilon);
            double c = full_fprop_aux(buffer);
            auxiliary_parameters[i]->add_epsilon(k, epsilon);
            auxiliary_parameters[i]->set_empirical_gradient(k, (a-c) / (2 * epsilon));
        }
        cerr << "p[" << i << "] -> " << std::flush;
        auxiliary_parameters[i]->print_gradient_differences();
    }

}

double BiRnnFeatureExtractor::aux_loss(){
    double loss = 0.0;
    for (int i = 0; i < auxiliary_output_nodes.size(); i++){
        for (int j = 0; j < auxiliary_output_nodes[i].size(); j++){
            Vec* v = auxiliary_output_nodes[i][j].back()->v();
            loss += - log((*v)[aux_targets[i][j]]);
        }
    }
    return loss;
}

double BiRnnFeatureExtractor::full_fprop_aux(vector<shared_ptr<Node>> &buffer){
    add_aux_graph(buffer);
    fprop_aux();
    return aux_loss();
}

void BiRnnFeatureExtractor::aux_reset_gradient_history(){
    for (int i = 0; i < auxiliary_parameters.size(); i++){
        auxiliary_parameters[i]->reset_gradient_history();
    }
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->reset_gradient_history();
    }
    for (int i = 0; i < lu->size(); i++){
        (*lu)[i].reset_gradient_history();
    }
    char_rnn.reset_gradient_history();
}
*/

void BiRnnFeatureExtractor::set_train_time(bool b){
    train_time = b;
}

//int BiRnnFeatureExtractor::n_aux_tasks(){
//    return aux_end - aux_start;
//}

