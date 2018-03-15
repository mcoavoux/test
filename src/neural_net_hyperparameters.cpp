
#include "neural_net_hyperparameters.h"



NetTopology::NetTopology():n_hidden_layers(2), size_hidden_layers(16), embedding_size_type{8,8,8,8}{}

CharRnnParameters::CharRnnParameters():dim_char(16), dim_char_based_embeddings(32), crnn(0){}

RnnParameters::RnnParameters()
    : cell_type(RecurrentLayerWrapper::LSTM),
      depth(2),
      hidden_size(64),
      features(2),
      //char_rnn_feature_extractor(false),
      auxiliary_task(false),
      auxiliary_task_max_target(0){};


NeuralNetParameters::NeuralNetParameters():
    learning_rate(0.02),
    decrease_constant(1e-6),
    clip_value(10.0),
    gaussian_noise_eta(0.1),
    gaussian_noise(false),
    gradient_clipping(false),
    soft_clipping(false),
    rnn_feature_extractor(false),
    header{"word", "tag"}{}

void NeuralNetParameters::print(ostream &os){
    os << "learning rate\t"       << learning_rate << endl;
    os << "decrease constant\t"   << decrease_constant << endl;
    os << "gradient clipping\t"   << gradient_clipping << endl;
    os << "clip value\t"          << clip_value << endl;
    os << "gaussian noise\t"      << gaussian_noise << endl;
    os << "gaussian noise eta\t"  << gaussian_noise_eta << endl;
    os << "hidden layers\t"       << topology.n_hidden_layers << endl;
    os << "size hidden layers\t"  << topology.size_hidden_layers << endl;
    os << "embedding sizes\t";
    for (int &i : topology.embedding_size_type){
        os << " " << i;
    } os << endl;
    os << "bi-rnn\t" << rnn_feature_extractor << endl;
    os << "cell type\t" << rnn.cell_type << endl;
    os << "rnn depth\t" << rnn.depth << endl;
    os << "rnn state size\t" << rnn.hidden_size << endl;
    os << "number of token feature (rnn)\t" << rnn.features <<endl;
    os << "char rnn\t" << rnn.crnn.crnn << endl;
    os << "char embedding size\t" << rnn.crnn.dim_char << endl;
    os << "char based embedding size\t" << rnn.crnn.dim_char_based_embeddings << endl;
    os << "auxiliary task\t" << rnn.auxiliary_task << endl;
    os << "auxiliary task max idx\t" << rnn.auxiliary_task_max_target << endl;
    os << "voc sizes\t";
    for (int &i : voc_sizes){
        os << " " << i;
    } os << endl;
}

void NeuralNetParameters::read_option_file(const string &filename, NeuralNetParameters &p){
    enum {CHECK_VALUE, LEARNING_RATE, DECREASE_CONSTANT,
          GRADIENT_CLIPPING, CLIP_VALUE, GAUSSIAN_NOISE,
          HIDDEN_LAYERS, SIZE_HIDDEN, EMBEDDING_SIZE,
          BI_RNN,
          RNN_CELL_TYPE, RNN_DEPTH, RNN_STATE_SIZE, RNN_FEATURE,
          CHAR_BIRNN, CHAR_EMBEDDING_SIZE, CHAR_BASED_EMBEDDING_SIZE,
          GAUSSIAN_NOISE_ETA,
         AUX_TASK, AUX_TASK_IDX,
         VOC_SIZES};
    unordered_map<string,int> dictionary{
        {"learning rate", LEARNING_RATE},
        {"decrease constant", DECREASE_CONSTANT},
        {"gradient clipping", GRADIENT_CLIPPING},
        {"clip value", CLIP_VALUE},
        {"gaussian noise", GAUSSIAN_NOISE},
        {"hidden layers", HIDDEN_LAYERS},
        {"size hidden layers", SIZE_HIDDEN},
        {"embedding sizes", EMBEDDING_SIZE},
        {"bi-rnn", BI_RNN},
        {"cell type", RNN_CELL_TYPE},
        {"rnn depth", RNN_DEPTH},
        {"rnn state size", RNN_STATE_SIZE},
        {"number of token feature (rnn)", RNN_FEATURE},
        {"char rnn", CHAR_BIRNN},
        {"char embedding size", CHAR_EMBEDDING_SIZE},
        {"char based embedding size", CHAR_BASED_EMBEDDING_SIZE},
        {"gaussian noise eta", GAUSSIAN_NOISE_ETA},
        {"auxiliary task", AUX_TASK},
        {"auxiliary task max idx", AUX_TASK_IDX},
        {"voc sizes", VOC_SIZES}
    };
    ifstream is(filename);
    string buffer;
    vector<string> tokens;
    while (getline(is,buffer)){
        str::split(buffer, "\t", "", tokens);
        if (tokens.size() == 2){
            int id = dictionary[tokens[0]];
            assert(id != CHECK_VALUE);
            switch (id){
            case LEARNING_RATE:     p.learning_rate = stod(tokens[1]);              break;
            case DECREASE_CONSTANT: p.decrease_constant = stod(tokens[1]);          break;
//            case GRADIENT_CLIPPING: p.soft_clipping = stoi(tokens[1]);              break;
            case GRADIENT_CLIPPING: p.gradient_clipping = stoi(tokens[1]);          break;
            case CLIP_VALUE:        p.clip_value = stod(tokens[1]);                 break;
            case GAUSSIAN_NOISE:    p.gaussian_noise = stoi(tokens[1]);             break;
            case GAUSSIAN_NOISE_ETA:p.gaussian_noise_eta = stod(tokens[1]);         break;
            case HIDDEN_LAYERS:     p.topology.n_hidden_layers = stoi(tokens[1]);   break;
            case SIZE_HIDDEN:       p.topology.size_hidden_layers = stoi(tokens[1]);break;
            case EMBEDDING_SIZE:{
                vector<string> sizes;
                str::split(tokens[1], " ", "", sizes);
                p.topology.embedding_size_type.clear();
                for (string &s : sizes){
                    p.topology.embedding_size_type.push_back(stoi(s));
                }
                break;
            }
            case VOC_SIZES:{
                vector<string> sizes;
                str::split(tokens[1], " ", "", sizes);
                p.voc_sizes.clear();
                for (string &s : sizes){
                    p.voc_sizes.push_back(stoi(s));
                }
                break;
            }
            case BI_RNN: p.rnn_feature_extractor = stoi(tokens[1]);     break;
            case RNN_CELL_TYPE: p.rnn.cell_type = stoi(tokens[1]);      break;
            case RNN_DEPTH: p.rnn.depth = stoi(tokens[1]);              break;
            case RNN_STATE_SIZE: p.rnn.hidden_size = stoi(tokens[1]);   break;
            case RNN_FEATURE: p.rnn.features = stoi(tokens[1]);         break;
            case CHAR_BIRNN: p.rnn.crnn.crnn = stoi(tokens[1]);         break;
            case CHAR_EMBEDDING_SIZE: p.rnn.crnn.dim_char = stoi(tokens[1]);      break;
            case CHAR_BASED_EMBEDDING_SIZE: p.rnn.crnn.dim_char_based_embeddings = stoi(tokens[1]); break;
            case AUX_TASK: p.rnn.auxiliary_task = stoi(tokens[1]);                break;
            case AUX_TASK_IDX: p.rnn.auxiliary_task_max_target = stoi(tokens[1]); break;
            default: assert(false && "Unknown nn option");
            }
        }else{
            cerr << "Unknown nn option or problematic formatting : " << buffer << endl;
        }
    }
}



