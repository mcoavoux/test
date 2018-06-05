#include <getopt.h>
#include <sys/stat.h>
#include <unordered_map>
#include <boost/functional/hash.hpp>

#include "conll_utils.h"
#include "bilstm_tagger.h"
#include "utils.h"
#include "neural_net_hyperparameters.h"

using std::pair;
using std::make_pair;
using namespace std;

struct EpochEval{
    Output output;
    float total;
    vector<float> a;
    vector<float> l;

    unordered_map<pair<int, int>, int, boost::hash<pair<int, int>>> confusion;

    EpochEval(Output &o): output(o), total(0.0){
        int size = 1;
        if (o.xpos){
            size ++;
        }
        if (o.morph){
            size ++;
        }
        a = vector<float>(size, 0.0);
        l = vector<float>(size, 0.0);
    }

    void update_losses(vector<float> &losses){
        int id = 0;
        int k = 0;
        l[id++] += losses[k++];
        if (output.xpos){
            l[id++] += losses[k++];
        }
        if (output.morph){
            for (int i = k; i < k + output.n_feats; i++){
                l[id] += losses[i];
            }
            id ++;
            k += output.n_feats;
        }
        //assert(id == a.size());
        //assert(k == losses.size());
    }

    void update_confusion(int g, int p){
        if (p > g){
            int tmp = p;
            p = g;
            g = tmp;
        }
        if (confusion.find(make_pair(g, p)) == confusion.end()){
            confusion[make_pair(g, p)] = 1;
        }else{
            confusion[make_pair(g, p)] += 1;
        }
    }

    Pair most_frequent_error(){
        int num_errors = 0;
        Pair p(-1, -1);
        for (auto &it : confusion){
            if (it.second > num_errors){
                num_errors = it.second;
                p.first = it.first.first;
                p.second = it.first.second;
            }
        }
        return p;
    }

    void update(vector<int> &gold, vector<int> &pred){
        assert(gold.size() == pred.size());
        total += 1;
        int id = 0;
        int k = 0;
        if (gold[k] == pred[k]){
            a[id] += 1;
        }else{
            update_confusion(gold[k], pred[k]);
        }
        id++;
        k++;
        if (output.xpos){
            if (gold[k] == pred[k]){
                a[id] += 1;
            }
            id++;
            k++;
        }
        if (output.morph){
            bool correct = true;
            for (int i = k; i < k + output.n_feats; i++){
                if (gold[i] != pred[i]){
                    correct = false;
                }
            }
            if (correct){
                a[id] += 1;
            }
            id++;
            k += output.n_feats;
        }
        //assert(id == a.size());
        //assert(k == pred.size());
    }
    int size(){
        return a.size();
    }
    float get_acc(int i){
        return a[i] / total * 100.0;
    }
    float get_loss(int i){
        return l[i] / total;
    }

    friend ostream& operator<<(ostream &os, EpochEval &ev){
        os << "a: ";
        for (int i = 0; i < ev.size(); i ++){
            os << std::setprecision(4) << ev.get_acc(i) << " ";
        }
        os << "l: ";
        for (int i = 0; i < ev.size(); i ++){
            os << std::setprecision(4) << ev.get_loss(i) << " ";
        }
        return os;
    }
};

struct EpochSummary{
    int epoch;
    EpochEval train;
    EpochEval dev;

    EpochSummary(int e,
                 EpochEval &train,
                 EpochEval &dev):epoch(e),
                                 train(train),
                                 dev(dev){}

    void print(ostream &os){
        os << "Epoch " << epoch
           << " train " << train
           << " dev " << dev << endl;
    }

    void log(ostream &os){
        os << epoch;
        for (int i = 0; i < train.size(); i++){
            os << "\t" << train.get_acc(i);
        }
        for (int i = 0; i < train.size(); i++){
            os << "\t" << train.get_loss(i);
        }
        for (int i = 0; i < dev.size(); i++){
            os << "\t" << dev.get_acc(i);
        }
        for (int i = 0; i < dev.size(); i++){
            os << "\t" << dev.get_loss(i);
        }
    }
};




struct Options{
    enum {TRAIN, TEST};
    string train_file;
    string dev_file;
    string test_file;
    string hyper_file;
    string output_dir = "mymodel";
    int epochs = 20;
    NeuralNetParameters params;
    int mode = 0;

    void assign_mode(char c_str[]){
        string mode_str(c_str);
        if (mode_str == "train"){
            cerr << "Mode = " << mode_str << endl;
            mode = TRAIN;
            return;
        }
        if (mode_str == "test"){
            cerr << "Mode = " << mode_str << endl;
            mode = TEST;
            return;
        }
        cerr << "Unknown argument for -m / --mode option" << endl;
        cerr << "Accepted arguments: 'train' or 'test'" << endl;
        exit(1);
    }
    bool check(){
        if (mode == TRAIN){
            if (train_file.empty()){
                cerr << "Please specify --train option" << endl;
                return false;
            }
            if (train_file.empty()){
                cerr << "Please specify --dev option" << endl;
                return false;
            }
            if (hyper_file.empty()){
                cerr << "Please specify --hyperparameters option" << endl;
                return false;
            }
            return true;
        }else{
            if (test_file.empty()){
                cerr << "Please specify --test file" << endl;
                return false;
            }
            if (output_dir.empty()){
                cerr << "Please specify --test file" << endl;
                return false;
            }
            return true;
        }
    }
};

void print_help(){
    cout << endl << endl <<"Yet another neural tagger. "<< endl << endl <<

        "Usage:" << endl <<
        "      ./main train -t <trainfile> - d <devfile> -i <epochs> -o <outputdir> [options]" << endl <<
        "      ./main test -T <testfile> -l <model> [options]" << endl << endl <<
        "Options:" << endl <<
        "  -h     --help                        displays this message and quits" << endl <<
        "  -m     --mode            [STRING]    train|test" << endl <<
        "Training mode options:" << endl <<
        "  -t     --train           [STRING]    training corpus (conll format)   " << endl <<
        "  -d     --dev             [STRING]    developpement corpus (conll format)   " << endl <<
        "  -i     --epochs          [INT]       number of iterations [default=20]" << endl <<
        "  -o     --output          [STRING]    output directory" << endl <<
        "  -p     --hyperparameters [STRING]    hyperparameters of neural net" << endl <<
        "  -M     --multitask       [STRING]    specify what to predict: xm" << endl <<
        "Testing mode options:" << endl <<
        "  -T     --test           [STRING]    training corpus (conll format)   " << endl <<
        "  -l     --load-model      [STRING]    model directory" << endl << endl;
}

void evaluate(shared_ptr<BiLstmTagger> tagger, Output &output, ConllTreebank &tbk, EpochEval &eval){
    vector<float> losses(output.n_labels.size(), 0.0);
    vector<STRCODE> X;
    vector<vector<int>> Y;
    vector<vector<int>> predictions;
    for (int i = 0; i < tbk.size(); i++){
        tbk[i]->to_training_example(X, Y, output);
        tagger->eval_one(X, Y, predictions, losses);
        eval.update_losses(losses);
        assert(Y.size() == predictions.size());
        for (int j = 0; j < Y.size(); j++){
            assert(Y[j].size() == predictions[j].size());
            eval.update(Y[j], predictions[j]);
        }
    }
}

int main(int argc, char *argv[]){
    srand(rd::Random::SEED);

    Options options;
    Output output("m");

    while(true){
        static struct option long_options[] ={
        {"help",no_argument,0,'h'},
        {"mode", required_argument, 0, 'm'},
        {"train", required_argument, 0, 't'},
        {"test", required_argument, 0, 'T'},
        {"dev", required_argument, 0, 'd'},
        {"epochs", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"load-model", required_argument, 0, 'l'},
        {"hyperparameters", required_argument, 0, 'p'},
        {"multitask", required_argument, 0, 'M'}};

        int option_index = 0;

        char c = getopt_long (argc, argv, "ht:T:d:i:o:p:m:l:M:",long_options, &option_index);

        if(c==-1){
            break;
        }

        switch(c){
        case 'h': print_help(); exit(0);
        case 'm': options.assign_mode(optarg);    break;
        case 't': options.train_file = optarg;    break;
        case 'T': options.test_file = optarg;     break;
        case 'd': options.dev_file = optarg;      break;
        case 'i': options.epochs = atoi(optarg);  break;
        case 'o': options.output_dir = optarg;    break;
        case 'l': options.output_dir = optarg;    break;
        case 'p': options.hyper_file = optarg;    break;
        case 'M': output = Output(optarg);        break;
        default:
            cerr << "unknown option: " << optarg << endl;
            print_help();
            exit(0);
        }
    }

    if (options.mode == Options::TRAIN){

        mkdir(options.output_dir.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

        ConllTreebank train;
        ConllTreebank dev;

        if (! options.check()){
            exit(1);
        }

        NeuralNetParameters::read_option_file(options.hyper_file, options.params);

        read_conll_corpus(options.train_file, train, true);
        read_conll_corpus(options.dev_file, dev, false);

        output.update_bigrams(train);

        train.update_vocsize_and_frequencies();
        enc::hodor.update_wordform_frequencies(train.get_frequencies_dict());

        int voc_size = enc::hodor.size(enc::TOK);
        int longest = enc::hodor.longest_size(enc::TOK) + 1;

        output.max_chars = longest;
        output.get_output_sizes();

        cout << "Hyperparameters" << endl;
        options.params.print(cout);
        cout << endl;

        ConllTreebank train_sample;
        train.shuffle();
        train.subset(train_sample, dev.size());

        BiLstmTagger tagger(voc_size, output.n_labels, options.params);

//        vector<shared_ptr<BiLstmTagger>> models;
//        vector<float> dev_accuracies;

        ofstream log_file(options.output_dir + "/logger");


        float training_accuracy = 0.0;

        float best_dev_acc = 0.0;
        for (int epoch = 0; epoch < options.epochs || training_accuracy <= 99.6; epoch ++){

            train.shuffle();

            vector<STRCODE> X;
            vector<vector<int>> Y;
            for (int i = 0; i < train.size(); i++){
                train[i]->to_training_example(X, Y, output);
                tagger.train_one(X, Y);
                cerr << "\r" << std::setprecision(4) << (i*100.0 / train.size()) << "%";
            }

            shared_ptr<BiLstmTagger> avg_t(tagger.copy());
            avg_t->average_parameters();

            EpochEval eval_train(output);
            evaluate(avg_t, output, train_sample, eval_train);

            training_accuracy = eval_train.get_acc(0);

            EpochEval eval_dev(output);
            evaluate(avg_t, output, dev, eval_dev);
            EpochSummary sum(epoch, eval_train, eval_dev);

            cerr << "\r";
            sum.print(cout);

            sum.log(log_file);

            float dev_acc = eval_dev.get_acc(0);
//            models.push_back(avg_t);
//            dev_accuracies.push_back(dev_acc);

            if (dev_acc >= best_dev_acc){
                best_dev_acc = dev_acc;
                output.export_model(options.output_dir);
                avg_t->export_model(options.output_dir);
                ofstream outfile(options.output_dir + "/best_epoch");
                outfile << epoch << endl;
                outfile.close();
            }

            if (output.experts){
                Pair p = eval_dev.most_frequent_error();
                if(output.add_expert(p.first, p.second)){
                    tagger.add_expert_classifier();
                }
            }

            if (epoch > 100){
                break;
            }
        }

        log_file.close();

//        int argmax = 0;
//        for (int i = 0; i < dev_accuracies.size(); i++){
//            if (dev_accuracies[i] >= dev_accuracies[argmax]){
//                argmax = i;
//            }
//        }
//        models[argmax]->export_model(options.output_dir);
//        output.export_model(options.output_dir);

    }else{
        assert(options.mode == Options::TEST);


        enc::import_encoders(options.output_dir);
        int voc_size = enc::hodor.size(enc::TOK);

        output.import_model(options.output_dir);

        NeuralNetParameters::read_option_file(options.output_dir + "/hyperparameters", options.params);
        cerr << "Hyperparameters" << endl;
        options.params.print(cerr);
        cerr << endl;

        output.get_output_sizes();
        BiLstmTagger tagger(voc_size, output.n_labels, options.params);
        tagger.import_model(options.output_dir);

        ConllTreebank test;
        read_conll_corpus(options.test_file, test, false);

        for (int i = 0; i < test.size(); i++){
            vector<STRCODE> X;
            vector<vector<int>> gold;
            vector<vector<int>> pred;
            test[i]->to_training_example(X, gold, output);
            tagger.predict_one(X, pred);
            test[i]->assign_tags(pred, output);
        }
        cout << test;
    }
}
