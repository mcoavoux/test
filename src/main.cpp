#include <getopt.h>
#include <sys/stat.h>

#include "conll_utils.h"
#include "bilstm_tagger.h"
#include "utils.h"
#include "neural_net_hyperparameters.h"


using namespace std;

struct EpochSummary{
    int epoch;
    float taccuracy;
    float tloss;
    float daccuracy;
    float dloss;

    EpochSummary(int e, float tacc, float tloss, float dacc, float dloss):
        epoch(e), taccuracy(tacc), tloss(tloss), daccuracy(dacc), dloss(dloss){}

    void print(ostream &os){
        os << "\rEpoch " << epoch
           << " train a=" << taccuracy << " l=" << tloss
           << " dev a=" << daccuracy << " l=" << dloss
           << endl;
    }

    void log(ostream &os){
        os << epoch
           << "\t" << taccuracy
           << "\t" << tloss
           << "\t" << daccuracy
           << "\t" << dloss << endl;
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
        "Testing mode options:" << endl <<
        "  -T     --test           [STRING]    training corpus (conll format)   " << endl <<
        "  -l     --load-model      [STRING]    model directory" << endl << endl;
}

int main(int argc, char *argv[]){
    srand(rd::Random::SEED);

    Options options;

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
        {"hyperparameters", required_argument, 0, 'p'},};

        int option_index = 0;

        char c = getopt_long (argc, argv, "ht:T:d:i:o:p:m:l:",long_options, &option_index);

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

        int voc_size = enc::hodor.size(enc::TOK);
        vector<int> n_labels{enc::hodor.size(enc::TAG)};

        cout << "Hyperparameters" << endl;
        options.params.print(cout);
        cout << endl;

        ConllTreebank train_sample;
        train.shuffle();
        train.subset(train_sample, dev.size());

        BiLstmTagger tagger(voc_size, n_labels, options.params);

        vector<shared_ptr<BiLstmTagger>> models;
        vector<float> dev_accuracies;

        for (int epoch = 0; epoch < options.epochs; epoch ++){

            train.shuffle();

            vector<STRCODE> X;
            vector<vector<int>> Y;
            for (int i = 0; i < train.size(); i++){
                train[i]->to_training_example(X, Y);
                tagger.train_one(X, Y);
                cerr << "\r" << std::setprecision(4) << (i*100.0 / train.size()) << "%";
            }

            shared_ptr<BiLstmTagger> avg_t(tagger.copy());
            avg_t->average_parameters();

            float ttotal = 0;
            vector<float> tlosses(1);
            vector<int> taccuracies(1);

            for (int i = 0; i < train_sample.size(); i++){
                train_sample[i]->to_training_example(X, Y);
                ttotal += X.size();
                avg_t->eval_one(X, Y, tlosses, taccuracies);
            }

            float dtotal = 0;
            vector<float> dlosses(1);
            vector<int> daccuracies(1);

            for (int i = 0; i < dev.size(); i++){
                dev[i]->to_training_example(X, Y);
                dtotal += X.size();
                avg_t->eval_one(X, Y, dlosses, daccuracies);
            }

            EpochSummary sum(epoch, taccuracies[0]/ttotal, tlosses[0]/ttotal,
                    daccuracies[0]/dtotal, dlosses[0]/dtotal);

            sum.print(cout);

            models.push_back(avg_t);
            dev_accuracies.push_back(daccuracies[0] / dtotal);
        }
        int argmax = 0;
        for (int i = 0; i < dev_accuracies.size(); i++){
            if (dev_accuracies[i] > dev_accuracies[argmax]){
                argmax = i;
            }
        }
        models[argmax]->export_model(options.output_dir);

    }else{
        assert(options.mode == Options::TEST);

        enc::hodor.import_model(options.output_dir);
        int voc_size = enc::hodor.size(enc::TOK);
        vector<int> n_labels{enc::hodor.size(enc::TAG)};

        NeuralNetParameters::read_option_file(options.output_dir + "/hyperparameters", options.params);
        cerr << "Hyperparameters" << endl;
        options.params.print(cerr);
        cerr << endl;

        BiLstmTagger tagger(voc_size, n_labels, options.params);
        tagger.import_model(options.output_dir);

        string fooout("redux");
        tagger.export_model(fooout);

        ConllTreebank test;
        read_conll_corpus(options.test_file, test, false);

        for (int i = 0; i < test.size(); i++){
            vector<STRCODE> X;
            vector<vector<int>> gold;
            vector<vector<int>> pred;
            test[i]->to_training_example(X, gold);
            tagger.predict_one(X, pred);
            test[i]->assign_tags(pred);
        }
        cout << test;
    }
}
