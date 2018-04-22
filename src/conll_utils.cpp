

#include "conll_utils.h"

const int Output::BASE = 100;

Output::Output(string s):
    code(s),
    xpos(false),
    morph(false),
    n_feats(0),
    n_chars(false),
    max_chars(0),
    bigram(false),
    trigram(false){
    initialize(s);
}

void Output::initialize(string s){
    if (s.find('x') != string::npos){
        xpos = true;
    }
    if (s.find('m') != string::npos){
        morph = true;
    }
    if (s.find('c') != string::npos){
        n_chars = true;
    }
    if (s.find('b') != string::npos){
        bigram = true;
    }
    if (s.find('t') != string::npos){
        trigram = true;
    }
}

void Output::get_output_sizes(){
    n_labels.clear();
    n_labels.push_back(enc::hodor.size(enc::UPOS));
    if (this->xpos){
        n_labels.push_back(enc::hodor.size(enc::XPOS));
    }
    if (this->morph){
        n_feats = enc::morph.size();
        for (int i = 0; i < enc::morph.size(); i++){
            n_labels.push_back(enc::morph.size(i));
        }
    }
    if (this->n_chars){
        assert(max_chars > 0);
        n_labels.push_back(max_chars / 3 + 1);
    }
    if (this->bigram){
        assert(bigrams.size() > 0);
        int max = 0;
        for (auto it = bigrams.begin(); it != bigrams.end(); ++it){
            if (it->second > max){
                max = it->second;
            }
        }
        assert(bigrams.size() == max);
        n_labels.push_back(max + 1);
    }
    if (this->trigram){
        assert(trigrams.size() > 0);
        int max = 0;
        for (auto it = trigrams.begin(); it != trigrams.end(); ++it){
            if (it->second > max){
                max = it->second;
            }
        }
        assert(trigrams.size() == max);
        n_labels.push_back(max + 1);
    }

    //    for (int i = 0; i < n_labels.size(); i ++){
//        cerr << "Output size " << i << "  " << n_labels[i] << endl;
//    }
}

void Output::export_model(string output_dir){
    ofstream out(output_dir + "/output_code");
    out << code << endl;
    out.close();
}

void Output::import_model(string output_dir){
    ifstream in(output_dir + "/output_code");
    in >> code;
    initialize(code);
    in.close();
}

void Output::update_bigrams(ConllTreebank &treebank){
    for (int i = 0; i < treebank.size(); i++){
        ConllTree *tree = treebank[i];
        for (int j = 0; j < tree->size(); j++){
            int first = 0;
            if (j > 0){
                first = (*tree)[j-1]->cpos();
            }
            int second = (*tree)[j]->cpos();

            int third = 0;
            if (j + 1 < tree->size()){
                third = (*tree)[j+1]->cpos();
            }

            int pair_id = first + second * BASE;
            if (bigrams.find(pair_id) == bigrams.end()){
                int id = bigrams.size() + 1;
                bigrams[pair_id] = id;
            }
            int triple_id = first + second * BASE + third * BASE * BASE;
            if (trigrams.find(triple_id) == trigrams.end()){
                int id = trigrams.size() + 1;
                trigrams[triple_id] = id;
            }
        }
    }
}


//ConllToken::ConllToken(int position, String _form, int _iform){
//    this->_position = position;
//    this->_form = _form;
//    this->_iform = _iform;
//}

ConllToken::ConllToken(int position, String _form, int _iform, int _cpos, int _fpos, vector<int> morpho){
    this->_position = position;
    this->_form = _form;
    this->_iform = _iform;
    this->_cpos = _cpos;
    this->_fpos = _fpos;
    this->_morpho = morpho;
}

int ConllToken::i(){
    return _position;
}

int ConllToken::form(){
    return _iform;
}

int ConllToken::cpos(){
    return _cpos;
}

int ConllToken::fpos(){
    return _fpos;
}

void ConllToken::cpos(int new_cpos){
    _cpos = new_cpos;
}

void ConllToken::fpos(int new_fpos){
    _fpos = new_fpos;
}

void ConllToken::print_morphology(ostream &os){
    if (! has_morpho()){
        os << "_\t";
        return;
    }
    vector<string> attributes;
    for (int i = 0; i < _morpho.size(); i++){
        if (_morpho[i] != enc::UNDEF && _morpho[i] != enc::UNKNOWN){
            string s = enc::morph.get_header(i) + "=" + enc::morph.decode_to_str(_morpho[i], i);
            attributes.push_back(s);
        }
    }
    os << attributes[0];
    for (int i = 1; i < attributes.size(); i++){
        os << "|" << attributes[i];
    }
    os << "\t";
}

bool ConllToken::has_morpho(){
    for (int i = 0; i < _morpho.size(); i++){
        //cerr << "mm" << _morpho[i] << endl;
        if (_morpho[i] != enc::UNDEF && _morpho[i] != enc::UNKNOWN){
            return true;
        }
    }
    return false;
}

int ConllToken::get_morpho(int type){
    if (_morpho.size() <= type){
        return enc::UNDEF;
    }
    return _morpho[type];
}

void ConllToken::set_morpho(int type, int val){
    while (_morpho.size() <= type){
        _morpho.push_back(enc::UNDEF);
    }
    _morpho[type] = val;
}

int ConllToken::len_form(){
    return _form.size();
}

ostream & operator<<(ostream &os, ConllToken &ct){
    os << ct.i() << "\t"
       << str::encode(ct._form) << "\t"
       << "_" << "\t"
       << enc::hodor.decode_to_str(ct._cpos, enc::UPOS) << "\t"
       << enc::hodor.decode_to_str(ct._fpos, enc::XPOS) << "\t";
    ct.print_morphology(os);
    os << "_" << "\t"  // head
       << "_" << "\t"  // rel
       << "_" << "\t"  // phead
       << "_" << "\t";  // prel
    return os;
}

ConllTree::ConllTree(vector<ConllToken> &tokens){
    this->tokens = tokens;
}

ConllToken* ConllTree::operator[](int i){
    assert(i >= 0 && i < tokens.size());
    return &tokens[i];
}

int ConllTree::size(){
    return tokens.size();
}

void ConllTree::to_training_example(vector<STRCODE> &X, vector<vector<int>> &Y, Output &output){
    X.clear();
    Y.clear();

    for (ConllToken &tok: tokens){
        X.push_back(tok.form());
        vector<int> label{tok.cpos()};
        if (output.xpos){
            label.push_back(tok.fpos());
        }
        if (output.morph){
            for (int i = 0; i < output.n_feats; i++){
                label.push_back(tok.get_morpho(i));
            }
        }
        if (output.n_chars){
            int size = tok.len_form() / 3 + 1;
            if (size >= output.max_chars / 3 + 1){
                size = 0;
            }
            label.push_back(size);
        }
        if (output.bigram){
            int second = tok.cpos();
            int first = 0;
            if (tok.i() -1 > 0){ // conll id starts at 1
                first = tokens[tok.i()-2].cpos();
            }
            int pair_id = first + second * Output::BASE;
            if (output.bigrams.find(pair_id) != output.bigrams.end()){
                label.push_back(output.bigrams[pair_id]);
            }else{
                label.push_back(0);
            }
        }
        if (output.trigram){
            int second = tok.cpos();
            int first = 0;
            if (tok.i() -1 > 0){ // conll id starts at 1
                first = tokens[tok.i()-2].cpos();
            }
            int third = 0;
            if (tok.i() < tokens.size()){
                third = tokens[tok.i()].cpos();
            }
            int triple_id = first + second * Output::BASE + third * Output::BASE * Output::BASE;
            if (output.trigrams.find(triple_id) != output.trigrams.end()){
                label.push_back(output.trigrams[triple_id]);
            }else{
                label.push_back(0);
            }
        }
        Y.push_back(label);
    }
    assert(X.size() > 0);
    assert(X.size() == Y.size());
}

void ConllTree::assign_tags(vector<vector<int>> &Y, Output &output){
    for (int i = 0; i < tokens.size(); i++){
        int k = 0;
        tokens[i].cpos(Y[i][k++]);
        if (output.xpos){
            tokens[i].fpos(Y[i][k++]);
        }
        if (output.morph){
//            cerr << "Y size " << Y[i].size() << endl;
//            cerr << "k " << k << endl;
//            cerr << enc::morph.size() << endl;
            for (int j = 0; j < enc::morph.size(); j++){
                //cerr << "Y size " << Y[i].size() << "  " << k << endl;
                assert(k < Y[i].size());
                tokens[i].set_morpho(j, Y[i][k++]);
            }
        }
    }
}

ostream & operator<<(ostream &os, ConllTree &ct){
    for (ConllToken & c: ct.tokens){
        os << c << endl;
    }
    return os;
}

ConllTreebank::ConllTreebank(){}
void ConllTreebank::add_tree(ConllTree &tree){
    trees.push_back(tree);
}
ConllTree* ConllTreebank::operator[](int i){
    return &trees[i];
}
int ConllTreebank::size(){
    return trees.size();
}

void ConllTreebank::update_vocsize_and_frequencies(){
    assert(frequencies.size() == 0);
    for (ConllTree &tree : trees){
        for (int i = 0; i < tree.size(); i++){
            ConllToken *token = tree[i];
            int form = token->form();
            if (frequencies.find(form) != frequencies.end()){
                frequencies[form] += 1;
            }else{
                frequencies[form] = 1;
            }
        }
    }
}

void ConllTreebank::shuffle(){
    std::shuffle(trees.begin(), trees.end(), rd::Random::re);
}

void ConllTreebank::subset(ConllTreebank &other, int n){
    for (int i = 0; i < n; i++){
        other.add_tree(trees[i]);
    }
}

ostream & operator<<(ostream &os, ConllTreebank &ct){
    for (ConllTree & c: ct.trees){
        os << c << endl;
    }
    return os;
}

void parse_morphology(String &s, vector<int> &morph, bool train){
    morph.clear();
    if (s == L"_"){
        return;
    }
    vector<int> keys;
    vector<int> values;
    int max_key = 0;

    vector<String> split_s;
    str::split(s, "|", "", split_s);
    for (int i = 0; i < split_s.size(); i++){
        vector<String> k_v;
        str::split(split_s[i], "=", "", k_v);
        assert(k_v.size() == 2);
        string type_str;
        str::encode(type_str, k_v[0]);

        int type_id = enc::morph.find_type_id(type_str, train);
        if (type_id == -1){ // If not a training corpus and feature is unknown -> ignore it
            continue;
        }
        int value = enc::morph.code(k_v[1], type_id);

        keys.push_back(type_id);
        values.push_back(value);

        if (type_id > max_key){
            max_key = type_id;
        }
    }
    morph = vector<int>(max_key + 1, enc::UNDEF);
    for (int i = 0; i < keys.size(); i++){
        morph[keys[i]] = values[i];
    }
}

void read_conll_corpus(std::string &filename,
                       ConllTreebank &treebank,
                       bool train){

    string type("word");
    enc::hodor.find_type_id(type, true);
    type = "tag";
    enc::hodor.find_type_id(type, true);
    type = "upos";
    enc::hodor.find_type_id(type, true);
    type = "xpos";
    enc::hodor.find_type_id(type, true);

    ifstream in(filename);
    string buffer;
    vector<ConllToken> tokens;

    while (getline(in, buffer)){
        if (buffer.size() == 0){
            if (tokens.size() > 0){
                ConllTree tree(tokens);
                treebank.add_tree(tree);
                tokens.clear();
            }
            continue;
        }
        if (buffer[0] == '#'){
            continue;
        }
        wstring wbuffer = str::decode(buffer);
        vector<wstring> split_tokens;
        str::split(wbuffer, "\t", "", split_tokens);
        assert(split_tokens.size() == 10);

        if (split_tokens[0].find(L"-") != std::string::npos){
            continue;
        }

        int id = stoi(split_tokens[ConllU::ID]);
        String form = split_tokens[ConllU::FORM];
        int iform = enc::hodor.code(form, enc::TOK);

        int cpos = enc::hodor.code(split_tokens[ConllU::UPOS], enc::UPOS);
        int fpos = enc::hodor.code(split_tokens[ConllU::XPOS], enc::XPOS);

        vector<int> morpho;
        parse_morphology(split_tokens[ConllU::FEATS], morpho, train);

        ConllToken tok(id, form, iform, cpos, fpos, morpho);
        tokens.push_back(tok);
    }
    in.close();
}
