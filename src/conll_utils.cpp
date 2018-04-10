

#include "conll_utils.h"

//ConllToken::ConllToken(int position, String _form, int _iform){
//    this->_position = position;
//    this->_form = _form;
//    this->_iform = _iform;
//}

ConllToken::ConllToken(int position, String _form, int _iform, int _cpos, int _fpos){
    this->_position = position;
    this->_form = _form;
    this->_iform = _iform;
    this->_cpos = _cpos;
    this->_fpos = _fpos;
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


ostream & operator<<(ostream &os, ConllToken &ct){
    os << ct.i() << "\t"
       << str::encode(ct._form) << "\t"
       << "_" << "\t"
       << enc::hodor.decode_to_str(ct._cpos, enc::TAG) << "\t"
       << enc::hodor.decode_to_str(ct._fpos, enc::TAG) << "\t"
       << "_" << "\t"  // morph
       << "_" << "\t"  // head
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

void ConllTree::to_training_example(vector<STRCODE> &X, vector<vector<int>> &Y){
    X.clear();
    Y.clear();
    for (ConllToken &tok: tokens){
        X.push_back(tok.form());
        vector<int> label{tok.cpos()};
        Y.push_back(label);
    }
    assert(X.size() > 0);
    assert(X.size() == Y.size());
}

void ConllTree::assign_tags(vector<vector<int>> &Y){
    for (int i = 0; i < tokens.size(); i++){
        tokens[i].cpos(Y[i][0]);
        tokens[i].fpos(Y[i][0]);
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


void read_conll_corpus(std::string &filename,
                       ConllTreebank &treebank,
                       bool train){

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


        int id = stoi(split_tokens[0]);
        String form = split_tokens[1];
        int iform = enc::hodor.code(form, enc::TOK);

        int cpos = enc::hodor.code(split_tokens[3], enc::TAG);

        ConllToken tok(id, form, iform, cpos, enc::UNKNOWN);
        tokens.push_back(tok);
    }
    in.close();
}
