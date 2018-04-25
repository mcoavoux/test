
from joblib import Parallel, delayed
import os
import subprocess
from itertools import product
import glob

hyper_templates="""learning rate	{lr}
decrease constant	{dc}
gradient clipping	1
clip value	{cv}
gaussian noise	1
gaussian noise eta	{ge}
hidden layers	{hl}
size hidden layers	{hs}
embedding sizes	 {w} 16 16
cell type	2
rnn depth	{depth}
rnn state size	{W}
number of token feature (rnn)	1
char rnn	1
char embedding size	{c}
char based embedding size	{C}
"""

#bi-rnn	1
#auxiliary task	1
#auxiliary task max idx	20

def print_hyperparameters(filename, parameters):
    out = open(filename, "w")
    out.write(hyper_templates.format(**parameters))
    out.close()

def generate_options(args):
    
    languages = [line.strip().split() for line in open(args.languages)]

    for lang, depth, hs, w, W, c, C, lr, dc, cv, ge, hl in product(languages,
                        args.depth_rnn,
                        args.dim_hidden,
                        args.dim_word,
                        args.dim_wrnn,
                        args.dim_char,
                        args.dim_crnn,
                        args.learning_rate,
                        args.decrease_constant,
                        args.hard_clipping,
                        args.gaussian_noise,
                        args.hidden_layers):
        yield {"args" : args,
                "lang" : lang[0],
                "multi": lang[1],
                "depth" : depth,
                "hs": hs,
                "w" : w,
                "W" : W,
                "c" : c,
                "C" : C,
                "lr" : lr,
                "dc" : dc,
                "cv" : cv,
                "ge" : ge,
                "hl" : hl}

def unix(command) :
    print(command)
    subprocess.call([command], shell=True)

def get_corpus_filenames(datadir, lang_id):
    
    files = glob.glob("{}/*/{}-ud-*.conllu".format(datadir, lang_id))
    
    if len(files) == 3:
        train = [x for x in files if "train" in x]
        dev = [x for x in files if "dev" in x]
        test = [x for x in files if "test" in x]
        
        assert(len(train) == 1)
        assert(len(dev) == 1)
        assert(len(test) == 1)
        
        return train[0], dev[0], test[0]
    else:
        print(files)
        exit()

def signature(params):
    p = {k : v for k, v in params.items() if k != "args"}
    return "_".join(["_".join([k,str(v)]) for k,v in sorted(p.items())])

def do_experiment(param):
    args = param["args"]
    
    train, dev, test = get_corpus_filenames(args.data, param["lang"])
    
    modeldir = "{}/{}/{}".format(args.output, param["lang"], signature(param))
    
    unix("mkdir -p {}".format(modeldir))
    
    hyperfile = "{}/hyperparameters_orig".format(modeldir)
    print_hyperparameters(hyperfile, param)
    
    train_command_line = '../bin/main -m train -t {t} -d {d} -p {hyp} -i {i} -o {modelname} -M {multi} > {modelname}/log.txt'
    train_command_line = train_command_line.format(t=train,
                                                   d=dev,
                                                   hyp=hyperfile,
                                                   i=args.iterations,
                                                   modelname=modeldir,
                                                   multi=param["multi"] + args.multitask)
    
    unix(train_command_line)
    
    
    test_command_line = '../bin/main -m test -T {corpus} -l {model}/ > {model}/pred_test'.format(corpus=test, model=modeldir)
    unix(test_command_line)
    
    eval_command_line="python3 tagger_evaluator.py {corpus} {model}/pred_test > {model}/eval_test".format(corpus=test, model=modeldir)
    unix(eval_command_line)
    
    
    test_command_line = '../bin/main -m test -T {corpus} -l {model}/ > {model}/pred_dev'.format(corpus=dev, model=modeldir)
    unix(test_command_line)
    
    eval_command_line="python3 tagger_evaluator.py {corpus} {model}/pred_dev > {model}/eval_dev".format(corpus=dev, model=modeldir)
    unix(eval_command_line)
    


def main(args):
    os.makedirs(args.output, exist_ok=True)
    Parallel(n_jobs=args.threads)(delayed(do_experiment)(p) for p in generate_options(args))


if __name__ == "__main__":
    
    import argparse
    
    usage="""Launch parallel multiple experiments."""
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("data", type=str, help="dataset")
    parser.add_argument("output", type=str, help="output folder")
    parser.add_argument("languages", type=str, help="Language list, with list of attributes to predict")

    parser.add_argument("--iterations", "-i", type=int, default=20, help="Number of iterations per experiment")
    parser.add_argument("--threads", "-N", type=int, default=1, help="Max number of experiments in parallel")
    parser.add_argument("--hidden-layers", "-L", type=int, nargs="+", default=[0], help="Number of hidden layers")
    parser.add_argument("--dim-hidden", "-H", type=int, nargs="+", default=[32], help="Size of hidden layers")
    parser.add_argument("--dim-word","-w", type=int, nargs="+", default=[16], help="Dimension of word embeddings")
    parser.add_argument("--dim-wrnn","-W", type=int, nargs="+", default=[32], help="Dimension of word lstm")
    parser.add_argument("--dim-char","-c", type=int, nargs="+", default=[16], help="Dimension of char embeddings")
    parser.add_argument("--dim-crnn","-C", type=int, nargs="+", default=[16], help="Dimension of char lstm")
    parser.add_argument("--depth-rnn","-D", type=int, nargs="+", default=[4], help="Depth of RNN")

    parser.add_argument("--learning-rate", "-l", type=float, nargs="+", default=[0.02])
    parser.add_argument("--decrease-constant", "-d", type=float, nargs="+", default=[1e-6])
    parser.add_argument("--hard-clipping", "-g", type=float, nargs="+", default=[10])
    parser.add_argument("--gaussian-noise", "-G", type=float, nargs="+", default=[0.01])
    
    parser.add_argument("--multitask", "-M", type=str, default="", help="Add: e(xpert), b(igram left), B(igram right), s(kip-gram)")
    
    args = parser.parse_args()
    
    main(args)














