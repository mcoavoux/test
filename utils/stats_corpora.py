from collections import defaultdict

ID,FORM,LEMMA,CPOS,FPOS,MORPH,HEAD,DEPREL,PHEAD,PDEPREL=range(10)

def read_conll(filename) :
    with open(filename) as f :
        sentences = [[line.split("\t") for line in sen.split("\n") if line and line[0] != "#"] for sen in f.read().split("\n\n") if sen.strip()]

    for i in range(len(sentences)):
        sentences[i] = [t for t in sentences[i] if "-" not in t[ID]]
        s = sentences[i]
        #for tok in s :
            #tok[ID] = int(tok[ID])
            #tok[HEAD] = int(tok[HEAD])
    return sentences

def eval_morph(g, p):
    if g == p:
        return True
    if "_" in {g, p}:
        return False
    d = dict([kv.split("=") for kv in g.split("|")])
    p = dict([kv.split("=") for kv in p.split("|")])
    return p == d

def evaluate(gold, pred) :
    
    confusion = defaultdict(lambda:defaultdict(int))
    
    cpos = 0.0
    fpos = 0.0
    morph = 0.0
    #uas = 0.0
    #las = 0.0

    tot = 0.0

    for sg, ss in zip(gold, pred) :

        for tg, ts in zip(sg, ss) :
            if tg[FORM] != ts[FORM] :
                sys.stderr.write("{} {}\n".format(tg[FORM], ts[FORM]))
            assert(tg[FORM] == ts[FORM])
            
            confusion[tg[CPOS]][ts[CPOS]] += 1
            
            if tg[CPOS] == ts[CPOS] :
                cpos += 1
            if tg[FPOS] == ts[FPOS] :
                fpos += 1
            
            if eval_morph(tg[MORPH], ts[MORPH]):
                morph += 1
            tot += 1
    
    cpos = round(cpos / tot * 100, 2)
    fpos = round(fpos / tot * 100, 2)
    morph = round(morph / tot * 100, 2)
    
    print("{}\t{}\t{}".format(cpos, fpos, morph))
    
    labels = sorted(confusion)
    print("\t".join([""] + labels))
    for l1 in labels:
        d = confusion[l1]
        l = [l1] + [str(d[l2]) for l2 in labels]
        print("\t".join(l))
    print()
    
    print("\t".join([""] + labels))
    for l1 in labels:
        d = confusion[l1]
        summ = sum(d.values())
        l = [round(d[l2] / summ * 100, 2) for l2 in labels]
        l = [l1] + [str(i) for i in l]
        print("\t".join(l))

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

def get_morph_vocabulary(corpus):
    voc = {}
    for sentence in  corpus:
        for token in sentence:
            morph = token[MORPH]
            voc.add(morph)
    return voc

def main(args):
    
    pred_dirs = glob.glob("{}/*".format(args.expe))
    
    for pred_dir in pred_dirs:
        lang = pred_dir.split("/")[-1].strip("/")
        print(lang)
        
        train, dev, _ = get_corpus_filenames(args.data, lang)
        
        pred_dev = glob.glob("{}/*/pred_dev".format(pred_dir))
        assert(len(pred_dev) == 1)
        pred_dev = pred_dev[0]

        train_corpus = read_conll(train)
        pred_dev_corpus = read_conll(pred_dev)

        train_voc = get_morph_vocabulary(train_corpus)
        pred_dev_voc = get_morph_vocabulary(pred_dev_corpus)
        
        print("Lang:{}".format(lang))
        print("train vocsize\t{}".format(len(train_voc)))
        print("pred_dev vocsize\t{}".format(len(pred_dev_voc)))
        print("predicted tags unknown in train")
        for e in pred_dev_voc:
            if e not in train_voc:
                print(e)
        print()

if __name__ == "__main__":
    
    import argparse
    
    usage="""Launch parallel multiple experiments."""
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("data", type=str, help="dataset")
    parser.add_argument("expe", type=str, help="experiment dir")
    parser.add_argument("languages", type=str, help="Language list")

    args = parser.parse_args()
    
    main(args)














