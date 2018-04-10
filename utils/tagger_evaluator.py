

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

def evaluate(gold, pred) :

    cpos = 0.0
    fpos = 0.0
    #uas = 0.0
    #las = 0.0

    tot = 0.0

    for sg, ss in zip(gold, pred) :

        for tg, ts in zip(sg, ss) :
            if tg[FORM] != ts[FORM] :
                sys.stderr.write("{} {}\n".format(tg[FORM], ts[FORM]))
            assert(tg[FORM] == ts[FORM])

            if tg[CPOS] == ts[CPOS] :
                cpos += 1
            if tg[FPOS] == ts[FPOS] :
                fpos += 1
            tot += 1
    
    cpos = round(cpos / tot * 100, 2)
    fpos = round(fpos / tot * 100, 2)
    
    print("{}\t{}".format(cpos, fpos))

if __name__ == "__main__" :
        import sys
        import argparse

        usage = """
        Evaluate POS tagging (TODO: morpho, las)
        """

        parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument("gold", type = str, help="gold conll corpus")
        parser.add_argument("pred", type = str, help="pred conll corpus)")

        args = parser.parse_args()

        gold = read_conll(args.gold)
        pred = read_conll(args.pred)

        evaluate(gold, pred)


