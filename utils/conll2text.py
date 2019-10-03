
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

def main():
    
    import argparse
    
    usage = """Converts a COnllU corpus to raw text (1 sentence per line, tokens separated by spaces"""
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input", help="Input (conllu)")
    parser.add_argument("output", help="Output")

    args = parser.parse_args()
    
    corpus = read_conll(args.input)
    
    out = open(args.output, "w")
    
    for s in corpus:
        tokens = [t[FORM] for t in s]
        out.write(" ".join(tokens))
        out.write("\n")
    out.close()


if __name__ == '__main__':

    main()

