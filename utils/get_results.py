
import os
import glob


def get_accuracy(model):
    
    dev = "{}/eval_dev".format(model)
    test = "{}/eval_test".format(model)
    
    if os.path.isfile(dev) and os.path.isfile(test):
        
        dev_acc = float(open(dev).readline().split()[0])
        test_acc = float(open(test).readline().split()[0])
        
        return dev_acc, test_acc
    return None

def get_results(expedir):
    
    result = {}
    
    for lang in os.listdir(expedir):
        path = os.path.join(expedir, lang)
        
        best = 0
        
        for model in os.listdir(path):
            full_path = os.path.join(path, model)
            
            
            acc = get_accuracy(full_path)
            
            if acc is not None:
                dev, test = acc
                
                if dev > best:
                    result[lang] = [dev, test, model]

    return result


def get_bilty_results():
    instream = open("results_bilty")
    res = {}
    for line in instream:
        if " | " in line and "Lang" not in line and ":|" not in line:
            sline = line.replace("|", "").split()
            assert(len(sline) == 3)
            
            n1, n2 = "-", "-"
            if sline[1] != "--":
                n1 = float(sline[1])
            if sline[2] != "--":
                n2 = float(sline[2])
            res[sline[0]] = [n1, n2]
    return res

def main(args):
    
    results = get_results(args.expedir)
    
    bilty_results = get_bilty_results()
    
    header = ["lang", "dev", "test", "bilty", "bilty+polyglot", "delta", "model"]
    
    print("\t".join(header))
    for lang in sorted(results):
        
        lres = results[lang]
        bl = bilty_results[lang]
        
        diff = lres[1] - max([i for i in bl if type(i) == float])
        
        l = [lang] +  lres[:2] + bl + [diff] + [lres[-1]]
        
        print("\t".join(map(str, l)))
    

if __name__ == "__main__":
    
    import argparse
    
    usage="""Print results"""
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("expedir", type=str, help="experiment directory")
    parser.add_argument("output", type=str, help="output file")

    args = parser.parse_args()
    main(args)
