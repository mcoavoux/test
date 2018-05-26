
import os
import glob


def get_accuracy(model):
    
    dev = "{}/eval_dev".format(model)
    test = "{}/eval_test".format(model)
    
    
    if os.path.isfile(dev) and os.path.isfile(test):
        
        dev_acc = float(open(dev).readline().split()[0])
        test_acc = float(open(test).readline().split()[0])
        
        epoch = open("{}/best_epoch".format(model)).readline().strip()
        
        return dev_acc, test_acc, epoch
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
                dev, test, epoch = acc
                
                if dev > best:
                    result[lang] = [dev, test, model, epoch]

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

def get_sag_mart_results():
    instream = open("results_sag_mart_2017")
    res = {}
    for line in instream:
        lang, score = line.strip().split()
        score = float(score)
        res[lang] = score
    return res

def main(args):
    
    results = get_results(args.expedir)
    
    bilty_results = get_bilty_results()
    sag_mart_results = get_sag_mart_results()
    
    header = ["lang", "dev", "test", "bilty", "bilty+polyglot", "delta-best", "delta-supervised", "sag&mart", "model", "epoch"]
    
    print("\t".join(header))
    
    avg = ["Avg", 0, 0, 0, 0, 0, 0, 0]
    for lang in sorted(results):
        
        lres = results[lang]
        bl = bilty_results[lang]
        
        if lang in sag_mart_results:
            sg = sag_mart_results[lang]
        else:
            sg = "-"
        
        diff = lres[1] - max([i for i in bl if type(i) == float])
        diff = round(diff, 2)
        
        diff_supervised = round(lres[1] - bl[0], 2)
        
        l = [lang] +  lres[:2] + bl + [diff, diff_supervised] + [sg] + lres[-2:]
        
        avg[1]+= lres[0]
        avg[2]+= lres[1]
        avg[3]+= bl[0]
        #avg[4]+= bl[1]
        avg[5]+= diff
        avg[6]+= diff_supervised
        
        if type(sg) == float:
            avg[7]+= sg
        
        print("\t".join(map(str, l)))
    
    for i in range(1, len(avg)):
        avg[i] = round(avg[i] / len(results), 2)
    print("\t".join(map(str, avg)))
    

if __name__ == "__main__":
    
    import argparse
    
    usage="""Print results"""
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("expedir", type=str, help="experiment directory")
    parser.add_argument("output", type=str, help="output file")

    args = parser.parse_args()
    main(args)
