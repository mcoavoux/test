
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

def main(args):
    
    results = get_results(args.expedir)
    
    for lang in sorted(results):
        l = [lang] + results[lang]
        print("\t".join(map(str, l)))
    

if __name__ == "__main__":
    
    import argparse
    
    usage="""Print results"""
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("expedir", type=str, help="experiment directory")
    parser.add_argument("output", type=str, help="output file")

    args = parser.parse_args()
    main(args)
