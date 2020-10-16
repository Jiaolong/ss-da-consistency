import glob
import numpy as np
from utils.config import get_config
from utils.utils import get_logger

def main():
    config = get_config()
    print(config.log_dir)
    log_dirs = glob.glob(config.log_dir.replace(str(config.random_seed), '*'))
    best_accs = []
    best_nonsib_accs = []
    for log_dir in sorted(log_dirs):
        log_file = sorted(glob.glob(log_dir + '/run*'))[-1]
        print('Reading results from {}'.format(log_file))
        with open(log_file, 'r') as f:
            lines = f.read().splitlines()
            best_line = ''
            for line in lines:
                if 'Best testing accuracy' in line:
                    best_line = line
            if best_line:
                print(best_line)
                if 'sib' in best_line:
                    best_nonsib = best_line.split(' ')[-6]
                    best = best_line.split(' ')[-3]
                    best_nonsib_accs.append(float(best_nonsib))
                else:
                    best = best_line.split(' ')[-2]
                best_accs.append(float(best))
   
    if best_nonsib_accs:
        accs = np.array(best_nonsib_accs, np.float32)
        print('===> non-sib mean: {:.2f}, std: {:.2f}'.format(np.mean(accs), np.std(accs)))

    if best_accs:
        accs = np.array(best_accs, np.float32)
        print('===> {} mean: {:.2f}, std: {:.2f}'.format(config.method, np.mean(accs), np.std(accs)))
    
if __name__ == '__main__':
    main()
