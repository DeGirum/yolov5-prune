
from cProfile import label
from cmath import nan
import numpy as np
from pathlib import Path
import yaml

def read_data_yaml(data : str):
    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    return data['names']

def zero_out_confusion(confusion, index):
    confusion[:, index] = 0
    for i in range(confusion.shape[0]):
        confusion[i, i] += confusion[index, i]
        confusion[index, i] = 0
    
def remove_unbalanced_high(confusion, thr : float):
    total_pos = []
    for i in range(confusion.shape[0] - 1):
        total_pos.append( np.sum( confusion[:,i] ) )
    
    total_cnt = np.sum(total_pos)
    return np.squeeze( np.where(total_pos >  total_cnt * thr) )


def remove_items(remove_list, confusion, cnt, f1_score : bool):
    total_pos = []
    pred_pos = []
    true_pos = []
    for i in range(confusion.shape[0]):
        total_pos.append( np.sum( confusion[:,i] ) )
        pred_pos.append( np.sum( confusion[i,:] ) )
        true_pos.append( confusion[i,i] )

    precision = np.divide(true_pos, pred_pos)
    recall = np.divide(true_pos, total_pos)

    score = precision
    if f1_score:
        score = 2 * np.divide( np.multiply(precision, recall), (precision + recall + 0.000001) )
    
    sorted = np.argsort(score)
    index2remove = sorted[0]
    remove_list.append( index2remove )
    cnt = cnt - 1
    if (cnt > 0):
        zero_out_confusion(confusion, index2remove)
        remove_items(remove_list, confusion, cnt, f1_score)


with open('./runs/val_load/exp110/confusion.txt', 'r') as f:
    confusion = [[int(float(num)) for num in line.split(' ')] for line in f]


confusion = np.array(confusion)

remove_list_high = remove_unbalanced_high(confusion, 0.1)

for i in remove_list_high:
    zero_out_confusion(confusion, i)

conf_copy_f1 = np.copy(confusion)
remove_list_f1 = []
remove_items(remove_list_f1, conf_copy_f1, 15, f1_score=True)
conf_copy_pr = np.copy(confusion)
remove_list_pr = []
remove_items(remove_list_pr, conf_copy_pr, 15, f1_score=False)

remove_list = remove_list_f1 + remove_list_pr + remove_list_high.tolist()
remove_list = np.unique(remove_list)
print(remove_list)


names = read_data_yaml('data/cachengo.yaml')
names.append('background')
for i in remove_list:
    print( names[i] )
