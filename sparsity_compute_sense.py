

import csv
import argparse
import os
def main(opt):
    
    csv_path = opt.sense_path
    layer_sense = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            elif line_count == 1:
                org_perf = row
            else:
                layer = layer_sense.get( row[1] )
                new_info = [float(row[3]), float(row[6])]
                if layer:
                    layer.append( new_info )
                else:
                    layer_sense[ row[1] ] = [ new_info ]

            line_count += 1
        print(f'Processed {line_count} lines.')

    ThrPerc = opt.threshold
    AccThr = ThrPerc * float(org_perf[6])
    pruning_dict = {}
    for layerName in layer_sense.keys():
        pruning_dict[layerName] = 0.0
        sense_info = layer_sense[layerName]
        for sense in sense_info:
            if sense[1] > AccThr:
                if sense[0] > pruning_dict[layerName] and sense[0] <= opt.max_pruning:
                    pruning_dict[layerName] = sense[0]


    from DG_Prune.utils import dump_json

    filename, file_extension = os.path.splitext(csv_path)
    dump_json( pruning_dict, '{}_Thr{}.json'.format(filename, int(ThrPerc*100) ) )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sense-path', type=str, help='sensitivity analysis csv file path')
    parser.add_argument('--threshold', type=float, default=0.95, help='Pruning Threshold Percentage')
    parser.add_argument('--max-pruning', type=float, default=0.85, help='Max Pruning Percentage')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)