import numpy as np
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('heap_name', type=str)
parser.add_argument('result_file', type=str)
args = parser.parse_args()

# with open('ycb_results_heapsearch.pkl', 'rb') as f:
with open(args.result_file, 'rb') as f:
    data = pkl.load(f)

N = 4

# print("Results for", args.heap_name)
# results = data[args.heap_name]
# print(results[:5])

print("TOTAL: ")
num_in_top = 0
for heap_name in data:
    top = data[heap_name]
    for r in top:
        if r[0] == heap_name:
            num_in_top += 1
            print(heap_name)
        break


print("NUMBER IN TOP {}: {}".format(N, num_in_top))
print("NUM HEAPS SO FAR: ", len(data))
