import sys


def count_bias_unbias(file_path):

    total, true_size = 0, 0
    with open(file_path, 'r') as f:

        for line in f:
            fields = line.split("\t")
            if fields[1] == "true":
                true_size += 1
            total += 1

    print(f"total: {total}, true_size: {true_size}, false_size: {total - true_size}.")


print(sys.argv[1])
count_bias_unbias(sys.argv[1])