import matplotlib.pylab as plt
import numpy as np


def read_test_data(file_path):

    with open(file_path, 'r') as file:
        data_1 = []
        data_2 = []
        data_3 = []
        data_4 = []
        for line_num, line in enumerate(file.readlines()):
            if line_num == 0:
                header = ['date', 'time', 'junction', 'id']
            else:
                parsed_line = line.replace('\n', '').replace(' ', ',').split(',')
                if parsed_line[2] == '1':
                    data_1.append({k: v for k, v in zip(header, line.replace('\n', '').replace(' ', ',').split(','))})
                if parsed_line[2] == '2':
                    data_2.append({k: v for k, v in zip(header, line.replace('\n', '').replace(' ', ',').split(','))})
                if parsed_line[2] == '3':
                    data_3.append({k: v for k, v in zip(header, line.replace('\n', '').replace(' ', ',').split(','))})
                if parsed_line[2] == '4':
                    data_4.append({k: v for k, v in zip(header, line.replace('\n', '').replace(' ', ',').split(','))})

    return data_1, data_2, data_3, data_4

if __name__ == '__main__':
    data = read_test_data('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/mc_kinsey_data_test.txt')