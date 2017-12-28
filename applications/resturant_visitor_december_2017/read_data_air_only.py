import matplotlib.pylab as plt
import numpy as np


def read_data_air_only(file_path):

    class resturant:
        date = []
        visitors = []

    data = {}
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file.readlines()):
            if line_num == 0:
                header = line.replace('\n', '').replace(' ', ',').split(',')
            else:
                line = line.replace('\n', '').replace(' ', ',').split(',')
                if line[0] in data.keys():
                    data[line[0]].date.append(line[1])
                    data[line[0]].visitors.append(int(line[2]))
                else:
                    rest = resturant()
                    rest.date = [line[1]]
                    rest.visitors = [int(line[2])]
                    data.update({line[0]: rest})

    # plt.plot(data[list(data.keys())[0]].visitors)
    # plt.show()

    return data

if __name__ == '__main__':
    data = read_data_air_only('/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/air_visit_data.csv')