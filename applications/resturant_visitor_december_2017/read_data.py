import matplotlib.pylab as plt
import numpy as np

from applications.resturant_visitor_december_2017.make_store_relation_dict import make_dict


def read_data(file_path, relation_path=None):

    if relation_path:
        relation_dict = make_dict(relation_path)
        data = {}
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file.readlines()):
                if line_num == 0:
                    header = line.replace('\n', '').replace(' ', ',').split(',')
                else:
                    line = line.replace('\n', '').replace(' ', ',').split(',')
                    try:
                        if relation_dict[line[0]] in data.keys():
                            try:
                                data[relation_dict[line[0]]][line[1]] += int(line[-1])
                            except KeyError:
                                data[relation_dict[line[0]]].update({line[1]: int(line[-1])})
                        else:
                            data.update({relation_dict[line[0]]: {line[1]: int(line[-1])}})
                    except KeyError:
                        pass
    else:
        data = {}
        num_resturants = 0
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file.readlines()):
                if line_num == 0:
                    header = line.replace('\n', '').replace(' ', ',').split(',')
                else:
                    line = line.replace('\n', '').replace(' ', ',').split(',')
                    if line[0] in data.keys():
                        try:
                            data[line[0]][line[1]] += int(line[-1])
                        except KeyError:
                            data[line[0]].update({line[1]: int(line[-1])})
                    else:
                        data.update({line[0]: {line[1]: int(line[-1])}})

    # plt.plot(data[list(data.keys())[0]].visitors)
    # plt.show()

    return data

if __name__ == '__main__':
    # data = read_data('/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/air_visit_data.csv')
    data = read_data('/Users/kamaloddindarchinimaragheh/git/ML/data/time_series/resturant_visitors/hpg_reserve.csv')