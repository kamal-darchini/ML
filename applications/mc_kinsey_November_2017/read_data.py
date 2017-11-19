import matplotlib.pylab as plt
import numpy as np


def read_data(file_path):

    with open(file_path, 'r') as file:
        data_1 = []
        data_2 = []
        data_3 = []
        data_4 = []
        for line_num, line in enumerate(file.readlines()):
            if line_num == 0:
                header = ['date', 'time', 'junction', 'vehicles', 'id']
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

    # return data
    cumsum, moving_aves_1 = [0], []
    N = 30
    for i, x in enumerate([x['vehicles'] for x in data_1], 1):
        cumsum.append(cumsum[i - 1] + int(x))
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            moving_aves_1.append(moving_ave)
    moving_aves_1 = [0] * 29 + moving_aves_1

    cumsum, moving_aves_2 = [0], []
    for i, x in enumerate([x['vehicles'] for x in data_2], 1):
        cumsum.append(cumsum[i - 1] + int(x))
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            moving_aves_2.append(moving_ave)
    moving_aves_2 = [0] * 29 + moving_aves_2

    cumsum, moving_aves_3 = [0], []
    for i, x in enumerate([x['vehicles'] for x in data_3], 1):
        cumsum.append(cumsum[i - 1] + int(x))
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            moving_aves_3.append(moving_ave)
    moving_aves_3 = [0] * 29 + moving_aves_2

    cumsum, moving_aves_4 = [0], []
    for i, x in enumerate([x['vehicles'] for x in data_4], 1):
        cumsum.append(cumsum[i - 1] + int(x))
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            moving_aves_4.append(moving_ave)
    moving_aves_4 = [0] * 29 + moving_aves_2

    for i, x in enumerate(data_1):
        data_1[i]['vehicles'] = int(data_1[i]['vehicles']) - moving_aves_1[i]
    for i, x in enumerate(data_2):
        data_2[i]['vehicles'] = int(data_2[i]['vehicles']) - moving_aves_2[i]
    for i, x in enumerate(data_3):
        data_3[i]['vehicles'] = int(data_3[i]['vehicles']) - moving_aves_3[i]
    for i, x in enumerate(data_4):
        data_4[i]['vehicles'] = int(data_4[i]['vehicles']) - moving_aves_4[i]

    vehicles_1 = [x['vehicles'] for x in data_1]
    vehicles_2 = [x['vehicles'] for x in data_2]
    vehicles_3 = [x['vehicles'] for x in data_3]
    vehicles_4 = [x['vehicles'] for x in data_4]
    std_1 = np.sqrt(np.var(vehicles_1))
    std_3 = np.sqrt(np.var(vehicles_3))
    std_4 = np.sqrt(np.var(vehicles_4))

    std_2 = np.sqrt(np.var(vehicles_2))
    for i, x in enumerate(data_1):
        data_1[i]['vehicles'] = data_1[i]['vehicles'] / std_1
    for i, x in enumerate(data_2):
        data_2[i]['vehicles'] = data_2[i]['vehicles'] / std_2
    for i, x in enumerate(data_3):
        data_3[i]['vehicles'] = data_3[i]['vehicles'] / std_3
    for i, x in enumerate(data_4):
        data_4[i]['vehicles'] = data_4[i]['vehicles'] / std_4

    # y = []
    # for x in data_1:
    #     y.append(x['vehicles'])
    # plt.plot(y)
    # # plt.plot(moving_aves_1, 'r')
    # plt.show()

    print('done reading file!')

    moving_avg = []
    moving_avg.append(moving_aves_1)
    moving_avg.append(moving_aves_2)
    moving_avg.append(moving_aves_3)
    moving_avg.append(moving_aves_4)

    std = []
    std.append(std_1)
    std.append(std_2)
    std.append(std_3)
    std.append(std_4)

    return data_1, data_2, data_3, data_4, moving_avg, std, moving_aves_1

if __name__ == '__main__':
    data = read_data('/Users/kamaloddindarchinimaragheh/Desktop/mk_kinsey/mc_kinsey_data.txt')