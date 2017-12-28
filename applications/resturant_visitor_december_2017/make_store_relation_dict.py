def make_dict(file_path):

    data = {}
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file.readlines()):
            if line_num == 0:
                header = line.replace('\n', '').replace(' ', ',').split(',')
            else:
                line = line.replace('\n', '').replace(' ', ',').split(',')
                data.update({line[1]: line[0]})

    return data