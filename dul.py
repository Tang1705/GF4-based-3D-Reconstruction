import xlrd
from collections import Counter


def fileload(filename='test.xlsx'):
    dataset = []
    workbook = xlrd.open_workbook(filename)
    table = workbook.sheets()[0]
    for row in range(table.nrows):
        dataset.append(table.row_values(row))
    return dataset


def plus(x, y):
    if x == y:
        return 0
    elif x == 0.0:
        return y
    elif y == 0.0:
        return x
    elif (x == 1.0 and y == 2.0) or (x == 2.0 and y == 1.0):
        return 3
    elif (x == 1.0 and y == 3.0) or (x == 3.0 and y == 1.0):
        return 2
    elif (x == 2.0 and y == 3.0) or (x == 3.0 and y == 2.0):
        return 1


def multi(x, y):
    if x == 0 or y == 0:
        return 0
    elif x == 2 and y == 2:
        return 3
    elif x == 3 and y == 3:
        return 2
    elif x == 1:
        return y
    elif y == 1:
        return x
    else:
        return 1


def GF(a, b, c, d, e, f):
    data = []

    for i in range(0, 4095):
        data.append(0)

    data[0] = a
    data[1] = b
    data[2] = c
    data[3] = d
    data[4] = e
    data[5] = f

    for i in range(6, 4095):
        data[i] = plus(data[i - 1], multi(data[i - 2], 3))
        data[i] = plus(data[i], multi(data[i - 3], 2))
        data[i] = plus(data[i], multi(data[i - 4], 1))
        data[i] = plus(data[i], multi(data[i - 5], 1))
        data[i] = plus(data[i], multi(data[i - 6], 3))

    res = []
    for i in range(0, 65):
        res.append([])
        for j in range(0, 63):
            res[i].append(0)
    for i in range(0, 4095):
        res[i % 65][i % 63] = data[i]

    return res


if __name__ == '__main__':
    # data = fileload()
    data = GF(1, 1, 1, 1, 1, 1)

    output = open('data.xls', 'w', encoding='gbk')

    for i in range(len(data)):
        for j in range(len(data[i])):
            output.write(str(data[i][j]))  # write函数不能写int类型的参数，所以使用str()转化

            output.write('\t')

        output.write('\n')

    output.close()

    mark = False
    for i in range(0, 63):
        for j in range(0, 61):
            for k in range(0, 63):
                for l in range(0, 61):
                    if abs(data[i][j] - data[k][l]) < 10 ** (-5) and abs(data[i][j + 1] - data[k][l + 1]) < 10 ** (
                            -5) and abs(data[i][
                                            j + 2] - data[k][
                                            l + 2]) < 10 ** (-5) and abs(data[i + 1][j] - data[k + 1][l]) < 10 ** (
                            -5) and abs(data[i + 1][
                                            j + 1] - data[k + 1][l + 1]) < 10 ** (-5) and \
                            abs(data[i + 1][j + 2] - data[k + 1][l + 2]) < 10 ** (-5) and i != k and j != l:
                        mark = True
                        print('first:')
                        print(data[i][j])
                        print(data[i][j + 1])
                        print(data[i][j + 2])

                        print(data[i + 1][j])
                        print(data[i + 1][j + 1])
                        print(data[i + 1][j + 2])

                        print('second:')
                        print(data[k][l])
                        print(data[k][l + 1])
                        print(data[k][l + 2])

                        print(data[k + 1][l])
                        print(data[k + 1][l + 1])
                        print(data[k + 1][l + 2])
                        break
                if mark:
                    break
            if mark:
                break
        if mark:
            break
    if not mark:
        print("success")
