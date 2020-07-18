import cv2
import math
import numpy as np
from GFmatrix import GF
import auto_whiteBalance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
on_EVENT_LBUTTONDOWN
获取鼠标点击位置的坐标
横向坐标存入m（列）
纵向坐标存入n（行）
"""


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        m.append(x)
        n.append(y)


"""
get_candidate_points:
将RGB图像转为单通道
计算每一个像素点横向和纵向的单通道色彩强度差，选择值最大的通道为该像素点的差值，将差值大于选定阈值的点视为候选点
在选定的窗口大小中进行计算，减少处理数据的数量
返回存有候选点数据的矩阵
-1为候选点
-3为非候选点
"""


def get_candidate_points():
    candidate = np.zeros((1024, 1280))  # 初始化图像点位置矩阵为0

    i = n[0]
    while i < n[1]:
        j = m[0]
        while j < m[1]:
            sum = [0, 0, 0]  # 存储r,g,b三个通道的差值结果
            tempxy = [0, 0, 0, 0, 0, 0]  # 存储三个通道纵向和横向的临时求和结果
            k = -6
            while k < 7:  # 十字掩码长度选择为菱形对角线长度的一半
                tempxy[0] = tempxy[0] + b[i + k][j]  # b通道水平方向
                tempxy[1] = tempxy[1] + b[i][j + k]  # b通道铅直方向
                tempxy[2] = tempxy[2] + g[i + k][j]  # g通道水平方向
                tempxy[3] = tempxy[3] + g[i][j + k]  # g通道铅直方向
                tempxy[4] = tempxy[4] + r[i + k][j]  # r通道水平方向
                tempxy[5] = tempxy[5] + r[i][j + k]  # r通道铅直方向
                k = k + 1
            sum[0] = sum[0] + abs(tempxy[0] - tempxy[1])  # r通道差值
            sum[1] = sum[1] + abs(tempxy[2] - tempxy[3])  # g通道差值
            sum[2] = sum[2] + abs(tempxy[4] - tempxy[5])  # b通道差值
            d = max(sum[0], sum[1], sum[2])  # 选择差值最大的通道
            # print(d)
            if d > 350:  # tq和zyy人工学习调参选阈值，阈值增大，候选点集中于球体中央
                candidate[i][j] = -1  # -1标记为候选点
            else:
                candidate[i][j] = -3  # -3标记为非候选点
            j = j + 1
        i = i + 1

    # 统计候选点个数
    print("Candidate points:")
    num = 0
    i = n[0]
    while i < n[1]:
        j = m[0]
        while j < m[1]:
            if candidate[i][j] == -1:
                num += 1
            j = j + 1
        i = i + 1
    print(num)

    return candidate


"""
get_grid_points:
将RGB图像根据公式Gray = 0.2989 * R + 0.5907 * G + 0.1140 * B 转为灰度图像
根据真正的角点具有严格的中心对称性，将相关系数大于选定阈值的点选做特征点
并根据P1和P2类型点的特征（左右或上下为模式元素）将特征点分类为两类特征点
白色背景的灰度值高于颜色元素的灰度值
圆邻域采用SUSAN角点检测法的圆邻域，直径为7
1为角点
2为非角点
"""


def get_grid_points(candidate):
    GrayImage = np.zeros((1024, 1280))

    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            # 转灰度图像，提高绿色通道的比例，使得白色背景与绿色元素易于区分
            GrayImage[i][j] = 0.2989 * r[i][j] + 0.5907 * g[i][j] + 0.1140 * b[i][j]
            j = j + 1
        i = i + 1

    gridpoints = np.zeros((1024, 1280))
    circle_neighborhood = [[0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0]]
    i = n[0]
    while i < n[1]:
        j = m[0]
        while j < m[1]:
            """
            element元素说明
            便于计算圆形邻域的相关系数，引入变量element
            第一个元素：M_{Ci} * M_{Ci}'之和
            第二个元素：M_{Ci}之和
            第三个元素：M_{Ci}'之和
            第四个元素：M_{Ci}^2之和
            第五个元素：(M_{Ci}')^2之和
            """
            element = [0, 0, 0, 0, 0]
            if candidate[i][j] == -1:
                # 计算直径为7的圆邻域的色彩强度，与数据结构中数组和矩阵的关系相似
                p = -3
                while p < 4:
                    q = -3
                    while q < 4:
                        if circle_neighborhood[p][q] == 1:
                            # 以圆心像素点为（0,0）
                            # 计算其余点的坐标和旋转180度后的坐标
                            # 原坐标
                            imgx = i + p
                            imgy = j + q
                            # 旋转180度后的坐标
                            imgxp = i - p
                            imgyp = j - q
                            element[0] = element[0] + int(GrayImage[imgx][imgy]) * int(GrayImage[imgxp][imgyp])
                            element[1] = element[1] + int(GrayImage[imgx][imgy])
                            element[2] = element[2] + int(GrayImage[imgxp][imgyp])
                            element[3] = element[3] + int(GrayImage[imgx][imgy] ** 2)
                            element[4] = element[4] + int(GrayImage[imgxp][imgyp] ** 2)
                        q = q + 1
                    p = p + 1

                pc = (37 * element[0] - element[1] * element[2]) / (
                        np.sqrt(37 * element[3] - element[1] ** 2) * np.sqrt(37 * element[4] - element[2] ** 2))
                if pc > 0.1:  # 相关系数足够大的点被判断为特征点(对称系数为0的为特征点——A Twofold...)
                    gridpoints[i][j] = 1

            else:
                gridpoints[i][j] = 0
            j = j + 1
        i = i + 1

    # 统计角点数量
    print("Grid points(greater than t):")
    num = 0
    i = n[0]
    while i < n[1]:
        j = m[0]
        while j < m[1]:
            if gridpoints[i][j] == 1:
                num += 1
            j = j + 1
        i = i + 1
    print(num)

    return gridpoints


"""
bfs8:
8邻域广度优先搜索
确定唯一特征点位置
"""


def bfs8(g, x, y):
    counter = 1
    queue = [[x, y]]
    ans = [0, 0]
    flag[x][y] = 1
    ans[0] = 1.0 * x
    ans[1] = 1.0 * y
    while len(queue) > 0:
        current = queue.pop(0)
        flag[current[0]][current[1]] = 1
        i = 0
        while i < 8:
            temp = [0, 0]
            temp[0] = current[0] + nx[i]
            temp[1] = current[1] + ny[i]
            if temp[0] < n[0] or temp[0] > n[1] or temp[1] < m[0] or temp[1] > m[1]:
                i = i + 1
                continue
            if flag[int(temp[0])][int(temp[1])] or g[int(temp[0])][int(temp[1])] == 0:
                i = i + 1
                continue
            flag[int(temp[0])][int(temp[1])] = 1
            queue.append(temp)
            ans[0] = ans[0] + 1.0 * temp[0]
            ans[1] = ans[1] + 1.0 * temp[1]
            counter = counter + 1
            i = i + 1
    ans[0] = ans[0] / counter
    ans[1] = ans[1] / counter
    return ans


"""
get_feature_point:
调用8邻域深度优先搜索
确定单一特征点位置
"""


def get_feature_point(gps):
    q = []
    i = n[0]
    while i < n[1]:
        j = m[0]
        while j < m[1]:
            if flag[i][j] == 1 or gps[i][j] == 0:
                j = j + 1
                continue
            temp = bfs8(gps, i, j)
            q.append(temp)
            j = j + 1
        i = i + 1
    print("Actual grid point:")
    print(len(q))
    return q


"""
transfer_feature_points:
将存有特征点浮点位置的列表转换为矩阵
"""


def transfer_feature_points(featurepoints_position):
    featurepoints = np.zeros((1024, 1280))
    for item in featurepoints_position:
        row = math.floor(item[0])
        column = math.floor(item[1])

        sum = [0, 0, 0]  # 存储r,g,b三个通道的差值结果
        tempxy = [0, 0, 0, 0, 0, 0]  # 存储三个通道纵向和横向的临时求和结果
        k = -6
        while k < 7:  # 十字掩码长度选择为菱形对角线长度的一半
            tempxy[0] = tempxy[0] + b[row + k][column]  # b通道水平方向
            tempxy[1] = tempxy[1] + b[row][column + k]  # b通道铅直方向
            tempxy[2] = tempxy[2] + g[row + k][column]  # g通道水平方向
            tempxy[3] = tempxy[3] + g[row][column + k]  # g通道铅直方向
            tempxy[4] = tempxy[4] + r[row + k][column]  # r通道水平方向
            tempxy[5] = tempxy[5] + r[row][column + k]  # r通道铅直方向
            k = k + 1
        sum[0] = sum[0] + abs(tempxy[0] - tempxy[1])  # b通道差值
        sum[1] = sum[1] + abs(tempxy[2] - tempxy[3])  # g通道差值
        sum[2] = sum[2] + abs(tempxy[4] - tempxy[5])  # r通道差值
        d = max(sum[0], sum[1], sum[2])  # 选择差值最大的通道
        if d == sum[0]:
            if tempxy[0] - tempxy[1] > 0:
                featurepoints[row][column] = -2
            else:
                featurepoints[row][column] = -1
        elif d == sum[1]:
            if tempxy[2] - tempxy[3] > 0:
                featurepoints[row][column] = -2
            else:
                featurepoints[row][column] = -1
        else:
            if tempxy[4] - tempxy[5] > 0:
                featurepoints[row][column] = -2
            else:
                featurepoints[row][column] = -1

    # 统计特征点个数
    print("Feature points:")
    num = 0
    i = n[0]
    while i < n[1]:
        j = m[0]
        while j < m[1]:
            if featurepoints[i][j] == -1 or featurepoints[i][j] == -2:
                num += 1
            j = j + 1
        i = i + 1
    print(num)

    return featurepoints


"""
calculate_color:
计算选定窗口内除特征点外的所有像素的颜色
传入标记好P1和P2的特征点
red-0
green-1
blue-2
black-3
"""


def calculate_color(gridpoints):
    i = n[0]
    while i < n[1]:
        j = m[0]
        while j < m[1]:
            if gridpoints[i][j] != -1 and gridpoints[i][j] != -2:

                rh = r[i][j]
                gh = g[i][j]
                bh = b[i][j]

                # 由于hsv颜色空间的计算方式，排除r=g=b的像素点
                if rh == gh and gh == bh:
                    gridpoints[i][j] = 0
                    j = j + 1
                    continue

                h = hsv(rh, gh, bh)
                result = max(h[0], h[1], h[2])
                k = h[3] - math.sqrt((1 - result ** 2))

                if result == h[0]:
                    if k < 0.2:  # 通常黑色的k值小于0.2
                        gridpoints[i][j] = 0
                    else:
                        gridpoints[i][j] = 1
                elif result == h[1]:
                    if k < - 0.2:  # 通常黑色的k值小于0.2
                        gridpoints[i][j] = 0
                    else:
                        gridpoints[i][j] = 2
                elif result == h[2]:
                    if k < 0.2:  # 通常黑色的k值小于0.2
                        gridpoints[i][j] = 0
                    else:
                        gridpoints[i][j] = 3

            j = j + 1
        i = i + 1

    return gridpoints


"""
计算hsv色彩空间
选出每个像素hr,hg,hb中的最大值，并通过色彩饱和度判断该点为r/g/b或黑色，黑色的k值通常很小
若结果小于选定的阈值，则该点颜色为黑色，反之为r/g/b的最大值
"""


def hsv(r, g, b):
    rh = int(r)
    gh = int(g)
    bh = int(b)
    h = [0, 0, 0, 0]
    h[0] = (2 * rh - gh - bh) / (2 * math.sqrt((rh - gh) ** 2 + (rh - bh) * (gh - bh)))
    h[1] = (2 * gh - rh - bh) / (2 * math.sqrt((gh - rh) ** 2 + (gh - bh) * (rh - bh)))
    h[2] = (2 * bh - gh - rh) / (2 * math.sqrt((bh - gh) ** 2 + (bh - rh) * (gh - rh)))
    h[3] = math.sqrt(1 - ((rh * gh + gh * bh + rh * bh) / (rh ** 2 + gh ** 2 + bh ** 2)))  # 色彩强度
    return h


"""
reconstruction:
根据窗口大小进行重建
参数：gridpoints-特征点
    num-特征点数量
    map-位置
返回值：coord-特征点三维坐标
        num_coord-特征点三维坐标数量
"""


def reconstructon(fps, num, hc, hp, map1, map2):
    num_of_coord = 0
    coordinates = np.zeros((num, 3))
    row = n[0]
    while row < n[1]:
        column = m[0]
        row_position = np.zeros((1, 100))
        counter = 0
        while column < m[1]:
            if fps[row][column] == -1 or fps[row][column] == -2:
                row_position[0][counter] = column
                counter = counter + 1
            column = column + 1

        c = 0
        while c < counter:
            matrix = np.zeros((3, 3))
            matrix[0, :] = hc[2, 0:3] * (row_position[0][c] + 1) - hc[0, 0:3]
            matrix[1, :] = hc[2, 0:3] * (row + 1) - hc[1, 0:3]

            temp_v = int(row_position[0][c])
            if fps[row][temp_v] == -1:
                matrix[2, :] = np.dot(hp[2, 0:3],
                                      map1[int(fps[row - 7][temp_v - 14])][int(fps[row - 7][temp_v])][
                                          int(fps[row - 7][temp_v + 14])][int(fps[row + 7][temp_v - 14])][
                                          int(fps[row + 7][temp_v])][int(fps[row + 7][temp_v + 14])]) - hp[0, 0:3]
                # ax = b的解为x
                tang = np.linalg.solve(matrix,
                                       np.mat(
                                           [hc[0, 3] - hc[2, 3] * (row_position[0][c] + 1),
                                            hc[1, 3] - hc[2, 3] * (row + 1),
                                            hp[0, 3] - hp[2, 3] *
                                            map1[int(fps[row - 7][temp_v - 14])][int(fps[row - 7][temp_v])][
                                                int(fps[row - 7][temp_v + 14])][int(fps[row + 7][temp_v - 14])][
                                                int(fps[row + 7][temp_v])][
                                                int(fps[row + 7][temp_v + 14])]]).transpose())

                if tang[2, 0] > 750 and tang[2, 0] < 1500:
                    coordinates[num_of_coord, :] = tang.transpose()
                    num_of_coord = num_of_coord + 1

            elif fps[row][temp_v] == -2:
                print([int(fps[row][temp_v - 7])], [int(fps[row][temp_v + 7])], [
                    int(fps[row][temp_v + 20])], [int(fps[row + 14][temp_v - 7])], [
                          int(fps[row + 14][temp_v + 7])], [int(fps[row + 14][temp_v + 20])])
                matrix[2, :] = np.dot(hp[2, 0:3],
                                      map2[int(fps[row][temp_v - 7])][int(fps[row][temp_v + 7])][
                                          int(fps[row][temp_v + 20])][int(fps[row + 14][temp_v - 7])][
                                          int(fps[row + 14][temp_v + 7])][int(fps[row + 14][temp_v + 20])]) - hp[
                                                                                                              0,
                                                                                                              0:3]
                # ax = b的解为x
                tang = np.linalg.solve(matrix,
                                       np.mat(
                                           [hc[0, 3] - hc[2, 3] * (row_position[0][c] + 1),
                                            hc[1, 3] - hc[2, 3] * (row + 1),
                                            hp[0, 3] - hp[2, 3] *
                                            map2[int(fps[row][temp_v - 7])][int(fps[row][temp_v + 7])][
                                                int(fps[row][temp_v + 20])][int(fps[row + 14][temp_v - 7])][
                                                int(fps[row + 14][temp_v + 7])][
                                                int(fps[row + 14][temp_v + 20])]]).transpose())

                if tang[2, 0] > 750 and tang[2, 0] < 1500:
                    coordinates[num_of_coord, :] = tang.transpose()
                    num_of_coord = num_of_coord + 1
            c = c + 1
        row = row + 1

    return coordinates, num_of_coord


"""
R12: 由相机到投影仪的旋转矩阵 3*3
T12: 由相机到投影仪的平移矩阵 1*3
Kc1: 相机内参矩阵 3*3
Kp2: 投影仪内参矩阵 3*3
Hc1: 相机单应性矩阵 3*4
Hp2: 投影仪单应性矩阵 3*4
"""

def reconstruction():
#if __name__ == "__main__":
    # 参数
    R12 = np.mat([[9.8163294080189145e-001, 4.8335868327196085e-003, -1.9071813225532785e-001],
                  [-3.1727563235811407e-003, 9.9995435144492506e-001, 9.0126934747814975e-003],
                  [1.9075298988467088e-001, -8.2420546400611000e-003, 9.8160346646971908e-001]])

    T12 = np.mat([[1.9800307959186890e+002, 1.6932750885389197e+001, -6.3891647030676380e-001]])
    Kc1 = np.mat([[2.1442937669341595e+003, 0., 6.5308645130447132e+002],
                  [0., 2.1438987235415575e+003, 4.9954659425293477e+002],
                  [0., 0., 1.]])
    Kp2 = np.mat([[1.7540362475016316e+003, 0., 4.2865171873700905e+002],
                  [0., 3.4946273684116309e+003, 5.1525832055625494e+002],
                  [0., 0., 1.]])

    Hc1 = np.dot(Kc1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    Hp2 = np.dot(Kp2, np.hstack((R12, T12.transpose())))

    global img
    # img = auto_whiteBalance.auto_whiteBalance()
    img = cv2.imread("square.png")
    # m是水平方向（列），n是铅直方向（行）
    global m, n
    m = []
    n = []

    # 确定重建物体像素坐标范围
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    if len(m) == 2 and len(n) == 2:
        cv2.destroyAllWindows()
    print(m, n)

    # 单通道分别计算，效果优于转灰度图像计算
    global b, g, r
    b, g, r = cv2.split(img)

    # print(b.shape)
    global flag, nx, ny
    flag = np.zeros((1024, 1280))
    nx = [-1, -1, -1, 0, 0, 1, 1, 1]
    ny = [1, 0, -1, 1, -1, 1, 0, -1]

    # 1024*1280 -1候选点，-3非候选点
    candidate = get_candidate_points()
    # 1024*1280 1角点，0非角点
    gridpoints = get_grid_points(candidate)
    # 存有特征点的列表 -1特征点
    featurepoints_position = get_feature_point(gridpoints)
    # 1024*1280 -1特征点P1 -2特征点P2 0非特征点
    featurepoints = transfer_feature_points(featurepoints_position)
    # 0红 1绿 2黑 3蓝
    featurepoints = calculate_color(featurepoints)

    num = 0
    i = n[0]
    while i < n[1]:
        j = m[0]
        while j < m[1]:
            if featurepoints[i][j] == -1 or featurepoints[i][j] == -2:
                num += 1
            j = j + 1
        i = i + 1
    print(num)

    # 计算map,第（i,j）特征点的位置
    data = GF(1, 1, 1, 1, 1, 1)
    map1 = np.zeros((4, 4, 4, 4, 4, 4))
    map2 = np.zeros((4, 4, 4, 4, 4, 4))

    i = 0
    while i < 42:
        if 27 * (i + 1) < 1140:
            j = 1
            while j < 62:
                if 27 * j < 1674:
                    map1[data[i][j - 1]][data[i][j]][data[i][j + 1]][data[i + 1][j - 1]][data[i + 1][j]][
                        data[i + 1][j + 1]] = (27 * j + 13.5) / 2
                    map2[data[i][j - 1]][data[i][j]][data[i][j + 1]][data[i + 1][j - 1]][data[i + 1][j]][
                        data[i + 1][j + 1]] = (27 * j) / 2
                else:
                    break
                j = j + 1
        else:
            break
        i = i + 1
    coord, num_coord = reconstructon(featurepoints, num, Hc1, Hp2, map1, map2)

    # 绘制特征点
    point_size = 1
    point_color = (255, 255, 255)
    thickness = 0  # 可以为 0 、4、8
    font = cv2.FONT_HERSHEY_SIMPLEX

    i = n[0]
    while i < n[1]:
        j = m[0]
        while j < m[1]:
            if featurepoints[i][j] == -1 or featurepoints[i][j] == -2:
                cv2.circle(img, (j, i), point_size, point_color, thickness)
            j = j + 1
        i = i + 1

    cv2.imwrite("Resource//main_scene//feature_point.png", img)

    # 将三维点坐标存入文本文件
    np.savetxt("data//result.txt", coord[:num_coord, :])
    print(num_coord)
    print(len(coord))
    # 绘制三维散点图
    x = coord[:num_coord, 0]
    y = coord[:num_coord, 1]
    z = coord[:num_coord, 2]
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(x, y, z, s=1, c='#5599FF')

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    # 显示前保存，防止保存空白图片
    plt.savefig("Resource//main_scene//reconstruction_result.png")
    plt.show()

    # cv2.namedWindow("image")
    # cv2.imshow('image', img)
    # cv2.waitKey(0)  # 按0退出
