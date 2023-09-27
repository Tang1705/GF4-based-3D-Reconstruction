# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw(img, name):
    plt.title(name)
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    col = ['b', 'g', 'r']
    j = 0
    for i in [hist_b, hist_g, hist_r]:
        plt.plot(i, color=col[j])
        j += 1
    plt.xlim([0, 256])
    # plt.show()


def detection(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    d_a, d_b, M_a, M_b = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            d_a = d_a + a[i][j]
            d_b = d_b + b[i][j]
    d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
    D = np.sqrt((np.square(d_a) + np.square(d_b)))

    for i in range(m):
        for j in range(n):
            M_a = np.abs(a[i][j] - d_a - 128) + M_a
            M_b = np.abs(b[i][j] - d_b - 128) + M_b

    M_a, M_b = M_a / (m * n), M_b / (m * n)
    M = np.sqrt((np.square(M_a) + np.square(M_b)))
    k = D / M
    print('偏色值:%f' % k)
    return


def auto_whiteBalance():
    img = cv2.imread('square.png')
    b, g, r = cv2.split(img)
    global m, n
    m, n = b.shape
    detection(img)
    plt.subplot(121)
    draw(img, 'origin')

    I_r, I_g, = 0, 0
    I_r_2 = np.zeros(r.shape)
    I_b_2 = np.zeros(b.shape)
    sum_I_r_2, sum_I_r, sum_I_b_2, sum_I_b, sum_I_g = 0, 0, 0, 0, 0
    max_I_r_2, max_I_r, max_I_b_2, max_I_b, max_I_g = int(r[0][0] ** 2), int(r[0][0]), int(b[0][0] ** 2), int(
        b[0][0]), int(
        g[0][0])
    for i in range(m):
        for j in range(n):
            I_r_2[i][j] = int(r[i][j] ** 2)
            I_b_2[i][j] = int(b[i][j] ** 2)
            sum_I_r_2 = I_r_2[i][j] + sum_I_r_2
            sum_I_b_2 = I_b_2[i][j] + sum_I_b_2
            sum_I_g = g[i][j] + sum_I_g
            sum_I_r = r[i][j] + sum_I_r
            sum_I_b = b[i][j] + sum_I_b
            if max_I_r < r[i][j]:
                max_I_r = r[i][j]
            if max_I_r_2 < I_r_2[i][j]:
                max_I_r_2 = I_r_2[i][j]
            if max_I_g < g[i][j]:
                max_I_g = g[i][j]
            if max_I_b_2 < I_b_2[i][j]:
                max_I_b_2 = I_b_2[i][j]
            if max_I_b < b[i][j]:
                max_I_b = b[i][j]

    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
    print(u_b, v_b, u_r, v_r)
    b0, g0, r0 = np.zeros(b.shape, np.uint8), np.zeros(g.shape, np.uint8), np.zeros(r.shape, np.uint8)
    for i in range(m):
        for j in range(n):
            b0[i][j] = u_b * (b[i][j] ** 2) + v_b * b[i][j]
            g0[i][j] = g[i][j]
            # r0[i][j] = r[i][j]
            r0[i][j] = u_r * (r[i][j] ** 2) + v_r * r[i][j]
    img_0 = cv2.merge([b0, g0, r0])
    cv2.imwrite("white-balance-image.png", img_0)

    detection(img_0)
    plt.subplot(122)
    draw(img_0, 'fix')
    plt.show()
    return img_0