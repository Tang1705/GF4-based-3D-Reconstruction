# -*- coding: utf-8 -*-

from PIL import Image
import math


def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def de_bruijn(k, n):
    """
    de Bruijn sequence for alphabet k
    and subsequences of length n.
    """
    try:
        # let's see if k can be cast to an integer;
        # if so, make our alphabet a list
        _ = int(k)
        alphabet = list(map(str, range(k)))

    except (ValueError, TypeError):
        alphabet = k
        k = len(k)

    a = [0] * k * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return "".join(alphabet[i] for i in sequence)


if __name__ == "__main__":
    DB = de_bruijn(3, 4)
    DB_list = list(DB)
    print(len(DB_list))
    print(DB_list)

    x = 912
    y = 1140

    bgcolor = 0xffffff  # 投影图案是白色背景
    c = Image.new("RGB", (x, y), bgcolor)
    h = 0
    s = 1

    skip = False
    i = 0
    while i < x:
        j = 0
        if i // 14 > 63:
            break
        t_h = DB_list[i // 14]
        if t_h == "0":
            h = 0
        elif t_h == "1":
            h = 120
        elif t_h == "2":
            h = 240
        v = 0.5 - 0.5 * math.cos(2 * math.pi * 64 / 896 * i)
        r, g, b = hsv2rgb(h, s, v)
        while j < y:
            c.putpixel([i, j], (r, g, b))
            j = j + 1
        i += 1

    c.show()
    c.save("structured_light.bmp")
