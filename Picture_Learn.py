import math


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

if __name__=="__main__":
    h=hsv(30,83,40)
    print(h)
    result = max(h[0], h[1], h[2])
    k = h[3] - math.sqrt((1 - result ** 2))
    print(k)