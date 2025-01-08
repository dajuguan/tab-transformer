import numpy as np
from math import tan
import matplotlib.pyplot as plt

import scienceplots
defaultTicks =  {'xtick.top':False,'ytick.right':False}
plt.style.use(['science','ieee','no-latex',defaultTicks])

def one_bezier_curve(a,b,t):
    return (1-t)*a + t*b

def n_bezier_curve(xs,n,k,t):
    if n == 1:
        return one_bezier_curve(xs[k],xs[k+1],t)
    else:
        return (1-t)*n_bezier_curve(xs,n-1,k,t) + t*n_bezier_curve(xs,n-1,k+1,t)
 
def bezier_curve(xs,ys,num):
    n = len(xs) - 1
    t = np.linspace(0.0,1,num,endpoint=True)
    b_xs = []
    b_ys = []
    for each in t:
        b_xs.append(n_bezier_curve(xs,n,0,each))
        b_ys.append(n_bezier_curve(ys,n,0,each))
    return np.array(b_xs), np.array(b_ys)

def stacking_line(root_angle, root_h, tip_angle, tip_h, extend_percent=1/3):
    print("root angle: %.2f, root_h: %.2f,tip angle: %.2f, tip_h: %.2f", root_angle, root_h, tip_angle, tip_h)
    root_angle = - root_angle
    x1, y1 = 0, 0
    x2 = root_h * tan(root_angle / 180 * 3.14)
    y2 = root_h
    x3 = x2
    y3 = root_h * (1+ extend_percent)

    hub_x, hub_y = bezier_curve([x1,x2,x3], [y1, y2, y3], 20)
    x4 = x3
    y4 = 1 - tip_h * (1+ extend_percent)
    x5 = x4
    y5 = 1- tip_h
    x6 = x5 + tip_h * tan(tip_angle / 180 * 3.14)
    y6 = 1

    shroud_x, shroud_y = bezier_curve([x4,x5,x6], [y4, y5, y6], 20)
    return hub_x.tolist() + shroud_x.tolist(), hub_y.tolist() + shroud_y.tolist()


if __name__ == '__main__':
    X, Y = stacking_line(0, 0, -11, 0.3, extend_percent=1/3)
    plt.plot(X,Y)
    plt.xlim([-0.5,0.5])
    plt.ylim([0,1])
    plt.savefig("./stacking_line_test.png")
 