import numpy as np
import re
from matplotlib import pyplot as plt
fh=open('Kalmann_final.txt','a')
f = open("kalmann.txt", "r")
Lines = f.readlines()
digit1=re.findall(r"[-+]?(?:\d*\.\d+|\d+)",Lines[0])

x_before=float(digit1[0])
y_before=float(digit1[1])


H=np.matrix('1 0')


qx=np.matrix('1 0;0 1')
rx=0.05


qy=np.matrix('1 0;0 1')
ry=0.05

Fx=np.matrix('1 1;0 1')
Px=np.matrix('0.01 0;0 0.01')
Fy=np.matrix('1 1;0 1')
Py=np.matrix('0.01 0;0 0.01')
i=1
x_list=[]
y_list=[]
while(i<len(Lines)):
    digit = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", Lines[i])
    v_x=float(digit[2])
    v_y=float(digit[3])
    x_estim=float(digit[0])
    y_estim=float(digit[1])
    x_prev = np.matrix([[x_before], [v_x]])
    y_prev = np.matrix([[y_before], [v_y]])
    x_guess = np.matmul(Fx, x_prev)
    y_guess = np.matmul(Fy, y_prev)

    Px = np.matmul(np.matmul(Fx, Px), Fx.transpose()) + qx
    Py = np.matmul(np.matmul(Fy, Py), Fx.transpose()) + qy

    Sx = (np.matmul(np.matmul(H, Px), H.transpose()) + rx)
    Sy = (np.matmul(np.matmul(H, Px), H.transpose()) + ry)
    Kx = np.matmul(Px, H.transpose()) / (np.matmul(np.matmul(H, Px), H.transpose()) + rx)
    Ky = np.matmul(Py, H.transpose()) / (np.matmul(np.matmul(H, Py), H.transpose()) + ry)

    x_final = x_guess + np.matmul(Kx, (x_estim - np.matmul(H, x_guess)))
    y_final = y_guess + np.matmul(Kx, (y_estim - np.matmul(H, y_guess)))

    x_list.append(x_final[0,0])
    y_list.append(y_final[0,0])

    Px = np.matmul((np.identity(2) - np.matmul(Kx, H)), Px)
    Py = np.matmul((np.identity(2) - np.matmul(Ky, H)), Py)

    print(f"step{i}: x={x_final[0,0]} y={y_final[0,0]} \n")
    Error_x=(1-Kx[0,0])*Sx[0,0]
    Error_y = (1 - Ky[0, 0]) * Sy[0, 0]
    print(f"Accuracy in X={1-Error_x}  Accuracy in Y={1-Error_y}\n\n")

    x_before=x_final[0,0]
    y_before = y_final[0,0]


    fh.write(f"step{i}: x={x_final[0,0]} y={y_final[0,0]} \nAccuracy in X={1-Error_x}  Accuracy in Y={1-Error_y}\n\n")

    i=i+1


fh.close()
plt.title("Y vs X")
plt.xlabel("x axis ")
plt.ylabel("y axis ")
plt.plot(x_list,y_list)
plt.show()
