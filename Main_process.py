import time
from Geometry import *
from HeatTransfer2D import *
from Post_process import*
from threading import Thread,active_count


if __name__ == "__main__":
    a = 1
    D = np.array([0,20,0,20])
    B = np.array([1,10,1,14])
    Mx = 200
    My = 100
    T,Tn = 20,10
    Nt = 20000

    Frame = 1000
    fq = 10
    u0 = 20
    u_env = 20
    ch = 0.1

    ### set boundary
    x1 = np.linspace(D[0],D[1],Mx+1)
    y1 = np.linspace(D[2],D[3],My+1)
    X,Y = np.meshgrid(x1,y1)
    # Layer boundary
    Mat_e, Mat_b, Mat_i = boundary_rect(X,Y,B)
    # print(Mat_e.shape,Mat_b.shape) 
    explicit_difference_euler(a,fq,ch,T,Tn,u0,u_env,D,Mx,My,Nt,Mat_e,Mat_b,Mat_i,frame=Frame,plot=False)

