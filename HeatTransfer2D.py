import os
import time
import numpy as np
from Post_process import *
from Geometry import *


def boundary_convection(matrix,mat_bc,mat_in,uf,Sv=0,coeh=0.1):
    #对流换热边界
    matrix[mat_bc] = (np.min(matrix[mat_in]) + coeh*uf + Sv) / (1+coeh)
    return
    
def explicit_difference_euler(a,fq,coeh,T,Tn,u0,u_env,D,Mx,My,Nt,Mat_e,Mat_b,Mat_i,frame=100,plot=False,order=1):
    """
    Solve 2DHeatTransfer problem △(U/t) = a*△(△U/(x,y)) + fq 
    Use 1-order & 2-order precision euler mathor

    Parameters
    ----------
    a : a = k / (den*Cp)              Coefficient of function.
        k   is the Thermal Conductivity, Unit: W/(m°C)
        den is the Density, Unit: kg/m3
        Cp  is the Specific heat capacity, Unit: J/(kg°C)
    fq : fq = Qv / (den*Cp)            Constant coefficient of function.
        Qv  is the Volume Heat Flux, Unit: W/m3
    coeh: coeh = Thermal Conductivity / Convection heat transfer coefficient, Unit: m
    T  : Total calculating time. Unit: s
    Tn : Cycle time. Adding a new layer after Tn. Unit: s
    u0 : Initial temperature at t=0. Unit: °C
    u_env : Environment temperature. Unit: °C, Constance
    D  : Diameter of the calculation area, array[xmin,xmax,ymin,ymax]
    Mx : Segment number of x-axis 
    My : Segment number of y-axis 
    Nt : Segment number of Total time
    frame: Frequency of results to save
    plot: Whether to display cloud image results in real time
    order: Method order, 1: Forward Euler Method, 2: Improved Euler method
    """
    
    dx = (D[1]-D[0])/Mx
    dy = (D[3]-D[2])/My
    dt = T/Nt


    #initial condition 
    U = u0*np.ones((My+1,Mx+1))
    # y方向二阶导系数矩阵A 
    A = (-2)*np.eye(Mx+1,k=0) + (1)*np.eye(Mx+1,k=-1) + (1)*np.eye(Mx+1,k=1)
    # x方向二阶导系数矩阵B 
    B = (-2)*np.eye(My+1,k=0) + (1)*np.eye(My+1,k=-1) + (1)*np.eye(My+1,k=1)

    rx,ry,ft = a/dx**2, a/dy**2, fq
    heat = 1
    start = time.time()
    for k in range(int(Nt+1)):
        tt = k*dt
        # tt>Tn，stop heating
        if tt>Tn: heat=0
        ### Boundary conditions heres
        Umax = U.max() 
        Umin = U.min() 
        # solve inside nodes
        if (order==1):
            U = U + dt * (rx*np.dot(U,A) + ry*np.dot(B,U) + heat*fq)
        elif (order==2):
            K1 = rx*np.dot(U,A) + ry*np.dot(B,U) + heat*fq
            K2 = rx*np.dot(U+K1*dt,A) + ry*np.dot(B,U+K1*dt) + heat*fq
            U = U + 0.5 * dt * (K1 + K2)

        # solve outside nodes
        U[Mat_e] = u_env
        # solve boundary nodes
        boundary_convection(U,Mat_b,Mat_e,u_env,Sv=0,coeh=coeh) 

        if k%frame == 0:
            # save data here
            end = time.time()
            print('T = {:.3f} s    max_U = {:.3f}  min_U = {:.3f}  Heat = {:.2f}  Uenv = {:.1f}'.format(tt,Umax,Umin,heat,u_env))
            if plot:
                showcontourf(U,D,vmin=20,vmax=100)
    return



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
    explicit_difference_euler(a,fq,ch,T,Tn,u0,u_env,D,Mx,My,Nt,Mat_e,Mat_b,Mat_i,frame=Frame,order=1)
    explicit_difference_euler(a,fq,ch,T,Tn,u0,u_env,D,Mx,My,Nt/4,Mat_e,Mat_b,Mat_i,frame=Frame/4,order=1)