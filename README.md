# pyHeatTransfer
Use explicit method to solve unsteady heat conduction problems.
At present, 2D heat conduction solution has been realized and updated.

# HeatTransfer2D.py
Solve 2D problem like △(U/t) = a*△(△U/(x,y)) + fq 

## Mathors
    1 1-order precision euler mathor
    2 2-order precision euler mathor
    3 4-order precision Runge-Kutta method
    4 ...(more solver to be support) 
## Parameters
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

# Geometry and Post
Geometry.py define some boundary condition of simple geometry————triangle/rectangle/circle/etc. Post_process.py define some function to show and save the result during the solution process

# More details
For more infomation please go to my zhihu webpage https://zhuanlan.zhihu.com/p/280084433  &  https://zhuanlan.zhihu.com/p/282878402