import numpy as np
from PIL import Image
from Post_process import*

def isRectIntersect(D1,D2):
    '''
    Determine whether the 2 rectangle area are intersect.
    2 rect must be parallel
    
    D1: Boundary coordinates of area1. Array type, np.array([xmin,xmax,ymin,ymax])
    D2: Boundary coordinates of area2. Array type, np.array([xmin,xmax,ymin,ymax])
    '''
    return not (D1[0] >= D2[1] or D1[2] >= D2[3] or D2[0] >= D1[1] or D2[2] >= D1[3])

def areaIntersect(D1,D2):
    '''
    Calculate the intersect area of 2 rectangle. Return the coordinates of intersect area.
    2 rect must be parallel
    
    D1: Boundary coordinates of area1. Array type, np.array([xmin,xmax,ymin,ymax])
    D2: Boundary coordinates of area2. Array type, np.array([xmin,xmax,ymin,ymax])
    '''
    return


def boundary_rect(X,Y,B):
    '''
    Setting the Rectangle boundary.
    
    Parameters
    ===========
    X: Grid matrix of computational domain
    Y: Grid matrix of computational domain
    B: Boundary coordinates of rect. Array type, np.array([xmin,xmax,ymin,ymax])

    Returns
    ===========
    Mat_e : External area, Logical matrix shaped like X and Y
    Mat_b : Boundary area, Logical matrix shaped like X and Y
    Mat_i : Inside area, Logical matrix shaped like X and Y
    ''' 
    # Rectangle boundary
    dx = abs(X[0,1] - X[0,0])
    dy = abs(Y[1,0] - Y[0,0])
    Mat_1,Mat_2,Mat_3,Mat_4 = X<=B[0],X>=B[1],Y<=B[2],Y>=B[3]
    Mat_1b = abs(X-B[0])<=0.5*dx * ~(Mat_3+Mat_4)
    Mat_2b = abs(X-B[1])<=0.5*dx * ~(Mat_3+Mat_4)
    Mat_3b = abs(Y-B[2])<=0.5*dy * ~(Mat_1+Mat_2)
    Mat_4b = abs(Y-B[3])<=0.5*dy * ~(Mat_1+Mat_2)
    Mat_e = Mat_1+Mat_2+Mat_3+Mat_4
    Mat_b = Mat_1b+Mat_2b+Mat_3b+Mat_4b
    Mat_i = ~(Mat_e + Mat_b)
    return Mat_e, Mat_b, Mat_i


def boundary_circle(X,Y,B):
    '''
    Setting the Circle boundary.
    
    Parameters
    ===========
    X: Grid matrix of computational domain
    Y: Grid matrix of computational domain
    B: Center coordinates of circle. Array type, np.array([x0,y0,R])

    Returns
    ===========
    Mat_e : External area, Logical matrix shaped like X and Y
    Mat_b : Boundary area, Logical matrix shaped like X and Y
    Mat_i : Inside area, Logical matrix shaped like X and Y
    ''' 
    dx = abs(X[0,1] - X[0,0])
    dy = abs(Y[1,0] - Y[0,0])
    Mat_e = (X-B[0])**2 + (Y-B[1])**2 >= B[2]**2
    Mat_b = abs(((X-B[0])**2 + (Y-B[1])**2)**0.5 - B[2]) <= 0.5*min(dx,dy)
    Mat_i = ~(Mat_e + Mat_b)
    return Mat_e, Mat_b, Mat_i


