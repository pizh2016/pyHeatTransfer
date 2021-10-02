import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def showmatrixlist(matlist,cmap=plt.cm.get_cmap('jet'),fsize=(12,12),vmin=0, vmax=100):
    plt.close()
    xn=len(matlist)
    plt.figure(figsize=fsize)
    for i in range(xn):
        plt.subplot(1,xn,i+1)
        plt.imshow(matlist[i], cmap=cmap, origin = 'lower',vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()
    return
    

def showcontourf(mat,D,cmap=plt.cm.get_cmap('jet'),fsize=(12,12),vmin=0, vmax=100):
    plt.clf()
    levels = np.arange(vmin,vmax,1)
    x=np.linspace(D[0],D[1],mat.shape[1])
    y=np.linspace(D[2],D[3],mat.shape[0])
    X,Y=np.meshgrid(x,y)
    z_max = np.max(mat)
    i_max,j_max = np.where(mat==z_max)[0][0], np.where(mat==z_max)[1][0]
    show_max = "U_max: {:.1f}".format(z_max)
    plt.plot(x[j_max],y[i_max],'ro') 
    plt.contourf(X, Y, mat, 100, cmap = cmap, origin = 'lower', levels = levels)
    plt.annotate(show_max,xy=(x[j_max],y[i_max]),xytext=(x[j_max],y[i_max]),fontsize=14)
    plt.colorbar()
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.axis('equal')
    plt.draw()
    plt.pause(0.001)
    plt.clf()

def savecontourf(matlist,D,fname,cmap=plt.cm.get_cmap('jet'),fsize=(12,12),vmin=0, vmax=100):
    plt.close()
    plt.clf()
    levels = np.arange(vmin,vmax,1)
    x=np.linspace(D[0],D[1],mat.shape[1])
    y=np.linspace(D[2],D[3],mat.shape[0])
    X,Y=np.meshgrid(x,y)
    z_max = np.max(mat)
    i_max,j_max = np.where(mat==z_max)[0][0], np.where(mat==z_max)[1][0]
    show_max = "U_max: {:.1f}".format(z_max)
    plt.plot(x[j_max],y[i_max],'ro') 
    plt.contourf(X, Y, mat, 100, cmap = cmap, origin = 'lower', levels = levels)
    plt.annotate(show_max,xy=(x[j_max],y[i_max]),xytext=(x[j_max],y[i_max]),fontsize=14)
    plt.colorbar()
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.savefig(fname,dpi=100)
    


if __name__ == "__main__":
    A = np.array([0,50,100])
    showmatrixlist([np.tile(A,(12,4))])