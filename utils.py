
import numpy as np
from numpy import random
from aspire.utils import coor_trans,Rotation



def generateBallVoxel(center,radius,L):
    
    grid = coor_trans.grid_3d(L)#,indexing='xyz')
    voxel = ((grid['x']-center[0])**2 + (grid['y']-center[1])**2 + (grid['z']-center[2])**2) <= np.power(radius,2)
    
    return np.single(voxel.reshape((1,L**3)))
    

def generateCylinderVoxel(center,radius,L,axis = 2):
    
    grid = coor_trans.grid_3d(L)#,indexing='xyz')
    dims= ('x','y','z')
    cylinder_axes = tuple(dims[i] for i in range(3) if i != axis)
    voxel = ((grid[cylinder_axes[0]]-center[0])**2 + (grid[cylinder_axes[1]]-center[1])**2) <= np.power(radius,2)
    
    return np.single(voxel.reshape((1,L**3)))


def rademacherDist(sz):
    val = random.randint(0,2,sz)
    val[val == 0] = -1
    return val

    
        


if __name__ == "__main__":
    L = 15
    voxel = generateBallVoxel([0,0,0],0.5,15)
    