import pandas as pd
import starfile
import sys

def remove_pose_from_star(input,output):
    star = starfile.read(input)
    star['particles'] = star['particles'].drop(columns=['rlnAngleRot','rlnAngleTilt','rlnAnglePsi','rlnOriginXAngst','rlnOriginYAngst','rlnOriginX','rlnOriginY'],errors='ignore')
    starfile.write(star,output)

if __name__ == "__main__":
    remove_pose_from_star(sys.argv[1],sys.argv[2])