import os
import sys
import cv2
import numpy as np
from glob import glob

indir = sys.argv[1]
outdir = sys.argv[2]
slices = 2

for imname in glob(f"{indir}\\*.jpg"):
    fname = imname.replace('.jpg','.txt')
    infile = open(fname, "r")
    s = infile.read()
    (fname, c1x, c1y, c1r, c2x, c2y, c2r) = s.split(',')
    m = np.array([[float(c1x),float(c2x)],[float(c1y),float(c2y)]])
    image = cv2.imread(imname)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    cM = np.array([cX,cY]).astype(float)
    
    angle = 0.
    inc = 360. / float(slices)
    while angle < 360:
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotatedimg = cv2.warpAffine(image, M, (w, h))
                
        theta = np.radians(angle)

        rotMatrix = np.array([[np.cos(theta), np.sin(theta)], 
                                [-np.sin(theta),  np.cos(theta)]])
        c1rm = np.matmul(rotMatrix,(m[:,0].astype(float)-cM))+cM
        c2rm = np.matmul(rotMatrix,(m[:,1].astype(float)-cM))+cM

        c1 = [round(c1rm[0]), round(c1rm[1]),int(c1r)]
        c2 = [round(c2rm[0]), round(c2rm[1]),int(c2r)]
        if c1[0] > c2[0]:
            c3 = c2
            c2=c1
            c1=c3
        dispimg = rotatedimg.copy()
        cv2.circle(dispimg, (c1[0],c1[1]),c1[2],
            (0, 255, 0))
        cv2.circle(dispimg, (c2[0],c2[1]),c2[2],
            (0, 0, 255))
        dispimg = cv2.resize(dispimg, (600,600))
        # cv2.imshow(f"Rotated by {angle} Degrees", dispimg)
        # cv2.waitKey(0)

        outname = f'{os.path.basename(imname).replace('.jpg','')}_{angle}'
        cv2.imwrite(f'{os.path.join(outdir,outname)}.jpg', rotatedimg)
        outfile = open(f'{os.path.join(outdir,outname)}.txt', 'w')
        outfile.write(f'{outname}.jpg,{c1[0]},{c1[1]},{c1[2]},{c2[0]},{c2[1]},{c2[2]}')
        outfile.close()

        outname = f'{os.path.basename(imname).replace('.jpg','')}_{angle}_hf'
        flippedimg = cv2.flip(rotatedimg, 1)
        c1[0] = cX - (c1[0]-cX)
        c2[0] = cX - (c2[0]-cX)
        if c1[0] > c2[0]:
            c3 = c2
            c2=c1
            c1=c3
        cv2.imwrite(f'{os.path.join(outdir,outname)}.jpg', flippedimg)
        outfile = open(f'{os.path.join(outdir,outname)}.txt', 'w')
        outfile.write(f'{outname}.jpg,{c1[0]},{c1[1]},{c1[2]},{c2[0]},{c2[1]},{c2[2]}')
        outfile.close()
        angle += inc
