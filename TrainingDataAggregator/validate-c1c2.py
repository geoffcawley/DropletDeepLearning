import os
import sys
import cv2
from glob import glob

indir = sys.argv[1]

c1infile = open(os.path.join(indir, 'c1box.csv'), 'r')
c2infile = open(os.path.join(indir, 'c2box.csv'), 'r')

while True:
    c1row = c1infile.readline()
    c2row = c2infile.readline()
    if not (c1row or c2row):
        break
    
    c1row = c1row.split(',')
    c2row = c2row.split(',')
    
    filename = c1row[0]
    c1box = c1row[1:]
    c2box = c2row[1:]
    
    image = cv2.imread(filename)
    cv2.rectangle(image, (int(c1box[0]),int(c1box[1])), (int(c1box[2]),int(c1box[3])), (0,255,0), 2)
    cv2.rectangle(image, (int(c2box[0]),int(c2box[1])), (int(c2box[2]),int(c2box[3])), (0,0,255), 2)
    cv2.imshow("bounding box to circlebox",image)
    key = cv2.waitKey(0)
    if key==27:
        break