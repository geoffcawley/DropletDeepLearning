import random
import cv2
import numpy as np

# circles in the form (x, y, r), debugmode=True shows images
def rotateImageAndCircles(image, c1, c2, degrees, debugmode=False):
    (h, w) = image.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    cm = np.array([cx,cy]).astype(float)
    
    m = np.array([[float(c1[0]), float(c2[0])],
                  [float(c1[1]), float(c2[1])]])
    M = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)
    rotatedimg = cv2.warpAffine(image, M, (w, h))
    
    theta = np.radians(degrees)
    
    rotMatrix = np.array([[np.cos(theta), np.sin(theta)],
                          [-np.sin(theta), np.cos(theta)]])
    c1rm = np.matmul(rotMatrix, (m[:,0].astype(float)-cm))+cm
    c2rm = np.matmul(rotMatrix, (m[:,1].astype(float)-cm))+cm
    
    c1 = [round(c1rm[0]), round(c1rm[1]),int(c1[2])]
    c2 = [round(c2rm[0]), round(c2rm[1]),int(c2[2])]
    c3 = c2
    if c1[0] > c2[0]:
        c2=c1
        c1=c3
    
    if debugmode is True:
        dispimg = rotatedimg
        cv2.circle(dispimg, (c1[0],c1[1]),c1[2],
            (0, 255, 0))
        cv2.circle(dispimg, (c2[0],c2[1]),c2[2],
            (0, 0, 255))
        dispimg = cv2.resize(dispimg, (600,600))
        windowname = f"Rotated by {degrees} Degrees"
        cv2.imshow(windowname, dispimg)
        cv2.waitKey(0)
        cv2.destroyWindow(windowname)
    
    return rotatedimg, c1, c2


def randomizeBox(box, dims, randrange):
    (bx1, by1, bx2, by2) = box
    
    xt = random.randrange(randrange) - int(randrange/2)
    yt = random.randrange(randrange) - int(randrange/2)
    border = int((random.randrange(randrange) - int(randrange/2))/2)
    bx1 += xt - border
    bx2 += xt + border
    by1 += yt - border
    by2 += yt + border
    # x += xt
    # y += yt
    
    box = (bx1, by1, bx2, by2)
    return box

def validateBox(box, dims):
    if box[0] < 0 or box[2] < 0:
        return False
    if box[1] > dims[1] or box[3] > dims[0]:
        return False
    return True

def getBoundingSquare(bbox, h, w):
    (bbx1, bby1, bbx2, bby2) = bbox
    (bcx, bcy) = ((bbx1+bbx2)/2, (bby1+bby2)/2)
    halfside = max([bbx2-bbx1, bby2-bby1])/2
    (bsx1, bsy1, bsx2, bsy2) = (int(bcx-halfside),
                                int(bcy-halfside), int(bcx+halfside), int(bcy+halfside))
    if bsx1 < 0:
        bsx2 = int(2*halfside)
        bsx1 = 0
    if bsy1 < 0:
        bsy2 = int(2*halfside)
        bsy1 = 0
    if bsx2 >= w:
        bsx1 -= (bsx2-(w-1))
        bsx2 = w-1
    if bsy2 >= h:
        bsy1 -= (bsy2-(h-1))
        bsy2 = h-1
        
    return (bsx1, bsy1, bsx2, bsy2)

def ensureBoxWithinImage(box, dims):
    (h, w) = dims
    if box[0] < 0:
        d = -box[0]
        box[0] = 0
        box[2] += d
    if box[1] < 0:
        d = -box[1]
        box[1] = 0
        box[3] += d
    if box[2] >= w:
        d = box[2] - w
        box[2] = w
        box[0] -= d
    if box[3] >= h:
        d = box[3] - h
        box[3] = h
        box[1] -= d
        
    return box