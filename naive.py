import numpy as np
import cv2
import sys
import argparse

def cost(l, r):
    return np.abs(l-r)

parser = argparse.ArgumentParser(description='Naive implementation of local algorithm for stereo matching')
parser.add_argument('left_image_path', metavar='Left Image Path', nargs='?', help='Left Image input')
parser.add_argument('right_image_path', metavar='Right Image Path', nargs='?', help='Right Image input')
parser.add_argument('output_path', metavar='Output Image Path', nargs='?', help='Output Image input')

args = parser.parse_args()

#Loading images in grayscale
left = cv2.imread(args.left_image_path, cv2.IMREAD_GRAYSCALE)
right = cv2.imread(args.right_image_path, cv2.IMREAD_GRAYSCALE)
rows, cols = left.shape

#Setting disparity levels
dmax = 64

#Setting wether to use aggregation or not
aggregation = True
aggregationbox = (21,21)

#Computing costs
DSI = np.zeros((rows, cols, dmax), dtype=np.uint8)
for d in range(dmax):
    M = np.float32([[1,0,d],[0,1,0]])
    dst = cv2.warpAffine(right,M,(cols,rows))
    DSI[:,:,d] = cost(left, dst)
    print("Computed all costs for disparity " + str(d))

#Aggregating costs
if(aggregation):
    kernel = np.ones(aggregationbox, dtype=np.uint8)
    for d in range(dmax):
        tmp = DSI[:,:,d]
        out = cv2.filter2D(tmp, -1, kernel)
        DSI[:,:,d] = out
        print("Aggregated costs for disparity " + str(d))

#Optimization (WTA)
DM = np.argmin(DSI, axis=2)

#Saving disparity MAP
cv2.imwrite(args.output_image_path,DM)
