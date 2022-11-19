# RADIUS_MEASUREMENT   #
#   SHIVAM AMBOKAR     #


# ME338 Course Project #
# GROUP: WOMAN-PRO     #
import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow

directory = '/content/NewF'
ref_dim = 0.05 #Reference dimension of the fillet radius (undeformed & intact)
for filename in os.listdir(directory):  #iterating over all the files
  f = os.path.join(directory, filename)
  if os.path.isfile(f):
    try:
      image = cv2.imread(f)
      print(filename, ": SUCCESS -")
      ht,wt = image.shape[:2]
      image = image[int(0.02*ht):int(0.98*ht),int(0.035*wt):int(0.6*wt)]
                #Cropping the image
      img = image
      img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
      mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
      mask = cv2.bitwise_or(mask1, mask2 )
      cropped = cv2.bitwise_and(img, img, mask=mask)
      im_gray = mask
      thim = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,1)
      bi = cv2.bilateralFilter(im_gray, 5, 75, 75)
      bi2 = cv2.GaussianBlur(bi, (5,5),0) #Smoothening of the boundary
      dst = cv2.cornerHarris(bi2, 2, 3, 0.04)  #Corner Detection
      newmask = np.zeros_like(im_gray)
      newmask[dst>0.01*dst.max()] = 255
      newarray = np.argwhere(newmask == 255)
      from sklearn.cluster import KMeans
      import numpy as np
      import pandas as pd
      X=newarray
      # BELOW: Separating the two corners
      kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
      kmeans.labels_
      cluster_map = pd.DataFrame(X)
      cluster_map['cluster'] = kmeans.labels_
      cluster_map['distO'] = cluster_map[0] #15*cluster_map[0]*cluster_map[0]+cluster_map[1]*cluster_map[1]
      cluster_map[cluster_map['cluster'] == 1 ].iloc[0,0]
      c1 = cluster_map[[0,1]][(cluster_map['cluster'] == 1)&(cluster_map['distO'] == cluster_map["distO"][cluster_map['cluster'] == 1 ].min())].iloc[0,[0,1]]
      c1 = np.array(c1)
      c2 = cluster_map[[0,1]][(cluster_map['cluster'] == 0)& (cluster_map['distO'] == cluster_map["distO"][cluster_map['cluster'] == 0 ].min())].iloc[0,[0,1]]
      c2 = np.array(c2)
      a = cluster_map[[0,1]][(cluster_map['cluster'] == 1)]
      b = cluster_map[[0,1]][(cluster_map['cluster'] == 0)]
      cross_p = [(x1,y1,x2,y2) for (x1,y1) in np.array(a) for (x2,y2) in np.array(b)]
      cross_p = pd.DataFrame(cross_p)
      cross_p['dist'] = np.square(cross_p[0]-cross_p[2])+np.square(cross_p[1]-cross_p[3])
      corners = cross_p[[0,1,2,3]][cross_p['dist']==cross_p['dist'].max()]
      corner1 = corners[[0,1]]
      corner2 = corners[[2,3]]
      contourdat = np.argwhere(thim== 0)
      X=contourdat
      contourmap = pd.DataFrame(X)
      c20 = max(c2[0],c1[0])
      c10 = min(c1[0],c2[0])
      dy = c20 - c10
      dy = dy/4
      y = c10+dy
      x = min(c1[1],c2[1])
      contourmap["dist"] = np.absolute(np.sqrt(np.square((contourmap[0]-y)) + np.square((contourmap[1]-x))))
      pt2 = contourmap[:][(contourmap[0] < y+0.03*dy) &  (contourmap[0] > y-0.03*dy)].min().iloc[[0,1]]
      y = c10+2*dy
      contourmap["dist"] = np.absolute(np.sqrt(np.square((contourmap[0]-y)) + np.square((contourmap[1]-x))))
      pt1 = contourmap[:][(contourmap[0] < y+0.03*dy) &  (contourmap[0] > y-0.03*dy)].min().iloc[[0,1]]
      y = c10+3*dy
      contourmap["dist"] = np.absolute(np.sqrt(np.square((contourmap[0]-y)) + np.square((contourmap[1]-x))))
      pt4 = contourmap[:][(contourmap[0] < y+0.03*dy) &  (contourmap[0] > y-0.03*dy)].min().iloc[[0,1]]
      pt4 = np.array(pt4).astype(int)
      pt1 = np.array(pt1).astype(int)
      pt2 = np.array(pt2).astype(int)
      pt3 = c2
      pt5 = c1
      y = c10+2*dy
      contourmap["dist"] = np.absolute(np.sqrt(np.square((contourmap[0]-y)) + np.square((contourmap[1]-x))))
      contourmap[:][(contourmap[0] < y+0.03*dy) &  (contourmap[0] > y-0.03*dy)]
      det0 = np.linalg.det([[pt1[1], -pt1[0], 1],[pt4[1], -pt4[0], 1],[pt5[1], -pt5[0], 1]])
      d1 = 0.5*(np.linalg.det([[pt1[1]*pt1[1] + pt1[0]*pt1[0], -pt1[1], 1],[pt5[1]*pt5[1] + pt5[0]*pt5[0], -pt5[1], 1],[pt4[1]*pt4[1] + pt4[0]*pt4[0], -pt4[1], 1]]))/det0
      e1 = 0.5*(np.linalg.det([[pt1[1]*pt1[1] + pt1[0]*pt1[0], pt1[0], 1],[pt5[1]*pt5[1] + pt5[0]*pt5[0], pt5[0], 1],[pt4[1]*pt4[1] + pt4[0]*pt4[0], pt4[0], 1]]))/det0
      det0 = np.linalg.det([[pt1[1], -pt1[0], 1],[pt2[1], -pt2[0], 1],[pt3[1], -pt3[0], 1]])
      d2 = -0.5*(np.linalg.det([[pt1[1]*pt1[1] + pt1[0]*pt1[0], -pt1[1], 1],[pt2[1]*pt2[1] + pt2[0]*pt2[0], -pt2[1], 1],[pt3[1]*pt3[1] + pt3[0]*pt3[0], -pt3[1], 1]]))/det0
      e2 = -0.5*(np.linalg.det([[pt1[1]*pt1[1] + pt1[0]*pt1[0], pt1[0], 1],[pt2[1]*pt2[1] + pt2[0]*pt2[0], pt2[0], 1],[pt3[1]*pt3[1] + pt3[0]*pt3[0], pt3[0], 1]]))/det0
      d = (d1); e = (e1)
      r_ref =  np.sqrt( np.square(pt1[1] - e) + np.square(pt1[0] - d) ) + np.sqrt( np.square(pt2[1] - e) + np.square(pt2[0] - d) )+ np.sqrt( np.square(pt3[1] - e) + np.square(pt3[0] - d) )
      r_ref = r_ref/3
      contourmap["dist"] = np.sqrt((contourmap[0]-d)*(contourmap[0]-d) + (contourmap[1]-e)*(contourmap[1]-e))
      maxdist = (contourmap[contourmap["dist"] == contourmap["dist"].max()].iloc[0,2])
      with open('document.csv','a') as fd:
          fd.write(str(filename) +","+ str(ref_dim*(maxdist-r_ref)/r_ref)+"\n")
    except:
      print(filename, ":", " ERROR")
