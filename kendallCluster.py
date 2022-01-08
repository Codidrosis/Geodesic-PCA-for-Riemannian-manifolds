# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 02:45:57 2021

@author: lahza
"""
import time
import numpy as np
import scipy.io
import math
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt 
import random
class PreShape():
    def __init__(self, k_landmarks, m_ambient):
        self.k_landmarks = k_landmarks
        self.m_ambient = m_ambient
    def center(self,point):
        """Center landmarks around 0.
        Parameters
        ----------
        points in Matrices space. : array-like, shape=[..., k_landmarks, m_ambient]
        Returns
        -------
        Points centered : array-like, shape=[..., k_landmarks, m_ambient]
        """
        mean = np.mean(point, axis=-2)
        return point - mean[..., None, :]
    
    def is_centered(self,point):
        """Check that landmarks are centered around 0.
        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point in Matrices space.
        Returns
        -------
        is_centered : array-like, shape=[...,]
            Boolean evaluating if point is centered.
        """
        atol=np.abs(np.min(point))*1e-7    #Tolerance at which to evaluate mean == 0.
        mean = np.mean(point, axis=-2)
        return np.all(np.isclose(mean, 0., atol=atol), axis=-1)
    def PreshapeElement(self, point):    
        """Project a point on the pre-shape space.
        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point in Matrices space.
        Returns
        -------
        projected_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point projected on the pre-shape space.
        """
        centered_point = self.center(point)        
        frob_norm=np.array([np.linalg.norm(point[i,...],ord='fro') for i in range(point.shape[0])])
        projected_point = np.einsum(
            '...,...ij->...ij', 1. / frob_norm, centered_point)

        return projected_point
    
    def kendallPreshape(self, point):    
        """Project a point on the pre-shape space.
        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point in Matrices space.
        Returns
        -------
        projected_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point projected on the pre-shape space.
        """
        Q=scipy.linalg.helmert(self.k_landmarks, full=True).T
        if point.ndim==2:
            initdim=2
            point=np.expand_dims(point, axis=0)
        else:
            initdim=3
        if point.shape[-2]!=self.k_landmarks:
            point=np.transpose(point,axes=(0,2,1))
            init=-1
        else:
            init=1
        
        standBy=point
        point=np.array([np.matmul(point[i,:,:].T,Q) for i in range(point.shape[0])])
        projected_point=np.array([point[i,:,:].T-np.sum(standBy[i,:,:],axis=0)/(np.sqrt(self.k_landmarks)) for i in range(point.shape[0])])
        kendallmatrixRep=projected_point[:,1:,:]
        
        if init==-1:
            projected_point,kendallmatrixRep=np.transpose(projected_point,axes=(0,2,1)),np.transpose(kendallmatrixRep,axes=(0,2,1))
        if initdim==2:
            projected_point,kendallmatrixRep=projected_point[0,:,:],kendallmatrixRep[0,:,:] 
        return projected_point,kendallmatrixRep


    def flip_determinant(matrix, det):
        """Change sign of the determinant if it is negative.
        For a batch of matrices, multiply the matrices which have negative
        determinant by a diagonal matrix :math: `diag(1,...,1,-1) from the right.
        This changes the sign of the last column of the matrix.
        Parameters
        ----------
        matrix : array-like, shape=[...,n ,m]
            Matrix to transform.
        det : array-like, shape=[...]
            Determinant of matrix, or any other scalar to use as threshold to
            determine whether to change the sign of the last column of matrix.
        Returns
        -------
        matrix_flipped : array-like, shape=[..., n, m]
            Matrix with the sign of last column changed if det < 0.
        """
        if np.any(det < 0):
            ones = np.ones(matrix.shape[-1])
            reflection_vec = np.concatenate(
                [ones[:-1], np.array([-1.])], axis=0)
            mask=det < 0
            mask=mask.astype( matrix.dtype)
            sign = (mask[..., None] * reflection_vec
                    + (1. - mask)[..., None] * ones)
            return np.einsum('...ij,...j->...ij', matrix, sign)
        return matrix

    def align(self, point, base_point, **kwargs):
        """Align point to base_point.
        Find the optimal rotation R in SO(m) such that the base point and
        R.point are well positioned.
        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the manifold.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the manifold.
        Returns
        -------
        aligned : array-like, shape=[..., k_landmarks, m_ambient]
            R.point.
        """
        mat = np.matmul(np.transpose(point[...,:,:],axes=(0,2,1)), base_point)
        left, singular_values, right = np.linalg.svd(mat)
        det = np.linalg.det(mat)
        # conditioning = (
        # (singular_values[..., -2]
        #  + np.sign(det) * singular_values[..., -1]) /
        # singular_values[..., 0])
        # if np.any(conditioning < gs.atol):
        #     logging.warning(f'Singularity close, ill-conditioned matrix '
        #                     f'encountered: {conditioning}')
        # if np.any(gs.isclose(conditioning, 0.)):
        #     logging.warning("Alignment matrix is not unique.")
        #flipped = self.flip_determinant(np.transpose(right[...,:,:],axes=(0,2,1)), det)
        flipped = self.flip_determinant(np.transpose(right[...,:,:],axes=(0,2,1)), det)
        result = reduce(np.matmul, [point, left, np.transpose(flipped[...,:,:],axes=(0,2,1))])
        return result
    
    def align2(self, point, base_point, **kwargs):
        if point.ndim==2:
            point=np.expand_dims(point, axis=0)
        if point.ndim==3 and point.shape[0]>=1:
            mat = np.matmul(np.transpose(point[...,:,:],axes=(0,2,1)), base_point)
            left, singular_values, right = np.linalg.svd(mat)
            result = reduce(np.matmul, [right, np.transpose(left,axes=(0,2,1)),np.transpose(point,axes=(0,2,1))])
            result = np.transpose(result,axes=(0,2,1))
            return result        
        else:
            return -1
    def kendaldistanace(self,s1,s2):
        s1=self.align2(s1,s2)[0,:,:]
        return self.PreShapeDist(s1,s2)
    
    def PreShapeDist(self,s1,s2):
        if self.k_landmarks==s1.shape[1] and self.m_ambient==s1.shape[0]:
            s1=s1.T;    s2=s2.T
        p=np.matmul(np.transpose(s1),s2)
        return np.arccos(np.trace(p))
    
    
        
    def dist_pairwise(self, point, **kwargs):
        distmatrix=np.zeros((np.shape(point)[0],np.shape(point)[0]))
        for i in range(np.shape(point)[0]):
            for j in range(np.shape(point)[0]):
                p=np.matmul(np.transpose(point[j]),point[i])
                distmatrix[i,j]=np.arccos(np.trace(p))
        return distmatrix,np.argsort(distmatrix, axis=0)
        
    def dist_pairwiseRealign(self, points, **kwargs):
        distmatrix=np.zeros((np.shape(points)[0],np.shape(points)[0]))
        distmatrix2=np.zeros((np.shape(points)[0],np.shape(points)[0]))
        h=[]
        for i in range(points.shape[0]):
            base_point=points[i,:,:]
            k=[]
            for j in range(0,points.shape[0]):
                point=points[j:j+1,:,:]
                Aligned=self.align3(point,base_point)
                p=np.matmul(np.transpose(Aligned[0]),base_point)
                k.append(np.arccos(np.trace(p)))
                distmatrix2[i,j]=np.arccos(np.trace(p))
                if j>i:
                    distmatrix[i,j]=np.arccos(np.trace(p))
            h.append(np.argsort(np.array(k)))
            #print(np.argsort(np.array(k)),k[np.argsort(np.array(k))[0]],k[i],k[i+1])
        #distmatrix=distmatrix+distmatrix.T
        return distmatrix,np.argsort(distmatrix, axis=0),h
        
    def dist_pairwiseRealign(self, points, **kwargs):
        distmatrix=np.zeros((np.shape(points)[0],np.shape(points)[0]))
        distmatrix2=np.zeros((np.shape(points)[0],np.shape(points)[0]))
        h=[]
        for i in range(points.shape[0]):
            base_point=points[i,:,:]
            k=[]
            for j in range(0,points.shape[0]):
                point=points[j:j+1,:,:]
                Aligned=self.align3(point,base_point)
                p=np.matmul(np.transpose(Aligned[0]),base_point)
                k.append(np.arccos(np.trace(p)))
                distmatrix2[i,j]=np.arccos(np.trace(p))
                if j>i:
                    distmatrix[i,j]=np.arccos(np.trace(p))
            h.append(np.argsort(np.array(k)))
            #print(np.argsort(np.array(k)),k[np.argsort(np.array(k))[0]],k[i],k[i+1])
        #distmatrix=distmatrix+distmatrix.T
        return distmatrix,np.argsort(distmatrix, axis=0),h
    
mat = scipy.io.loadmat('contours.mat')
contours=mat.get('b')[0]
contours=np.array([contours[i] for i in range(contours.shape[0])])
a=PreShape(100,2)
contoursPre=a.PreshapeElement(contours)
def destroyCorrespandance(s1):
    return np.roll(s1,random.randint(0,90),axis=0)

def shiftpoints(s1,s2,number=1):
    minn=np.sum([np.linalg.norm(s1[i,:]-s2[i,:]) for i in range(number)])
    j=0
    for i in range(1,s1.shape[0]):
        s3=np.roll(s2,-i,axis=0)
        q=np.sum([np.linalg.norm(s1[i,:]-s3[i,:]) for i in range(number)])
        if minn>q:
            minn=q
            j=i
    return np.roll(s2,-j,axis=0)

contoursPre1=[];    contoursPre2=[];
s1=contoursPre[0];  s2=contoursPre[72]
print("it might take a minute")
for i in range(0,1400):
    print(i)
    s2=contoursPre[i]
    contoursPre1.append(shiftpoints(s1,s2,number=random.randint(0,50)))
    contoursPre2.append(destroyCorrespandance(s2))

contoursPre1=np.array(contoursPre1);
contoursPre2=np.array(contoursPre2);
Y=[]
for i in range(contours.shape[0]):
    Y.append(i//20)

print("end")
def align2(point, base_point, **kwargs):
    if point.ndim==2:
        point=np.expand_dims(point, axis=0)
    if point.ndim==3 and point.shape[0]>=1:
        mat = np.matmul(np.transpose(point[...,:,:],axes=(0,2,1)), base_point)
        left, singular_values, right = np.linalg.svd(mat)
        #print(left.shape,mat.shape)
        inter=np.eye(mat.shape[1]);
        inter[mat.shape[0]-1,mat.shape[0]-1]=-1
        for i in range(len(left)):
            if np.linalg.det(left[i])<0:
                left[i]=np.matmul(left[i],inter)
            if np.linalg.det(right[i])<0:
                right[i]=np.matmul(right[i],inter)
        result = reduce(np.matmul, [right, np.transpose(left,axes=(0,2,1)),np.transpose(point,axes=(0,2,1))])
        result = np.transpose(result,axes=(0,2,1))
        return result        
    else:
        return -1
def kendaldistanace(s1,s2):
    s1=align2(s1,s2)[0,:,:]
    return PreShapeDist(s1,s2)
def PreShapeDist(s1,s2):
    if s1.shape[0]>s1.shape[1]:
        s1=s1.T;    s2=s2.T
    p=np.matmul(np.transpose(s1),s2)
    return np.arccos(np.trace(p))

def PreShapeDistKNN(s1,s2):   
    #start_time = time.time()
    s1=align2(s1.reshape(99,2),s2.reshape(99,2))[0,:,:].T
    s2=s2.reshape(99,2).T;    
    
    p=np.matmul(np.transpose(s1),s2)
    #print(time.time()-start_time)
    if np.trace(p)>1 or np.trace(p)<-1:
        return 0
    return np.arccos(np.trace(p))

def PreShapeDistKNN2(s1,s2):
    s2=shiftpoints(s1.reshape(99,2),s2.reshape(99,2),number=1)
    s1=align2(s1,s2)[0,:,:].T
    s2=s2.reshape(99,2).T;    
    p=np.matmul(np.transpose(s1),s2)
    if math.isnan(np.arccos(np.trace(p))):
        return 1
    return np.arccos(np.trace(p))

def giveAccuracy(contoursPre,Y,PreShapeDistKNN,seed=100):    
    b,c=a.kendallPreshape(contoursPre)
    X=[c[i].reshape(-1) for i in range(c.shape[0])]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.9,test_size=0.1,random_state=seed)
    print(X_train[0].shape)
    neigh = KNeighborsClassifier(n_neighbors=3,algorithm='auto',metric=PreShapeDistKNN)
    neigh.fit(X_train, y_train)
    prediction=neigh.predict(X_test)
    total=0 ;       
    for i in range(len(prediction)):
        if prediction[i]==y_test[i]:
            total+=1
    accuracy=total/len(prediction)
    return accuracy

def giveAccuracy2(contoursPre,Y,PreShapeDistKNN,seed=100):    
    b,c=a.kendallPreshape(contoursPre)
    X=[c[i].reshape(-1) for i in range(c.shape[0])]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.9,test_size=0.1,random_state=seed)
    acc=[]
    for nei in range(1,6):
        neigh = KNeighborsClassifier(n_neighbors=nei,algorithm='auto',metric=PreShapeDistKNN)
        neigh.fit(X_train, y_train)
        prediction=neigh.predict(X_test)
        total=0 ;       
        for i in range(len(prediction)):
            if prediction[i]==y_test[i]:
                total+=1
        accuracy=total/len(prediction)
        print('For k=',nei,"accuracy is",accuracy)
        print('wait again')
        acc.append(accuracy)
    return acc



a12=0;  a23=0
for i in range(1):
    print("wait")
    seed=random.randint(0,1000)
    a1=giveAccuracy2(contoursPre2,Y,PreShapeDistKNN,seed)
    print("Accuracy for initial mpoint ",a1)
    a2=giveAccuracy2(contoursPre,Y,PreShapeDistKNN,seed)
    print("Accuracy with no correspandance (Starting point unknown)",a2)
    # a3=giveAccuracy(contoursPre1,Y,PreShapeDistKNN3,seed)
    # print("Accuracy with optimal starting point on kendall shape(this might take few minutes to compute)",a2)
    # a3=giveAccuracy(contoursPre2,Y,PreShapeDistKNN,seed)
    # print("9number ",a3)
    # if  a1<a2:
    #     a12+=1
    # if  a2<a3:
    #     a23+=1
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
k = [1,2,3,4,5]
ax.bar(k,a1)
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
k = [1,2,3,4,5]
ax.bar(k,a2)
plt.show()

        

# def show(s1,s2):
#     plt.plot(s1[:,0],s1[:,1],s1[0:1,0],s1[0:1,1],'*',s1[1:2,0],s1[1:2,1],'s')
#     plt.figure()
#     plt.plot(s2[:,0],s2[:,1],s2[0:1,0],s2[0:1,1],'*',s2[1:2,0],s2[1:2,1],'s')
#     minn=np.linalg.norm(s1[0,:]-s2[0,:])
#     j=0
#     for i in range(1,s1.shape[0]):
#         q=np.linalg.norm(s1[0,:]-s2[i,:])
#         if minn>q:
#             minn=q
#             j=i
#     s3=np.roll(s2,-j,axis=0)
#     plt.plot(s3[:,0],s3[:,1],s3[0:1,0],s3[0:1,1],'*',s3[1:2,0],s3[1:2,1],'s')

    
#     s2=contoursPre[i]
#     s4=shiftpoints(s1,s2,number=99)
#     s3=shiftpoints(s1,s2,number=1)
#     plt.figure()
#     plt.plot(s4[:,0],s4[:,1],s4[0:1,0],s4[0:1,1],'*',s2[0:1,0],s2[0:1,1],'s',s3[0:1,0],s3[0:1,1],'v')

        