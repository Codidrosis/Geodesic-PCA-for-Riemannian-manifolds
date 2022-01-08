# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:06:11 2021

@author: lahza

"""

import multiprocessing
import time
import numpy as np
import scipy.io
import scipy
from math import cos
from math import sin
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt 
import random
from math import *




def worker(queue,tryy,i,pp,x,v,t):
    ll=[]
    for ppp in pp:
        p=np.matmul(tryy.optimalTransformToGeodesicOptimal(pp,x,v,t),pp)
        ret={};    ret['content'] = p;    ret['index']= i
        ll.append(ret)
    queue.put(ll)
def getOptimallyPositionned(tryy,points,x,v):
    """returns an array containing qi
    """
    queue = multiprocessing.Queue()
    #queue.put()
    t=0
    l2=np.zeros(points.shape)
    
    for i in range(10):
        i=iii+ii
        print("starting",i)
        y = multiprocessing.Process(target=worker, args=(queue,tryy,i,points[i],x,v,t,)) 
        y.start()
        l.append(y)
    for x in l:
        x.join()
        x.terminate()
        ret=queue.get()
        print("finished",ret['index'])
        l2[ret['index']]=ret['content']
    return np.array(l2)
    
class PreShape():
    def __init__(self, k_landmarks, m_ambient):
        self.k_landmarks = k_landmarks
        self.m_ambient = m_ambient

    def center(self,point):#checked
        """Center landmarks around 0.
        ----------
        points in Matrices space. : array-like, shape=[..., m_ambient, k_landmarks]
        Returns
        -------
        Points centered : array-like, shape=[..., m_ambient, k_landmarks]
        """
        mean = np.mean(point, axis=-1)
        return point - mean[..., :, None]
    def rotate(self,points,angle=1j):
        points=np.copy(points)
        if points.ndim==2:   initdim=2;            points=np.expand_dims(points, axis=0)
        else:      initdim=3
        
        if points.shape[-1]!=self.k_landmarks and points.shape[-1]!=self.k_landmarks-1:
            init=-1;    points=np.transpose(points,axes=(0,2,1))
        else:
            init=1
        rot=np.zeros((points.shape[0],1,points.shape[2]), dtype=complex)
        for i in range(len(points)):
            for k in range(points.shape[2]):
                rot[i,0,k]=(points[i,0,k]+points[i,1,k]*1j)*np.exp(angle*1j)
                points[i,0,k]=np.real(rot[i,0,k]) 
                points[i,1,k]=np.imag(rot[i,0,k])
        return points
    def is_centered(self,point):#checked
        """Check that landmarks are centered around 0.
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.
        Return
        is_centered : array-like, shape=[...,] ; allCenterd, bool
            Boolean evaluating if point is centered.
        """
        atol=np.abs(np.min(point))*1e-7    #Tolerance at which to evaluate mean == 0.
        mean = np.mean(point, axis=-1)
        is_centered=np.all(np.isclose(mean, 0., atol=atol), axis=-1)
        allCenterd=np.all(is_centered)
        return is_centered,allCenterd
    def PreshapeElement(self, point):   
        """Project a point on the pre-shape space.
        Parameters
        ----------
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.
        Returns
        -------
        projected_point : array-like, shape=[..., m_ambient, k_landmarks]
            Point projected on the pre-shape space.
        """
        centered_point = self.center(point)        
        frob_norm=np.array([np.linalg.norm(point[i,...],ord='fro') for i in range(point.shape[0])])
        projected_point = np.einsum(
            '...,...ij->...ij', 1. / frob_norm, centered_point)
        return projected_point
    def is_normalized(self,point):
        """Check that landmarks are centered around 0.
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.
        Return
        is_centered : array-like, shape=[...,] ; allCenterd, bool
            Boolean evaluating if point is centered.
        """
        atol=np.abs(np.min(point))*1e-7    #Tolerance at which to evaluate mean == 0.
        norm = np.linalg.norm(point,axis=(1,2))
        is_normal=np.isclose(norm, 1., atol=atol)
        allNormal=np.all(is_normal)
        return is_normal,allNormal
    def kendallPreshape(self, point):    
        """Project a point on the pre-shape space.
        Parameters
        ----------
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.
        Returns
        -------
        projected_point : array-like, shape=[..., m_ambient, k_landmarks]
            Point projected on the pre-shape space.
        """
        Q=scipy.linalg.helmert(self.k_landmarks, full=True).T
        if point.ndim==2:   initdim=2;            point=np.expand_dims(point, axis=0)
        else:      initdim=3
        if point.shape[-1]!=self.k_landmarks:
            init=-1;    point=np.transpose(point,axes=(0,2,1))
        else:
            init=1

        plt.plot(point[0,0],point[0,1])
        standBy=point;        point=np.array([np.matmul(point[i,:,:],Q) for i in range(point.shape[0])])

        projected_point=np.array([(point[i,:,:].T-point[i,:,0]).T for i in range(point.shape[0])])
        plt.plot(projected_point[0,0],projected_point[0,1])
        kendallmatrixRep=projected_point[:,:,1:]
        
        if init==-1:
            projected_point,kendallmatrixRep=np.transpose(projected_point,axes=(0,2,1)),np.transpose(kendallmatrixRep,axes=(0,2,1))
        if initdim==2:
            projected_point,kendallmatrixRep=projected_point[0,:,:],kendallmatrixRep[0,:,:] 
        
        return projected_point,kendallmatrixRep
    def reconstruct(self,points,add=True):
        if points.ndim==2:   initdim=2;            points=np.expand_dims(points, axis=0)
        else:      initdim=3
        if points.shape[-1]!=self.k_landmarks:
            point=np.zeros(shape=(points.shape[0],self.m_ambient,self.k_landmarks))
            for i in range(len(points)):
                point[i,:,:]=np.concatenate((np.zeros(shape=(2,1)),points[i] ), axis=1)
            points=point
        Q=scipy.linalg.helmert(self.k_landmarks, full=True).T
        INV=np.linalg.inv(Q)
        inversed=np.array([np.matmul(points[i,:,:],INV) for i in range(points.shape[0])])
        return inversed[0] if initdim==2 else inversed     
    def PreShapeDist(self,s1,s2):
        if s1.ndim==3:
            s1=s1[0]
        if s2.ndim==3:
            s2=s2[0] 
        if self.k_landmarks==s1.shape[1] and self.m_ambient==s1.shape[0]:
            s1=s1.T;    s2=s2.T
        p=np.trace(np.matmul(np.transpose(s1),s2))
        return 0 if p>1 or p<-1 else np.arccos(p)   
    
    def dist_pairwiseNoalign(self, point, **kwargs):
        distmatrix=np.zeros((np.shape(point)[0],np.shape(point)[0]))
        for i in range(np.shape(point)[0]):
            for j in range(i+1,np.shape(point)[0]):
                distmatrix[i,j]=self.PreShapeDist(point[j],point[i])
        distmatrix=distmatrix+distmatrix.T
        return distmatrix,np.argsort(distmatrix, axis=0)
    def normalize(self,v):
        return v / np.linalg.norm(v)
class KendallSpace(PreShape):
    def __init__(self, k_landmarks, m_ambient):
        super().__init__(k_landmarks, m_ambient)
    def KendallDist(self,s1,s2):
        s1=self.align(s1,s2)[0,:,:]
        return self.PreShapeDist(s1,s2)   
    def alignF(self, point, base_point, **kwargs):
        if point.ndim==2:
            point=np.expand_dims(point, axis=0)
        if point.ndim==3 and point.shape[0]>=1:
            mat = np.matmul(point[...,:,:], base_point.T)
            left, singular_values, right = np.linalg.svd(mat)
            result = reduce(np.matmul, [right, np.transpose(left,axes=(0,2,1)),point[...,:,:]])
            return result
        else:
            return -1
    def align(self, point, base_point):
        if point.ndim==2:
            point=np.expand_dims(point, axis=0)
        if point.ndim==3 and point.shape[0]>=1:
            mat = np.matmul(point[...,:,:], base_point.T)
            left, singular_values, right = np.linalg.svd(mat)
            inter=np.eye(self.m_ambient);
            inter[self.m_ambient-1,self.m_ambient-1]=-1
            for i in range(len(left)):
                if np.linalg.det(left[i])<0:
                    left[i]=np.matmul(left[i],inter)
                if np.linalg.det(right[i])<0:
                    right[i]=np.matmul(right[i],inter)
            result = reduce(np.matmul, [right, np.transpose(left,axes=(0,2,1)),point[...,:,:]])
            return result
        else:
            return -1
    def dist_pairwiseRealign(self, points, **kwargs):
        distmatrix=np.zeros((np.shape(points)[0],np.shape(points)[0]))
        h=[]
        for i in range(points.shape[0]):
            base_point=points[i,:,:]
            k=[]
            for j in range(0,points.shape[0]):
                point=points[j:j+1,:,:]
                Aligned=self.align(point,base_point)
                p=np.matmul(np.transpose(Aligned[0]),base_point)
                k.append(np.arccos(np.trace(p)))
                distmatrix[i,j]=np.arccos(np.trace(p))
            h.append(np.argsort(np.array(k)))
            #print(np.argsort(np.array(k)),k[np.argsort(np.array(k))[0]],k[i],k[i+1])
        #distmatrix=distmatrix+distmatrix.T
        return distmatrix,np.argsort(distmatrix, axis=1),h
    def innerproduct(self,s1,s2):
        """inner product of two matrixes.
        Parameters
        ----------
        p1,p2, : array-like, shape=[k_landmarks, m_ambient] Point in preshape space.
        Returns
        -------
        inner product : float
        """
        p=np.array(np.matmul(s1,np.transpose(s2)))

        if p.ndim==0:
            r=float(p)
        else:
            r=np.trace(p)  
        if r>1 or r<-1:
            if np.isclose(r,1.,0.001):
                r=1.0
            if np.isclose(r,-1.,0.001):
                r=-1.0
        return r    
    def eucInnerProduct(self,s1,s2):
        if s1.shape[-1]!=self.k_landmarks and s1.shape[-1]!=self.k_landmarks-1:
            init=-1;    s1=s1.T;s2=s2.T
        else:
            init=1
        print(s1.shape)
        return np.sum([s1[0,i]*s2[0,i]+s1[1,i]*s2[1,i] for i in range(s1.shape[1])])
    def ProjectToGeodesic(self,p,x,v):
        """projects p to the geodesic at x with velocity v.
        Parameters
        ----------
        p,x, : array-like, shape=[k_landmarks, m_ambient] Point in preshape space.
            v:Point in Matrices space.
        Returns
        -------
        projection, point on the geodesic : array-like
        """
        up=self.innerproduct(x,p)*x+self.innerproduct(v,p)*v
        dn=np.sqrt(self.innerproduct(x,p)**2+self.innerproduct(v,p)**2)
        return up/dn
    def distanceToGeodesic(self,p,x,v):
        """Distance from p to the geodesic at x with velocity v.
        Parameters
        ----------
        p,x, : array-like, shape=[k_landmarks, m_ambient] Point in preshape space.
            v:Point in Matrices space.
        Returns
        -------
        distance to geodesic : float
        """
        dist=np.sqrt(self.innerproduct(p,x)**2+self.innerproduct(p,v)**2)

        return 0 if dist>1 else np.arccos(dist)
    def optimalTransformToGeodesicBasic(self,p,x,v,t=0):
        """Distance from p to the geodesic at x with velocity v.
        Parameters
        ----------
        v,x, : array-like, shape=[m_ambient, k_landmarks] Point in preshape and tangent space.
            v:Point in Matrices space.
        Returns
        -------
        distance to geodesic : float
        
        depends on initial t, can give a local non global maxima ==> another method is needed
        """
        for i in range(100):
            mat = np.matmul(p, (x*cos(t)+v*sin(t)).T)
            left, singular_values, right = np.linalg.svd(mat)
            g=np.matmul(left,right.T)
            up=np.trace(reduce(np.matmul, [g, p,v.T]))
            dn=np.trace(reduce(np.matmul, [g, p,x.T]))
            t=np.arctan(up/dn)
        return g
    """Numerical experiments show that the diagonal-optimization algorithm converges to the global maximum in the majority of cases. Further research necessary to develop a faster method"""
    def generateDElemntaryP(self,m):
        l=[np.eye(m)]
        for h in range(m//2):
            if h==0:
                last=np.copy(l);
            else:
                last=np.copy(list(np.unique(np.array(lastly),axis=0)));
            lastly=[]
            for kk in last:
                for i in range(m):
                    for j in range(i+1,m):
                        if kk[i,i]!=-1 and kk[j,j]!=-1 and i!=j:
                            k=np.copy(kk); k[i,i]=-1;k[j,j]=-1
                            lastly.append(k)
                l.extend(lastly)
        l=np.unique(np.array(l),axis=0)
        return l  
    def generateDElemntary2Restricted(self,m):
        """Use generateDElemntaryP instead"""
        l=[np.eye(m)]
        for i in range(m):
            for j in range(i+1,m):
                if i!=j:
                    k=np.copy(l[0]); k[i,i]=-1;k[j,j]=-1
                    l.append(k)
        return l 
    def optimalTransformToGeodesicOptimal(self,p,x,v,t=0):
        """Distance from p to the geodesic at x with velocity v.
        Parameters
        ----------
        p,x, : array-like, shape=[m_ambient, k_landmarks] Point in preshape space.
            v:Point in Matrices space.
        Returns
        -------
        optimal transform : array-like, shape=[m_ambient, k_landmarks] Point
        
        depends on initial t, can give a local non global maxima ==> another method is needed
        """
        m=self.m_ambient
        Em=self.generateDElemntaryP(m)        
        for i in range(1):
            mat = np.matmul(p, (x*cos(t)+v*sin(t)).T)
            left, singular_values, right = np.linalg.svd(mat)
            for i in Em:
                if np.all(np.array(i)==Em[0]):
                    E=np.array(i)
                    minimalDist=self.distanceToGeodesic(reduce(np.matmul, [left, E,right.T,p]),x,v)
                    continue
                newDist=self.distanceToGeodesic(reduce(np.matmul, [left, i,right.T,p]),x,v)
                if minimalDist>newDist:
                    E=np.array(i)
                    minimalDist=newDist
            g=reduce(np.matmul, [left,E,right.T])
            up=np.trace(reduce(np.matmul, [g, p,v.T]))
            dn=np.trace(reduce(np.matmul, [g, p,x.T]))
            t=np.arctan(up/dn)
        return g
    def optimalTransformToGeodesicOptimal2(self,p,x,v,t=0):
        """Distance from p to the geodesic at x with velocity v.
        Parameters
        ----------
        p,x, : array-like, shape=[m_ambient, k_landmarks] Point in preshape space.
            v:Point in Matrices space.
        Returns
        -------
        optimal transform : array-like, shape=[m_ambient, k_landmarks] Point
        
        depends on initial t, can give a local non global maxima ==> another method is needed
        """
        m=self.m_ambient
        Em=self.generateDElemntaryP(m)        
        for i in range(50):
            mat = np.matmul(p, (x*cos(t)+v*sin(t)).T)
            left, singular_values, right = np.linalg.svd(mat)
            for i in Em:
                if np.all(np.array(i)==Em[0]):
                    E=np.array(i)
                    minimalDist=self.distanceToGeodesic(reduce(np.matmul, [left, E,right.T,p]),x,v)
                    continue
                newDist=self.distanceToGeodesic(reduce(np.matmul, [left, i,right.T,p]),x,v)
                if minimalDist<newDist:
                    E=np.array(i)
                    minimalDist=newDist
            g=reduce(np.matmul, [left,E,right.T])
            up=np.trace(reduce(np.matmul, [g, p,v.T]))
            dn=np.trace(reduce(np.matmul, [g, p,x.T]))
            t=np.arctan(up/dn)
        return g
    def gramShmidt(self,E):    
        def normalize(v):
            return v / np.linalg.norm(v)
        U=[normalize(E[0])]
        for i in range(1,len(E)):
            v=E[i]
            for j in U:
                v-=self.innerproduct(j,E[i])*j/(np.linalg.norm(j)**2)
            U.append(normalize(v))


class GeoPCA(KendallSpace): 
    def __init__(self, k_landmarks, m_ambient):
        super().__init__(k_landmarks, m_ambient)
    def getorthogonalBase2(self):
        from scipy.stats import ortho_group
        x = ortho_group.rvs(2)
        from scipy.stats import special_ortho_group
        x = special_ortho_group.rvs(2)
        return x
    def getorthogonalBase3(self):
        pass
    def getorthogonalBase(self,l):
        """returns a list containing wj
        """
        if l==2:
            return [self.getorthogonalBase2()]
        elif l==3:
            return self.getorthogonalBase3()
    def unitHorizentalProjection(self,x,v,w):
        def normalize(v):
            return v / np.linalg.norm(v)
        z=-1*np.sum(np.array([tryy.innerproduct(np.matmul(w[j],x),v)*np.matmul(w[j],x) for j in range(len(w))]),axis=0)
        z=z+v-self.innerproduct(x, v)*x
        return normalize(z)
    
    # def thread_function(self,l,i,pp,x,v,t):
    #     p=np.matmul(self.optimalTransformToGeodesicOptimal(pp,x,v,t),pp)
    #     l[i]=p
    def getOptimallyPositionned(self,points,x,v):
        """returns an array containing qi
        """
        l=[];   t=0
        # l=list(np.zeros(len(points)))
        # for ii in range(0,len(points),10):
        #     for j in range(10):
        #         i=ii+j
        #         x = threading.Thread(target=self.thread_function, args=(l,i,points[i],x,v,t,)) 
        #         x.start()
        #         x.join()
        for i in range(points.shape[0]):
            l.append(np.matmul(self.optimalTransformToGeodesicOptimal(points[i],x,v,t=0),points[i]))
      
        return np.array(l)
        
    
    def getOptimallyPositionned2(self,points,x,v):
        """returns an array containing qi
        """
        l=[]
        for i in range(points.shape[0]):
           l.append(np.matmul(self.optimalTransformToGeodesicBasic(points[i],x,v,t=0),points[i]))
        return np.array(l)    



    def GetIM(self,points,iterr=100):
        x=points[0];
        for i in range(iterr):
            qs=self.align(points,x);Cj=[];Ej=[];
            for i in range(len(qs)):
                cj=self.innerproduct(x,qs[i])
                if cj>=1:
                    Cj.append(1.0)
                    Ej.append(1.0)
                else:
                    Cj.append(cj)
                    Ej.append(np.arccos(cj)/np.sqrt(1-cj**2))

            lamba=np.sum([Ej[i]*self.innerproduct(qs[i],x) for i in range(len(Ej))])
            
            psy=np.sign(lamba)*np.sum([Ej[i]*qs[i] for i in range(len(Ej))],axis=0)
            print(lamba)
            x=self.normalize(psy)
        return x
    def FGPC(self,points):
        """RETUNRS FIRST GPC.
        Parameters
        ----------
        p : array-like, shape=[...,k_landmarks, m_ambient] Point in preshape space.
        Returns
        -------
        x,v : float
        """
        def normalize(v):
            return v / np.linalg.norm(v)

        def getAbbreviationsF(self,points,x,v):
            Cj=[];   Ej=[]
            for ii in range(len(points)):
                i=points[ii]
                cj=np.sqrt(self.innerproduct(x,i)**2+self.innerproduct(v,i)**2)
                if cj>1:
                    cj=1
                Cj.append(cj)
                if cj==1:
                    Ej.append(1)
                    continue
                up=np.arccos(cj);dn=(cj*np.sqrt(1-cj**2))
                Ej.append(up/dn)
            return Cj, Ej
        def getlambdas(self,points,x,v,W,Cj,Ej):
            lbda=[np.sum([Ej[i]*(self.innerproduct(x,points[i])**2) for i in range(len(points))])]
            lbda.append(np.sum([Ej[i]*(self.innerproduct(x,points[i]))*(self.innerproduct(v,points[i])) for i in range(len(points))]))
            lbda.append(np.sum([Ej[i]*(self.innerproduct(v,points[i])**2) for i in range(len(points))]))
            for w in W:
                element=np.sum([Ej[i]*self.innerproduct(v,points[i])*self.innerproduct(np.matmul(w,x),points[i]) for i in range(len(points))])
                lbda.append(element)
            return lbda 
        def getPsy(self,points,x,v,W,Cj,Ej,lambdas):
            a=np.sum([Ej[i]*(self.innerproduct(x,points[i]))*points[i] for i in range(len(points))])
            b=lambdas[2]*v
            c=np.sum([lambdas[3+i]*reduce(np.matmul,[W[i].T,v]) for i in range(len(W))])
            v1=a-b-c
            a=np.sum([Ej[i]*self.innerproduct(v,points[i])*points[i] for i in range(len(points))])
            b=lambdas[2]*v
            c=np.sum([lambdas[3+i]*reduce(np.matmul,[W[i],x]) for i in range(len(W))])
            v2=a-b-c
            return v1,v2
        l=(self.m_ambient)*(self.m_ambient-1)/2   
        x=points[0];        v=points[1]-points[0];
        #W=self.getorthogonalBase(2)
        W=[np.array([[cos(np.pi/2),-sin(np.pi/2)],[sin(np.pi/2),cos(np.pi/2)]])]
        v=self.unitHorizentalProjection(x,v,W)
        ox=np.copy(x);ov=np.copy(v)
        print(np.linalg.norm(x-ox),np.linalg.norm(v-ov))
        for it in range(5):
            qs=self.getOptimallyPositionned(points,x,v)
            Cj, Ej=getAbbreviationsF(self,qs,x,v)
            lambdas=getlambdas(self,qs,x,v,W,Cj,Ej)
            psy1,psy2=getPsy(self,qs,x,v,W,Cj,Ej,lambdas)
            x=normalize(psy1)
            v=self.unitHorizentalProjection(x,psy2,W)
            print(np.linalg.norm(x-ox),np.linalg.norm(v-ov))
            ox=np.copy(x);ov=np.copy(v)
            
        # print('x',x)
        # print('v',v)
        # print('psy1',psy1)
        # print('psy2',psy2)
        # print('Cj',Cj)
        # print('Ej',Ej)
        # print('lambdas',lambdas)
        return x,v
    
    
    def SecondGPC(self,points,x,v,t=0,*kwargs):
        """RETUNRS second GPC.
        Parameters
        ----------
        Returns
        -------
        y,w : array[m,k]
        """
        def normalize(v):
            return v / np.linalg.norm(v)
        def GetG(self,a,b,Ej,qs):
            return np.sum([Ej[i]*self.innerproduct(a,qs[i])*self.innerproduct(b,qs[i]) for i in range(len(points))])
        def GetA(self,a,b,W,lambdas):
            return np.sum([lambdas[i+3]*self.innerproduct(np.matmul(W[i],a),b) for i in range(len(W))])
        def getAbbreviationsF(self,points,x,v):
            Cj=[];   Ej=[]
            for ii in range(len(points)):
                i=points[ii]
                cj=np.sqrt(self.innerproduct(x,i)**2+self.innerproduct(v,i)**2)
                if cj>1:
                    cj=1
                Cj.append(cj)
                if cj==1:
                    Ej.append(1)
                    continue
                up=np.arccos(cj);dn=(cj*np.sqrt(1-cj**2))
                Ej.append(up/dn)
            return Cj, Ej
        def getlambdas(self,qs,x,v,y,w,W,Cj,Ej):
            lbda=[0,0,0]
            for wj in W:
                lbda.append(self.GetG(w,np.matmul(wj,y),Ej,qs))    
            lj=lbda[3:]
            lbda[0]=self.GetG(w,x,Ej,qs)-np.sum([lj[i]*self.innerproduct(np.matmul(W[i],y),x) for i in range(len(W))])
            lbda[1]=self.GetG(w,v,Ej,qs)-np.sum([lj[i]*self.innerproduct(np.matmul(W[i],y),x) for i in range(len(W))])
            lbda[2]=self.GetG(w,w,Ej,qs)-np.sum([lj[i]*self.innerproduct(np.matmul(W[i],y),x) for i in range(len(W))])
            return lbda 
        def getPsy(self,qs,x,v,w,W,Cj,Ej,tau=0):
            a=np.sum([Ej[i]*(self.innerproduct(w,points[i]))*points[i] for i in range(len(points))])
            b=GetG(self,w,x,Ej,qs)*x
            d=np.sum([GetG(self,np.matmul(W[i],x),w,Ej,qs)*reduce(np.matmul,[W[i],x]) for i in range(len(W))])
            c=GetG(self,w,v,Ej,qs)*v
            v1=a-b-c-d
            # a=GetA(self,v,w,W,lambdas)*cos(tau)
            # b=GetA(self,x,w,W,lambdas)*sin(tau)
            # v2=a-b
            a=GetG(self,w,np.matmul(W[0],x),Ej,qs)*self.innerproduct(np.matmul(W[0],v),w)-GetG(self,x,v,Ej,qs)
            b=GetG(self,v,v,Ej,qs)-GetG(self,x,x,Ej,qs)
            v2=a/b
            
            return v1,v2
        xi=x;vi=v;
        l=(self.m_ambient)*(self.m_ambient-1)/2   
        x=x;        v=v;    w=points[1]-points[0];      
        W=[np.array([[cos(np.pi/2),-sin(np.pi/2)],[sin(np.pi/2),cos(np.pi/2)]])]
        z=w-self.innerproduct(x,w)*x-self.innerproduct(v,w)*v-np.sum([self.innerproduct(np.matmul(W[i],x),w)*np.matmul(W[i],x) for i in range(len(W)) ])
        w0=normalize(z)
        for it in range(15):
            print(it)
            qs=self.getOptimallyPositionned(points,x,w0)
            Cj, Ej=getAbbreviationsF(self,qs,x,v)
            psy1,psy2=getPsy(self,qs,x,v,w0,W,Cj,Ej,0)
            x=x*cos(psy2)+v*sin(psy2)
            v=v*cos(psy2)-x*sin(psy2)
            w0=normalize(psy1)
        pass
                    
    def HigherOrdeGPC(self,points,xtild,v,w,dimension,limit,*kwargs):
        """returns the rest of principal components.
        ----------
        xtild : Principal mean
        v : Unit Tangent vector of FGPC
        w : Unit Tangent vector of second GPC
        points in Matrices space. : array-like, shape=[..., m_ambient, k_landmarks]
        Returns
        -------
        Points centered : array-like, shape=[..., m_ambient, k_landmarks]
        """
        
        pass
        pass

    def VarianceToGeo(self,points,x,v):
        pass
    def exponentialMap(self,x,v):
        inn=self.innerproduct(x,v)
        return cos(inn)*x+sin(inn)*v/inn
    def LogMap(self,x,y):
        pass
        return (tet/sin(tet))*(self.align(x, y)[0]-cos(tet)*x)
    def paralelTransportApproxiamation(self,x,y):
        pass
class Tests(GeoPCA): 
    def __init__(self, k_landmarks, m_ambient):
        super().__init__(k_landmarks, m_ambient)
    def testPlotIM(self,points):
        #input: kendall preshape elements
        #out: plot the IM
        IM=self.GetIM(points)
        showIM=self.reconstruct(IM,add=True)
        plt.plot(showIM[0],showIM[1])
    def proveRec(self,a,recons):
    
        plt.plot(recons[0,0],recons[0,1],a[0,0],a[0,1])
        return np.sum([np.linalg.norm(recons[0,:,i]-a[0,:,i]) for i in range(recons.shape[0])])
    def orthogonality(self,x,v,t):
        iszero=self.innerproduct(x, v)
        def Ga(x,v,t):
            return x*cos(t)+v*sin(t)
        def dGa(x,v,t):
            return v*cos(t)-x*sin(t)
        i=np.array([[cos(np.pi/2),-sin(np.pi/2)],[sin(np.pi/2),cos(np.pi/2)]])
        a=self.innerproduct(dGa(x,v,t), np.matmul(i,Ga(x,v,t)))
        b=self.innerproduct(np.matmul(i,x), v)
        t=0
        c=self.innerproduct(dGa(x,v,t), np.matmul(i,Ga(x,v,t)))
        return a,b,c
    def convert(self,p):
        p2=np.zeros((1,p.shape[1]),dtype="complex")
        p2[0] = p[0]+1j*p[1]
        return p2
    def complexinnerproduct(self,p,s):
        return np.trace(np.matmul(p.T,s))
    def testalign2dim(self,p,x):
        p1=self.convert(p);     x1=self.convert(x)
        a=self.complexinnerproduct(x1, p1)**2+self.complexinnerproduct(x1, 1j*p1)**2
        if a!=0:
            e=self.complexinnerproduct(x1, p1)+1j*self.complexinnerproduct(x1, 1j*p1)
            e=e/np.sqrt(a)
        t=np.angle(e)
        r2=np.array([[cos(t),-sin(t)],[sin(t),cos(t)]])
        point=p;base_point=x
        if point.ndim==2:
            point=np.expand_dims(point, axis=0)
        if point.ndim==3 and point.shape[0]>=1:
            mat = np.matmul(point[...,:,:], base_point.T)
            left, singular_values, right = np.linalg.svd(mat)
            inter=np.eye(self.m_ambient);
            inter[self.m_ambient-1,self.m_ambient-1]=-1
            for i in range(len(left)):
                if np.linalg.det(left[i])<0:
                    left[i]=np.matmul(left[i],inter)
                if np.linalg.det(right[i])<0:
                    right[i]=np.matmul(right[i],inter)
            result = reduce(np.matmul, [right, np.transpose(left,axes=(0,2,1))])
        return result,r2
        
    def geoPath(self,p1,p2,t):
        teta=self.KendallDist(p1,p2)
        al=self.align(p1, p2)[0]
        a=1/sin(teta)
        b=sin((1-t)*teta)*p2
        c=sin(t*teta)*al
        return a*(b+c)
    def tgeopath(self,points):
        tt=np.linspace(0,1,10)
        p1=points[random.randint(0,1400)];        p2=points[random.randint(0,1400)]
        #p1=points[0]   ;        p2=points[30]
        for t in tt:
            o=self.geoPath(p1,p2,t)
            out=self.reconstruct(o,add=True)
            plt.figure()
            #plt.plot(o[0],o[1])
            plt.plot(out[0],out[1])
        
        
# tests=Tests(100,2)
# tests.tgeopath(pres)
if __name__ == "__main__":
    k_landmarks=100;    m_ambient=2;    
    tests=Tests(100,2);     tryy=GeoPCA(100,2);
    
    mat = scipy.io.loadmat('contours.mat');    contours=mat.get('b')[0]
    contours=np.array([contours[i].T for i in range(contours.shape[0])])
    
    a=tryy.PreshapeElement(contours)
    b,pres=tryy.kendallPreshape(a)
#     # recons=tryy.reconstruct(pres,add=True)
    
    def checkOptimal(tryy,points):
        x=points[0];        v=points[1]-points[0];
        W=tryy.getorthogonalBase(2)
        v=tryy.unitHorizentalProjection(x,v,W)
        d1=[]
        for i in range(len(points)):
            b=tryy.distanceToGeodesic(points[i],x,v); d1.append(b)
        d2=[]
        qs=tryy.getOptimallyPositionned(points,x,v)
        qs2=tryy.getOptimallyPositionned(points,x,v)
        for i in range(len(points)):
            b=tryy.distanceToGeodesic(qs[i],x,v); d2.append(b)
        for i in range(len(points)):
            b=tryy.distanceToGeodesic(qs[i],x,v); d2.append(b)
        count=0
        
        for i in range(len(qs)):
            if d2[i]>d1[i]:
                count+=1
        print(count,count/len(points))
        return d1,d2,qs,qs2
    #d1,d2,qs,qs2=checkOptimal(tryy,pres) 
    def showConvergence(tryy,p,x,v,t=0):
        tt=np.linspace(0,3,30)
        cccc=[]
        for t in tt:
            m=2
            Em=tryy.generateDElemntaryP(m)     
            T=[t];G=[]
            cv=[tryy.distanceToGeodesic(p,x,v)]
            inter=np.eye(m);
            inter[m-1,m-1]=-1
            total=0
            for i in range(50):
                mat = np.matmul(p, (x*cos(t)+v*sin(t)).T)
                left, singular_values, right = np.linalg.svd(mat)
                if np.linalg.det(left)<0:
                    left=np.matmul(left,inter)
                    print("left");total+=1
                    pass
                if np.linalg.det(right)<0:
                    right=np.matmul(right,inter)
                    print("right");total+=1
                    pass
                for i in range(len(Em)):
                    if i==0:
                        E=Em[0]
                        minimalDist=tryy.distanceToGeodesic(reduce(np.matmul, [left, E,right.T,p]),x,v)
                        continue
                    newDist=tryy.distanceToGeodesic(reduce(np.matmul, [left, Em[i],right.T,p]),x,v)
                    if minimalDist>newDist:
                        E=Em[i]
                        minimalDist=newDist
                g=reduce(np.matmul, [left,E,right.T]);      G.append(g)
                cv.append(tryy.distanceToGeodesic(np.matmul(g,p),x,v))
                
                up=np.trace(reduce(np.matmul, [g, p,v.T]))
                dn=np.trace(reduce(np.matmul, [g, p,x.T]))
                t=np.arctan(up/dn);            T.append(t)
            print(total/2)
            cccc.append(np.array(cv))
        return cccc
        return T,G,cv
    def showConvergence2(tryy,p,x,v,t=0):
        m=2
        Em=tryy.generateDElemntaryP(m)     
        T=[t];G=[]
        cv=[tryy.distanceToGeodesic(p,x,v)]
        inter=np.eye(m);
        inter[m-1,m-1]=-1        
        for i in range(200):
            mat = np.matmul(p, (x*cos(t)+v*sin(t)).T)
            left, singular_values, right = np.linalg.svd(mat)
            if np.linalg.det(left)<0 or np.linalg.det(right)<0:
                print("differetn")
            if np.linalg.det(left)<0:
                left=np.matmul(left,inter)
                print("corrected left", np.linalg.det(left))
                pass
            if np.linalg.det(right)<0:
                right=np.matmul(right,inter)
                print("corrected right", np.linalg.det(right))
                pass
            g=np.matmul(left,right.T)
            minimalDist=tryy.distanceToGeodesic(reduce(np.matmul, [g,p]),x,v)
            cv.append(minimalDist);G.append(g);T.append(t)
            up=np.trace(reduce(np.matmul, [g, p,v.T]))
            dn=np.trace(reduce(np.matmul, [g, p,x.T]))
            t=np.arctan(up/dn)
        return T,G,cv    
    
#         T=np.linspace(0,3.14,100)
#         minn=tryy.distanceToGeodesic(p,x,v)
#         opt=p
#         cv=[minn]
#         optt=0
#         m=2
#         inter=np.eye(m);
#         inter[m-1,m-1]=-1   
#         for t in T:
#             #◘al=tryy.align( p, (x*cos(t)+v*sin(t)))[0]
#             mat = np.matmul(p, (x*cos(t)+v*sin(t)).T)
#             left, singular_values, right = np.linalg.svd(mat)
#             if np.linalg.det(left)<0 or np.linalg.det(right)<0:
#                 print("differetn")
#             if np.linalg.det(left)<0:
#                 left=np.matmul(left,inter)
#                 print("corrected left", np.linalg.det(left))
#                 pass
#             if np.linalg.det(right)<0:
#                 right=np.matmul(right,inter)
#                 print("corrected right", np.linalg.det(right))
#                 pass
#             al = reduce(np.matmul, [right, left.T,p])
#             minimalDist=tryy.distanceToGeodesic(al,x,v)
#             if minn>minimalDist:
#                 opt=al
#                 minn=minimalDist
#                 optt=t
#             cv.append(minimalDist);
#         return optt,cv,opt
    
    #tests.tgeopath(pres)
    aa=random.randint(0,1400);bb=random.randint(0,1400);cc=random.randint(0,1400)
    p=pres[aa];
    x=pres[bb];        v=pres[cc]-pres[bb];
    W=[np.array([[cos(np.pi/2),-sin(np.pi/2)],[sin(np.pi/2),cos(np.pi/2)]])] #Only in case m=2
    #W=tryy.getorthogonalBase(2)
    v=tryy.unitHorizentalProjection(x,v,W)
    isTangant=tryy.innerproduct(x,v)
    xx,vv=tryy.FGPC(pres) #return FGPC 
    yy,ww=tryy.SecondGPC(pres,xx,vv) #return secondGPC
    tests.tgeopath(pres) #return a random trajectory
    tests.testPlotIM(pres[0:20]) #return the intrinsic mean on kendall's shape space

    

#     points=pres
#     l=1   
#     #W=self.getorthogonalBase(self.m_ambient)
#     W=[np.array([[cos(np.pi/2),-sin(np.pi/2)],[sin(np.pi/2),cos(np.pi/2)]])]
#     v=tryy.unitHorizentalProjection(x,v,W)
#     print(tryy.innerproduct(x,v))
#     for it in range(100):
#         qs=tryy.getOptimallyPositionned(points,x,v)
#         Cj, Ej=tryy.getAbbreviationsF(qs,x,v)
#         lambdas=tryy.getlambdas(qs,x,v,W,Cj,Ej)
#         psy1,psy2=tryy.getPsy(qs,x,v,W,Cj,Ej,lambdas)
#         x=tryy.normalize(psy1)
#         #W=tryy.getorthogonalBase(2)
#         v=tryy.unitHorizentalProjection(x,psy2,W)
#         print(tryy.innerproduct(v,x));
#     # recons=tryy.reconstruct(x,add=True)
#     # recons2=tryy.reconstruct(v,add=True)
#     # plt.plot(recons[0],recons[1],recons2[0],recons2[1])
#     #x,v=tryy.FGPC(pres)
#     # projection=[];prec=[]
#     # for p in pres: 
#     #     proj=tryy.ProjectToGeodesic(p,x,v)
#     #     projection.append(proj)
#     #     prec.append(tryy.reconstruct(proj,add=True))
#     #     plt.plot(proj[0],proj[1])





#     ####################################""""TESTS
    
#     # shouldBeCloseToZero=tests.proveRec(a,recons)
#     # tests.testPlotIM(pres[0:20])
#     # tests.testPlotIM(pres[40:60])
    
#     ###########################################
#     rotated=tryy.rotate(pres,1j);i=5
#     print(tryy.eucInnerProduct(pres[i],rotated[i]));
    
#     # def thread_function(queue,tryy,l,i,pp,x,v,t):
#     #     p=np.matmul(tryy.optimalTransformToGeodesicOptimal(pp,x,v,t),pp)
#     #     ret={};    ret['content'] = p;    ret['index']= i
#     #     queue.put(ret)
        
#     # def getOptimallyPositionned(tryy,points,x,v):
#     #     """returns an array containing qi
#     #     """
#     #     queue = multiprocessing.Queue()
#     #     l=[];   t=0
#     #     l2=np.zeros(len(points))
#     #     for i in range(0,len(points)):
#     #         y = multiprocessing.Process(target=thread_function, args=(queue,tryy,l,i,points[i],x,v,t,)) 
#     #         y.start()
#     #         l.append(y)
#     #     for x in l:
#     #         x.join()
#     #         ret=queue.get()
#     #         l2[ret['index']]=ret['content']
#     #     return np.array(l2)
#     #getOptimallyPositionned(tryy,points,x,v)

#     # def thread_function(queue,tryy,l,i,pp,x,v,t):
#     #     p=np.matmul(tryy.optimalTransformToGeodesicOptimal(pp,x,v,t),pp)
#     #     ret={};    ret['content'] = p;    ret['index']= i
#     #     queue.put(ret)
        

#     # from multiprocessing import Pool

#     # ppp=[]
#     # for k in range(len(points)):
#     #     t=0
#     #     ppp.append([tryy,points[k],x,v,t])
#     # start = time.time()
#     # with Pool(4) as p:
#     #     res=p.map(worker2, ppp)
#     # end = time.time()
#     # with Pool(5) as p:
#     #     print(p.map(worker2, [1, 2, 3]))
    
#     # ret = {'foo': False}
#     # queue = multiprocessing.Queue()
#     # queue.put(ret)
#     # p = multiprocessing.Process(target=worker, args=(queue,))
#     # p.start()
#     # p.join()
#     # print(queue.get())  # Prints {"foo": True}
    
#     #a,b,c=tests.orthogonality(x,v,2)
# # count=0
# # for i in range(len(pres)):
# #     for j in range(len(pres)):
# #         if i==j:
# #             continue
# #         if tryy.innerproduct(pres[i],pres[j])>1 or tryy.innerproduct(pres[i],pres[j])<-1:
# #             print(tryy.innerproduct(pres[i],pres[j]), i,j)
# #             count+=1
#     # start = time.time()
#     # qs=getOptimallyPositionned(tryy,points,x,v)
#     # end = time.time()
#     # print(end - start) #100s /310threads with prints,/ 125 thread no print/// with pool 83//109 16//113.49487257003784




# #multiprocessing.cpu_count()
