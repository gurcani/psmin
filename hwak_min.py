import numpy as np
import os
mls_backend="numpy"

os.environ['MLSARRAY_BACKEND']=mls_backend
if mls_backend=="cupy":
    import cupy as xp 
else: 
    import numpy as xp

import gsol.mlsarray.mlsarray as mls
import h5py as h5
from time import time
from gsol.gsol import gsol
from gsol.gsol import callbacks
from gsol.h5tools import save_data
# Physics Paramteres
C=1.0
kap=1.0
nu=5e-4
D=5e-4

# Simulation Parameters
flname="out.h5"
wecontinue=True
Npx,Npy=1024,1024
t0,t1=0,500.0
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=12*np.pi,12*np.pi
dkx,dky=2*np.pi/Lx,2*np.pi/Ly

#setting up the grid
sl=mls.slicelist(Nx,Ny)
lkx,lky=mls.init_kspace_grid(sl)
Nk=lkx.size
kx,ky=lkx*dkx,lky*dky
ksqr=kx**2+ky**2
sigk=(ky>0)

# some shortcuts
irft = lambda x : mls.irft2(x,sl)
rft = lambda x : mls.rft2(x,sl)
get = mls.get

#Initial conditions
Ak,wk=1e-4,2.0
zk0=xp.zeros((2,kx.size),dtype=complex)
zk0[0,:]=Ak*xp.exp(-lkx**2/2/wk**2-lky**2/wk**2)*xp.exp(1j*2*xp.pi*xp.random.rand(lkx.size).reshape(lkx.shape));
zk0[1,:]=Ak*xp.exp(-lkx**2/wk**2-lky**2/wk**2)*xp.exp(1j*2*xp.pi*xp.random.rand(lkx.size).reshape(lkx.shape));

#Linear Matrix
Lk=np.zeros(kx.shape+(2,2),dtype=complex)
Lk[:,0,0]=get(-C*sigk/ksqr-nu*sigk*ksqr)
Lk[:,0,1]=get(C*sigk/ksqr)
Lk[:,1,0]=get(C*sigk-1j*kap*ky)
Lk[:,1,1]=get(-C*sigk-D*ksqr*sigk)

#The nonlinear terms
def rhsnl(t,zk):
    dzkdt=xp.zeros_like(zk)
    phik,nk=zk[0,:],zk[1,:]
    dphikdt,dnkdt=dzkdt[0,:],dzkdt[1,:]
    dxphi=irft(1j*kx*phik)
    dyphi=irft(1j*ky*phik)
    om=irft(-ksqr*phik)
    n=irft(nk)
    dphikdt[:]=(-1j*kx*rft(dyphi*om)+1j*ky*rft(dxphi*om))/ksqr
    dnkdt[:]=1j*kx*rft(dyphi*n)-1j*ky*rft(dxphi*n)
    return dzkdt


#Save Stuff
def save_callback(fl,t,zk,flag):
    phink=zk.reshape((2,kx.size))
    phik,nk=phik,nk=phink[0,:],phink[1,:]
    save_data(fl,'last',ext_flag=False,zk=get(zk),t=get(t),dt=r.hlast,tnexts=r.cbs.tnexts)
    if flag=='fields':
        print('saving fields')
        om=irft(-phik*(kx**2+ky**2))
        n=irft(nk)
        save_data(fl,'fields',ext_flag=True,om=get(om),n=get(n),t=get(t))
    if flag=='energies':
        print('saving energies')
        Etot=xp.sum(xp.abs(phik)**2*(kx**2+ky**2))
        Ez=xp.sum(xp.abs(phik)**2*(kx**2+ky**2)*(ky==0))
        Ftot=xp.sum(xp.abs(nk)**2)
        Fz=xp.sum(xp.abs(nk**2)*(ky==0))
        save_data(fl,'energies',ext_flag=True,Etot=get(Etot),Ez=get(Ez),Ftot=get(Ftot),Fz=get(Fz),t=get(t))

# initialize the hdf5 file
if(wecontinue):
    fl=h5.File(flname,'r+',libver='latest')
    fl.swmr_mode = True
    zk0=fl['last/zk'][()]
#    omk,nk=rft(xp.array(fl['fields/om'][-1,])),rft(xp.array(fl['fields/n'][-1,]))
#    phik=-omk/(kx**2+ky**2)
    t0=fl['last/t'][()]
    tnexts=fl['last/tnexts'][()]
    dtstep=fl['last/dt'][()]
#    zk0=xp.hstack((phik,nk))
else:
    if os.path.exists(flname):
        os.remove(flname)
    fl=h5.File(flname,'w',libver='latest')
    fl.swmr_mode = True
    save_data(fl,'data',ext_flag=False,kap=kap,C=C,nu=nu,D=D,Lx=Lx,Ly=Ly)
    dtstep=1.0
    tnexts=None

# define callbacks
ct=time()
fcbs = [(lambda t,y : print('t=',t,', ',time()-ct,' secs elapsed')),
        (lambda t,y : save_callback(fl,t,y,flag='fields')),
         (lambda t,y : save_callback(fl,t,y,flag='energies'))]
dtcbs=[1.0,1.0,10.0]
cbs=callbacks(dtcbs,fcbs,tnexts)
# initiate and run the solver
#r=gsol(rhsnl,t0,zk0,t1,Lk,dtstep,callbacks=cbs,sv="etdrk4cp",tol=1e-8)
#r=gsol(rhsnl,t0,zk0,t1,Lk,dtstep,callbacks=cbs,sv="scipy.DOP853",atol=1e-12,rtol=1e-9)
r=gsol(rhsnl,t0,zk0,t1,Lk,dtstep,callbacks=cbs,sv="scipy_old.vode",atol=1e-8,rtol=1e-7)
r.run()
fl.close()
