"""
Sampling grids for 2D canonical coordinate systems.
Each coordinate system is defined by a pair of coordinates u, v.
Each grid function maps from a grid in (u, v) coordinates to a collection points in Cartesian coordinates.
"""

import numpy as np
import torch
from visdom import Visdom
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F


def identity_grid(output_size, ulim=(-1, 1), vlim=(-1, 1), out=None, device=None):
    """Cartesian coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian x-coordinate limits
        vlim: (float, float), Cartesian y-coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us
    ys = vs
    return torch.stack([xs, ys], 2, out=out)


def polar_grid(output_size, ulim=(0, np.sqrt(2.)), vlim=(-np.pi, np.pi), out=None, device=None):
    """Polar coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), radial coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us * torch.cos(vs)
    ys = us * torch.sin(vs)
    return torch.stack([xs, ys], 2, out=out)


def logpolar_grid(output_size, ulim=(None, np.log(2.)/2.), vlim=(-np.pi, np.pi), out=None, device=None):
    """Log-polar coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), radial coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-np.log(nu), ulim[1])
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    rs = torch.exp(us)
    xs = rs * torch.cos(vs)
    ys = rs * torch.sin(vs)
    return torch.stack([xs, ys], 2, out=out)


def shearx_grid(output_size, ulim=(-1, 1), vlim=(-5, 5), out=None, device=None):
    """Horizontal shear coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian y-coordinate limits
        vlim: (float, float), x/y ratio limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    ys = us
    xs = us * vs
    return torch.stack([xs, ys], 2, out=out)


def sheary_grid(output_size, ulim=(-1, 1), vlim=(-5, 5), out=None, device=None):
    """Vertical shear coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian x-coordinate limits
        vlim: (float, float), y/x ratio limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us
    ys = us * vs
    return torch.stack([xs, ys], 2, out=out)


def scalex_grid(output_size, ulim=(None, 0), vlim=(-1, 1), out=None, device=None):
    """Horizontal scale coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), logarithmic x-coordinate limits
        vlim: (float, float), Cartesian y-coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-np.log(nu/2), ulim[1])
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])

    xs = torch.exp(us)
    ys = vs

    if nv % 2 == 0:
        xs = torch.cat([xs, -xs])
        ys = torch.cat([ys, ys])
    else:
        xs = torch.cat([xs, xs.narrow(0, xs.shape[0]-1, 1), -xs])
        ys = torch.cat([ys, ys.narrow(0, ys.shape[0]-1, 1), ys])
    return torch.stack([xs, ys], 2, out=out)


def scaley_grid(output_size, ulim=(None, 0), vlim=(-1, 1), out=None, device=None):
    """Vertical scale coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), logarithmic y-coordinate limits
        vlim: (float, float), Cartesian x-coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-np.log(nu/2), ulim[1])
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])

    ys = torch.exp(us)
    xs = vs

    if nv % 2 == 0:
        xs = torch.cat([xs, xs])
        ys = torch.cat([ys, -ys])
    else:
        xs = torch.cat([xs, xs.narrow(0, xs.shape[0]-1, 1), xs])
        ys = torch.cat([ys, ys.narrow(0, ys.shape[0]-1, 1), -ys])
    return torch.stack([xs, ys], 2, out=out)


def hyperbolic_grid(output_size, ulim=(-np.sqrt(0.5), np.sqrt(0.5)), vlim=(-np.log(6.), np.log(6.)), out=None, device=None):
    """Hyperbolic coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), hyperbolic angular coordinate limits
        vlim: (float, float), hyperbolic log-radial coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])

    rs = torch.exp(vs)
    xs = us * rs
    ys = us / rs

    if nv % 2 == 0:
        xs = torch.cat([xs, xs])
        ys = torch.cat([ys, -ys])
    else:
        xs = torch.cat([xs, xs.narrow(0, xs.shape[0]-1, 1), xs])
        ys = torch.cat([ys, ys.narrow(0, ys.shape[0]-1, 1), -ys])
    return torch.stack([xs, ys], 2, out=out)


def perspectivex_grid(output_size, ulim=(1, 8), vlim=(-0.99*np.pi/2, 0.99*np.pi/2), out=None, device=None):
    """Horizontal perspective coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), x^{-1} "radial" coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])



    xl = -1 / us.flip([1])
    xr =  1 / us
    yl = -xl * torch.tan(vs)
    yr =  xr * torch.tan(vs)

    if nv % 2 == 0:
        xs = torch.cat([xl, xr])
        ys = torch.cat([yl, yr])
    else:
        xs = torch.cat([xl, xl.narrow(0, xl.shape[0]-1, 1), xr])
        ys = torch.cat([yl, yl.narrow(0, yl.shape[0]-1, 1), yr])

    return torch.stack([xs, ys], 2, out=out)


def perspectivey_grid(output_size, ulim=(1, 8), vlim=(-0.99*np.pi/2, 0.99*np.pi/2), out=None, device=None):
    """Vertical perspective coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), y^{-1} "radial" coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)

    vrange = torch.linspace(vlim[0], vlim[1], nv//2, device=device)
    vs, us = torch.meshgrid([vrange, urange])




    yl = -1 / us.flip([1])
    yr =  1 / us
    xl = -yl * torch.tan(vs)
    xr =  yr * torch.tan(vs)


    if nv % 2 == 0:
        xs = torch.cat([xl, xr])
        ys = torch.cat([yl, yr])
    else:
        xs = torch.cat([xl, xl.narrow(0, xl.shape[0]-1, 1), xr])
        ys = torch.cat([yl, yl.narrow(0, yl.shape[0]-1, 1), yr])
    #print(xs.shape)
    #a = torch.stack([xs, ys], 2, out=out)
    #print('tensor:',a.shape, 2)
    #plt.plot(xs, ys)
    #plt.show()
    return torch.stack([xs, ys], 2, out=out)


def spherical_grid(output_size, ulim=(-np.pi/4, np.pi/4), vlim=(-np.pi/4, np.pi/4), out=None, device=None):
    """Spherical coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), latitudinal coordinate limits
        vlim: (float, float), longitudinal coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    su, cu = torch.sin(us), torch.cos(us)
    sv, cv = torch.sin(vs), torch.cos(vs)
    xs = cu * sv / (np.sqrt(2.) - cu * cv)
    ys = su / (np.sqrt(2.) - cu * cv)
    return torch.stack([xs, ys], 2, out=out)



class IPM(object):
    """
    Inverse perspective mapping to a bird-eye view. Assume pin-hole camera model.
    There are detailed explanation of every step in the comments, and variable names in the code follow these conventions:
    `_c` for camera coordinates
    `_w` for world coordinates
    `uv` for perspective transformed uv 2d coordinates (the input image)
    """

    def __init__(self, camera_info, ipm_info):
        self.camera_info = camera_info
        self.ipm_info = ipm_info

        ## Construct matrices T, R, K
        self.T = np.eye(4)
        self.T[2, 3] = -camera_info.camera_height  # 4x4 translation matrix in 3d space (3d homo coordinate)
        _cy = np.cos(camera_info.yaw * np.pi / 180.)
        _sy = np.sin(camera_info.yaw * np.pi / 180.)
        _cp = np.cos(camera_info.pitch * np.pi / 180.)
        _sp = np.sin(camera_info.pitch * np.pi / 180.)
        _cr = np.cos(camera_info.roll * np.pi/180.)
        _sr = np.sin(camera_info.roll * np.pi/180.)
        tyaw = np.array([[_cy, 0, -_sy],
                         [0, 1, 0],
                         [_sy, 0, _cy]])
        tyaw_inv = np.array([[_cy, 0, _sy],
                             [0, 1, 0],
                             [-_sy, 0, _cy]])
        tpitch = np.array([[1, 0, 0],
                           [0, _cp, -_sp],
                           [0, _sp, _cp]])
        tpitch_inv = np.array([[1, 0, 0],
                               [0, _cp, _sp],
                               [0, -_sp, _cp]])
        troll = np.array([[_cr, -_sr, 0],
                         [_sr, _cr, 0],
                         [0, 0, 1]])
        troll_inv = np.array([[_cr, _sr, 0],
                         [-_sr, _cr, 0],
                         [0, 0, 1]])
        yp = np.dot(tyaw,tpitch)
        yr = np.dot(tyaw,troll)

        self.R = np.dot(troll,tpitch)  # 3x3 Rotation matrix in 3d space
        #print(self.R)
        self.R_inv = np.dot(tpitch_inv, troll_inv, tyaw_inv)
        self.K = np.array([[camera_info.f_x, 0, camera_info.u_x],
                           [0, camera_info.f_y, camera_info.u_y],
                           [0, 0, 1]]).astype(np.float)  # 3x3 intrinsic perspective projection matrix

        #a = torch.ones(3,3)
        #a.unsqueeze(0)

        #print(a.size())

        ## The ground plane z=0 in the world coordinates, transform to a plane `np.dot(self.normal_c, point) = self.const_c` in the camera coordinates.
        # This is used to find (x,y,z)_c according to (u,v). See method `uv2xy` for detail.
        self.normal_c = np.dot(self.R,
                               np.array([0, 0, 1])[:, None])  # normal of ground plane equation in camera coordinates
        #print(self.normal_c.shape)#(normal_c.shape = 3x1)
        #print(self.normal_c)
        self.const_c = np.dot(self.normal_c.T,
                              np.dot(self.R,
                                     np.dot(self.T, np.array([0, 0, 0, 1])[:, None])[
                                     :3]))  # constant of ground plane equation in camera coordinates

        #print(self.const_c) # const_c.shape = 1x1
        ## Get the limit to be converted on the uv map (must below vanishing point)
        # To calculate (u,v) of the vanishing point on the uv map of delta vector v=[0,1,0] in the world coordinates
        # homo coordinates of a vector will be v_4 = [0, 1, 0, 0], mapping this vector to camera coordinate:
        # vc_3 = np.dot(R_4, np.dot(T_4, v_4))[:3] = np.dot(R, v), the 2d homo coordinate of the vanishing point will be at
        # lim_{\lambda -> \infty} np.dot(K, lambda * vc_3) = np.dot(K, vc_3)

        # lane_vec_c = np.dot(self.R, np.array([0,1,0])[:, None]) # lane vector in camera coordinates
        # lane_vec_homo_uv = np.dot(self.K, lane_vec) # lane vector on uv map (2d homo coordinate)
        lane_vec_homo_uv = np.dot(self.K, np.dot(self.R, np.array([0, 1, 0])[:,
                                                         None]))  # lane vector on uv map (2d homo coordinate)
        #print(lane_vec_homo_uv.shape)#3x1
        vp = self.vp = lane_vec_homo_uv[:2] / lane_vec_homo_uv[
            2]  # coordinates of the vanishing point of lanes on uv map
        #print(vp)

        # UGLY: This is an ugly op to ensure the converted area do not goes beyond the vanishing point, as the camera intrinsic/extrinsic parameters are not accurate in my case.
        ipm_top = self.ipm_top = max(ipm_info.top, vp[1] + ipm_info.input_height / 15)
        uv_limits = self.uv_limits = np.array([[ipm_info.left, ipm_top],
                                               [ipm_info.right, ipm_top],
                                               [vp[0], ipm_top],
                                               [vp[0],
                                                ipm_info.bottom]]).T  # the limits of the area on the uv map to be IPM-converted

        ## The x,y limit in the world coordinates is used to calculate xy_grid, and then the corresponding uv_grid
        self.xy_limits = self.uv2xy(uv_limits)
        xmin, xmax = min(self.xy_limits[0]), max(self.xy_limits[0])
        ymin, ymax = min(self.xy_limits[1]), max(self.xy_limits[1])

        #print(xmax, ymax ,xmin ,ymin)
        stepx = (xmax - xmin) / ipm_info.out_width  # x to output pixel ratio
        stepy = (ymax - ymin) / ipm_info.out_height  # y to output pixel ratio
        #print(uv_limits)
        #print(self.xy_limits)
        # xy_grid: what x,y coordinates in world coordinates will be stored in every output image pixel
        self.xy_grid = np.array(
            [[(xmin + stepx * (0.5 + j), ymax - stepy * (0.5 + i)) for j in range(ipm_info.out_width)]
            for i in range(ipm_info.out_height)]).reshape(-1, 2).T

        #print(xmin+stepx*0.5,xmin+stepx*(ipm_info.out_width+0.5),ymax-stepy*0.5,ymax-stepy*960.5)
        #print(self.xy_grid)
        #plt.scatter(self.xy_grid[0],self.xy_grid[1])
        #plt.show()
        self.xmin = min(self.xy_grid[0])
        self.xmax = max(self.xy_grid[0])
        self.ymin = min(self.xy_grid[1])
        self.ymax = max(self.xy_grid[1])
        print(self.xmin, self.xmax,self.ymin, self.ymax)

        '''
        x, y = torch.meshgrid(torch.linspace(-14400,14400, ipm_info.out_width),
                              torch.linspace(2880, 21150, ipm_info.out_height))
        self.grid_torch = torch.stack([x, y],0)
        print(self.grid_torch.shape)
        self.grid_torch_new = self.grid_torch.reshape(2,ipm_info.out_height*ipm_info.out_width)
        print(self.grid_torch_new.shape)




        x_t, y_t = np.meshgrid(np.linspace(-10000, 10000,1280),
                               np.linspace(20000, 1200, 960))
        #ones = torch.ones(40000,1)
        self.grid = np.stack([x_t, y_t])
        print(self.grid.shape)
        #plt.plot(self.grid[1],self.grid[0])
        #plt.show()
        #self.grid=(x_t,y_t)
        print(self.grid.shape)
  
        self.grid_new = self.grid.reshape(2,1280*960)
        #plt.plot(self.grid[1],self.grid[0])
        #plt.plot(x_t,y_t)
        #plt.show()


        # uv_grid: what u,v coordiantes on the uv map will be stored in every output image pixel
        #self.uv_grid = self.xy2uv(self.grid_torch_new).int()#.astype(int)
        self.uv_grid = self.xy2uv_np(self.grid_new).astype(int)
        #self.uv_grid = self.uv_grid * ((self.uv_grid[0] > ipm_info.left) * (self.uv_grid[0] < ipm_info.right) * \
         #                              (self.uv_grid[1] > ipm_info.top) * (self.uv_grid[1] < ipm_info.bottom))
        self.uv_grid = self.uv_grid.reshape(2, ipm_info.out_height, ipm_info.out_width)
        #print(type(self.uv_grid))
        #self.uv_grid = (self.uv_grid[1], self.uv_grid[0])
        #a = self.uv_grid[0]
        #print(type(a))
        #print(a.ravel()[np.flatnonzero(a)])
        #self.u = transforms.ToTensor()(self.uv_grid[0])
        #self.v = transforms.ToTensor()(self.uv_grid[1])
        #self.grid = torch.stack([self.u, self.v]).float()
        self.u = self.uv_grid[0]
        self.v = self.uv_grid[1]
        self.u = (self.u[:]-ipm_info.out_width/2)/ipm_info.out_width
        self.v = (self.v[:]-ipm_info.out_height/2)/ipm_info.out_height
        #self.u = self.standardization(self.uv_grid[0])
        #self.v = self.standardization(self.uv_grid[1])
        #self.us = (self.uv_grid[0])//(self.u.max()-self.u.min())
        #self.vs = (self.uv_grid[1])//(self.v.max()-self.v.min())

        #print(self.u.min())
        #plt.plot(self.u, self.v)
        #plt.show()
        self.pers_grid = torch.stack([self.u, self.v],2)
        #self.pers_grid = np.stack([self.u, self.v],2)
        #self.pers_grid = torch.from_numpy(self.pers_grid)
        #print(self.pers_grid)
        self.pers_grid = self.pers_grid.unsqueeze(0).float()
        #self.pers_grid = F.normalize(self.pers_grid)
        #self.pers_grid = transforms.Normalize((0.5),(0.5))(self.pers_grid)
        #print(self.uv_grid[1].shape)
        #self.perspective_grid = np.meshgrid(self.uv_grid[1], self.uv_grid[0])
        '''

    '''
    def xy2uv(self, xys):  # all points have z=0 (ground plane): w (u,v,1) = KRT (x,y,z)_w
        print(xys.shape) #2x1228800
        ones = -self.camera_info.camera_height * torch.ones(1,xys.shape[1])
        xyzs = torch.cat((xys, ones),0)
        #xyzs = np.vstack((xys, -self.camera_info.camera_height * np.ones(xys.shape[1])))  # (x,y,z) after translation
        print(xyzs.shape)
        #xyzs_tensor = torch.from_numpy(xyzs)
        #print(xyzs_tensor.shape)
        #a = np.dot(self.R, xyzs)
        #print(a.shape)
        #print(self.K.shape)
        print(self.K)
        #self.K = torch.from_numpy(self.K).float()
        print(self.K)
        #self.R = torch.from_numpy(self.R).float()
        #xyzs_c = torch.matmul(self.K, torch.matmul(self.R, xyzs))  # w(u,v,1) (2d homo)

        #print('xyzs_c:', xyzs_c.shape) #3x1228800
        uv = xyzs_c[:2] / xyzs_c[2]
        print('uv:',uv.shape) #2x1228800
        #print((xyzs_c[:2]/xyzs_c[2]).astype(int))
        return xyzs_c[:2] / xyzs_c[2]
    '''
    def xy2uv_np(self, xys):  # all points have z=0 (ground plane): w (u,v,1) = KRT (x,y,z)_w
        print(xys.shape) #2x1228800
        xyzs = np.vstack((xys, -self.camera_info.camera_height * np.ones(xys.shape[1])))  # (x,y,z) after translation
        print(xyzs.shape)
        #xyzs_tensor = torch.from_numpy(xyzs)
        #print(xyzs_tensor.shape)
        #a = np.dot(self.R, xyzs)
        #print(a.shape)
        #print(self.K.shape)
        xyzs_c = np.dot(self.K, np.dot(self.R, xyzs))  # w(u,v,1) (2d homo)
        return xyzs_c[:2] / xyzs_c[2]





    def grid_generator_np(self):
        x, y = np.meshgrid(np.linspace(self.xmin,self.xmax, ipm_info.out_width),
                              np.linspace(self.ymin, self.ymax, ipm_info.out_height))
        self.grid_np = np.stack([x, y],0)
        print(self.grid_np.shape)
        self.grid_np_new = self.grid_np.reshape(2,ipm_info.out_height*ipm_info.out_width)
        self.grid_np_new = np.vstack((self.grid_np_new, -self.camera_info.camera_height * np.ones(self.grid_np_new.shape[1])))
        self.uv_grid = np.dot(self.K, np.dot(self.R, self.grid_np_new))  # w(u,v,1) (2d homo)

        self.uv_grid = self.uv_grid[:2] / self.uv_grid[2]
        plt.plot(self.uv_grid[0],self.uv_grid[1])
        plt.show()
        self.uv_grid = self.uv_grid * ((self.uv_grid[0] > ipm_info.left) * (self.uv_grid[0] < ipm_info.right) * \
                                       (self.uv_grid[1] > ipm_info.top) * (self.uv_grid[1] < ipm_info.bottom))
        self.uv_grid = self.uv_grid.reshape(2, ipm_info.out_height, ipm_info.out_width)
        self.u = self.uv_grid[0]
        self.v = self.uv_grid[1]
        self.u = (self.u[:]-ipm_info.out_width/2)/ipm_info.out_width
        self.v = (self.v[:]-ipm_info.out_height/2)/ipm_info.out_height
        plt.plot(self.u,self.v)
        plt.show()
        self.pers_grid = np.stack([self.u, self.v],2)
        self.pers_grid = torch.from_numpy(self.pers_grid)
        self.pers_grid = self.pers_grid.unsqueeze(0).float()
/home/mingshu/PycharmProjects/Boosted-IPM-Surround
        return self.pers_grid

    def grid_generator_torch(self):
        x, y = torch.meshgrid(torch.linspace(self.xmin, self.xmax, ipm_info.out_height,dtype=torch.float),
                              torch.linspace(self.ymin, self.ymax, ipm_info.out_width,dtype=torch.float))
        print(x.shape)
        self.grid_torch = torch.stack([x, y],0)
        print(self.grid_torch.shape)
        self.grid_torch_new = self.grid_torch.reshape(2,ipm_info.out_height*ipm_info.out_width)
        ones = -self.camera_info.camera_height * torch.ones(1,self.grid_torch_new.shape[1])
        self.uv_grid = torch.cat((self.grid_torch_new, ones),0)
        self.K = torch.from_numpy(self.K).float()
        self.R = torch.from_numpy(self.R).float()
        self.K = self.K.unsqueeze(0)
        self.R = self.R.unsqueeze(0)
        print(self.R.shape)
        self.uv_grid = torch.matmul(self.K, torch.matmul(self.R, self.uv_grid))  # w(u,v,1) (2d homo)
        print(self.uv_grid.shape)
        self.uv_grid = (self.uv_grid[:,:2,:] / self.uv_grid[:,2,:]).int()
        print(self.uv_grid.shape)
        plt.plot(self.uv_grid[:,0],self.uv_grid[:,1])
        plt.show()
        self.uv_grid = self.uv_grid * ((self.uv_grid[0] > ipm_info.left) * (self.uv_grid[0] < ipm_info.right) * \
                                       (self.uv_grid[1] > ipm_info.top) * (self.uv_grid[1] < ipm_info.bottom))
        self.uv_grid = self.uv_grid.reshape(2, ipm_info.out_height, ipm_info.out_width)
        self.u = self.uv_grid[0]
        self.v = self.uv_grid[1]
        self.u = (self.u[:]-ipm_info.out_width/2)/ipm_info.out_width
        self.v = (self.v[:]-ipm_info.out_height/2)/ipm_info.out_height
        #plt.plot(self.u,self.v)
        #plt.show()
        self.pers_grid = torch.stack([self.u, self.v],2)
        self.pers_grid = self.pers_grid.unsqueeze(0).float()

        return self.pers_grid

    def uv2xy(self, uvs):  # all points have z=0 (ground plane): find (x,y,z)_c first, then x_w, y_w = (R^-1 (x,y,z)_c)[:2]

        uvs = (uvs - np.array([self.camera_info.u_x, self.camera_info.u_y])[:, None]) / \
              np.array([self.camera_info.f_x, self.camera_info.f_y])[:,
              None]  # converted using camara intrinsic parameters
        #plt.scatter(uvs[0],uvs[1])
        #plt.show()
        uvs = np.vstack((uvs, np.ones(uvs.shape[1])))
        #print(uvs.shape)
        xyz_c = (-1440 / np.dot(self.normal_c.T, uvs)) * uvs  # solve the equation, get (x,y,z) on the ground plane in camera coordinates
        #plt.plot(xyz_c[0],xyz_c[1],xyz_c[2])
        #plt.show()
        xy_w = np.dot(self.R_inv, xyz_c)[:2, :]  # (x, y) on the ground plane in the world coordinates
        print((xy_w.shape))
        #plt.scatter(xy_w[1], xy_w[0])
        #plt.show()
        print((xy_w))
        return xy_w

    def __call__(self, img):
        return self.ipm(img)


    def ipm(self, img):
        out_img = np.zeros(img.shape)
        #print(img.shape)
        #out_img.imshow()
        out_img[...] = img[self.uv_grid]
        #plt.plot(self.xy_grid[0],self.xy_grid[1])
        #plt.show()
        return out_img



    def reverse_ipm(self, img, shape=None):
        if shape is None:
            shape = img.shape
        out_img = np.zeros(shape)
        out_img[self.uv_grid] = img
        return out_img


class _DictObjHolder(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]





if __name__ == "__main__":

    import matplotlib

    import matplotlib.pyplot as plt
    import cv2

    camera_info = _DictObjHolder({
        "f_x": 940,  # focal length x
        "f_y": 940,  # focal length y
        "u_x": 640,  # optical center x
        "u_y": 490,  # optical center y
        "camera_height": 1440,  # camera height in `mm`
        "pitch": 96,  # rotation degree around x
        "yaw": 0, # rotation degree around y
        "roll": 0 # rotation degree around z
    })
    ipm_info = _DictObjHolder({
        "input_width": 1280,
        "input_height": 960,
        "out_width": 1280,
        "out_height": 960,
        "left": 0,
        "right": 1280,
        "top": 0,
        "bottom": 960
    })

    img_path = '/home/mingshu/Downloads/image9786.png'
    im = cv2.imread(img_path)
    #im = im[0:1280,600:960]
    #im = cv2.resize(im, (200, 200), interpolation=cv2.INTER_AREA)
    img_torch = transforms.ToTensor()(im)
    #ipm = IPM(camera_info,ipm_info)
    #print(ipm.R)

    print(img_torch.shape)
    #ones_img = torch.ones([1,1280,960,2])
    print(img_torch.unsqueeze(0).shape)

    #grid1 = perspectivey_grid([1280,960])
    #print(grid1.size())
    #grid1 = grid1.unsqueeze(0)
    #print(grid1.shape)


    '''
    x_t, y_t = torch.meshgrid(torch.linspace(-1, 1, 200),
                           torch.linspace(-1, 1, 200))
    ones = torch.ones(40000,1)
    grid = torch.stack([x_t, y_t],2)
    print(grid.shape)
    grid = torch.flatten(grid, start_dim=0, end_dim=1)
    print(grid.shape)
    grid = torch.cat((grid,ones), 1)
    print(grid.shape)

    theta = torch.tensor([[1., 0., 0.],
                          [0., 1., 0. ],
                          [0., 0., 1.]])

    #theta = torch.randn(3,3)
    print(theta)
    grid = torch.matmul(grid,theta)
    print(grid.shape)
    #z = grid[:,2:3]
    #print(z.shape)
    #grid_xy = grid[:,0:2]
    #print(grid_xy.shape)
    grid_xyz = grid[:]/grid[2]
    grid_xy = grid_xyz[:,0:2]
    print(grid_xy.shape)

    grid_xy= grid_xy.reshape(200,200,-1)

    grid_xy = grid_xy.unsqueeze(0)
    print(grid_xy.shape)
    #plt.plot(grid[1],grid[0])
    #plt.show()
    out_put = F.grid_sample(img_torch.unsqueeze(0),grid_xy, mode='bilinear')
    new_img_torch = out_put[0]
    plt.imshow(new_img_torch.numpy().transpose(1,2,0))
    plt.show()
    '''



    #ipm = IPM(camera_info, ipm_info)
    #plt.plot(ipm.xy_grid[1], ipm.xy_grid[0])
    #plt.show()
    #plt.plot(ipm.uv_grid[1], ipm.uv_grid[0])
    #plt.show()


    img_path = '/home/mingshu/Downloads/image9786.png'
    im = cv2.imread(img_path)
    im = cv2.resize(im, (ipm_info.input_width, ipm_info.input_height), interpolation=cv2.INTER_AREA)
    img_torch = transforms.ToTensor()(Image.open(img_path))
    #img_torch = transforms.Normalize(mean=(0.5,0.5,0.5)
                                    # , std =(0.5,0.5,0.5))(img_torch)
    img_torch = img_torch.float()
    ipm = IPM(camera_info,ipm_info)
    #grid = ipm.pers_grid
    grid_np = ipm.grid_generator_np()
    grid_torch = ipm.grid_generator_torch()
    #grid1 = perspectivey_grid([1280,960])
    #print(grid1.size())
    #grid1 = grid1.unsqueeze(0)
    #print(grid1.shape)
    #grid3=grid1.unsqueeze(0)
    #print(grid1.shape)
    #print(grid.shape)
    #print(grid3.shape)
    #theta = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float)
    #theta = theta.view(-1, 3,3)
    #theta = theta.unsqueeze(0)

    #print(theta.shape)
    #grid4 = torch.matmul(theta, grid)


    out_put = F.grid_sample(img_torch.unsqueeze(0),grid_torch, mode='bilinear', padding_mode='border', align_corners=None)
    print(out_put.shape)
    new_img_torch = out_put[0]
    plt.imshow(new_img_torch.numpy().transpose(1,2,0))
    plt.show()
