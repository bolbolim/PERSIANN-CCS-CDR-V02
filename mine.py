#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import time
import gzip
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
from collections import Counter
from scipy import stats
import joblib
import sys
import multiprocessing
import os
import shutil
import filecmp
import cmocean
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import geopandas as gpd
from shapely.geometry import shape
import rasterio.features
from shapely.geometry import Polygon
import rasterio
from rasterio.features import rasterize
from rasterio.features import shapes


# In[3]:


def bi_to_np(bi, _type, endi=4, feed_back=True):
    reset_text = '\033[0m'
    if not isinstance(endi,str):
        raise ValueError("\033[1mdefine Endian!!")
    if feed_back:
        print(f"binary file length is:  \033[1m{len(bi)}{reset_text}")
        print(f"chosen data format is: \033[92m{str(_type)}{reset_text}")
    stream1 = io.BytesIO(bi)
    stream = stream1.read()
    dt = np.dtype(_type)
    if endi=='big':
        dt=dt.newbyteorder('>')
    array = np.frombuffer(stream, dtype=dt)
    return array


# In[4]:


def write_gz(file,fn,proj_folder=proj_folder,force_home=False,feed_back=True):
##     put in your home dir
    dist_dir=os.path.join("/zfs_data2/bolbolim/projects/",proj_folder)
    reset_text = '\033[0m'
    save_flag=False
    if type(file)==dict or type(file)==gpd.geodataframe.GeoDataFrame:
        save_flag=True
        SIZE=0
    elif type(file)==list:
        SIZE=0
        for item in file:
            SIZE+=item.itemsize*item.size
    else:
        SIZE=file.itemsize*file.size
    if sys.getsizeof(file)/(1024.0 ** 3)>3.9 or SIZE/(1024**3)>3.9 or save_flag:
        if feed_back:
            print(f"using \033[1m joblib{reset_text} your file will be saved as a\033[1m .npy{reset_text} file and then compressed in\033[1m .gz{reset_text}")
        if force_home:
            path=os.path.join(dist_dir,fn+'.npy.joblib')
        else:
            path=fn+'.npy.joblib'
        if feed_back:
            print(f'file path is:"{path}"')
            print(f'Compressing the \033[94m\033[1m{str(fn)}{reset_text} using joblib')
        joblib.dump(file,path+'.gz')
        if feed_back:
            print(f'\033[1m Compressing is Done!!{reset_text}')
    else:
        if feed_back:
            print(f"your file will be saved as a\033[1m .pickle{reset_text} file and then compressed in\033[1m .gz{reset_text}")
        pickled_f=pickle.dumps(file)
        if force_home:
            path=os.path.join(dist_dir,fn+'.pickle.gz')
        else:
            path=fn+'.pickle.gz'
        if feed_back:
            print(f'file path is:"{path}"')
            print(f'Compressing the \033[94m\033[1m{str(fn)}{reset_text} using Pickle')
        with gzip.open(path,'wb') as gz_f:
            gz_f.write(pickled_f)
        if feed_back:
            print(f'\033[1m Compressing is Done!!{reset_text}')


# In[5]:


def load_gz(fn,proj_folder=proj_folder,force_home=False,feed_back=True,plot=False,clim=None):
##     put in your home dir
    dist_dir=os.path.join("/zfs_data2/bolbolim/projects/",proj_folder)
    reset_text = '\033[0m'
    if feed_back:
        print(f"this function is to load a \033[1m .pickle/.joblib{reset_text} file that is compressed in\033[1m .gz{reset_text}")
    if force_home:
        path=os.path.join(dist_dir,fn+'.pickle.gz')
    else:
        path=fn
    if feed_back:
        print(f'Loading the file: "{path}"')
    if path[-10:-3]=='.joblib':
        if feed_back:
            print(f'using joblib')
        content=joblib.load(path)
    else:
        with gzip.open(path,"rb") as f:
            if feed_back:
                print(f'using Pickle')
            content=pickle.load(f)
    if feed_back:
        print(f'\033[1m File loaded!!{reset_text}')
    if plot:
        plt.imshow(content,clim=clim,cmap='viridis')
        plt.colorbar(orientation='horizontal')
    return content


# In[6]:


def c_map_maker(_map):
    map_colors = _map(np.arange(_map.N))
    map_colors[0, -1] = 0
    custom_map = LinearSegmentedColormap.from_list('cc', map_colors)
    c= ListedColormap(custom_map(np.arange(custom_map.N)))
    c.set_bad(color='lightgray')
    return c


# In[7]:


cc=c_map_maker(cmocean.cm.ice_r)


# In[8]:


##correct for diffrent reolutions and make a way to get gsd file seperatly

def load_map(path,dataset='perssian',_type=np.float32,feed_back=True,gz=True,plot=False,clim=None,get_pic=False             ,world=True,figsize=(10,6),cmap=cc,roll=False):
    _dict={
        'perssian':[(3000,9000),'small',True,np.float32,'/zfs_data2/bolbolim/projects/!FILES/world_map.pickle'],\
        'gpcp':[(72,144),'big',False,np.float32],"ccs_b1_3hr":[(3000,9000),'small',True,np.int16],\
        'cdr':[(480, 1440),'small',True,np.float32,'/zfs_data2/bolbolim/projects/!FILES/world_map_cdr.pickle'],\
        'st4':[(1000,1750),'big',True,np.int16],
        'IR':[(480,1440),'small',True,np.float32],
          }
    if dataset in ['gpcp','ccs_b1_3hr']:
        world=False
    if dataset not in _dict:
        raise ValueError("\033[1mGiven Dataset is not Recognized!!")
    else:
        size=_dict[dataset][0]
        endi=_dict[dataset][1]
        gz=_dict[dataset][2]
        _type=_dict[dataset][3]
    reset_text = '\033[0m'
    if not os.path.exists(path):
        print(f"\033[1mFile at {path} not found!!")
        return np.zeros([3000,9000])*np.nan
    if feed_back:
        print(f'loading the\033[1m {os.path.basename(path)}{reset_text} file to a\033[1m numpy{reset_text} with shape \033[94m\033[1m{size}{reset_text}')
    if gz:
        with gzip.open(path,"rb") as bin_f:
#             with open(f,'rb') as bin_f:
                bi=bin_f.read()
    else:
        with open(path,'rb') as bin_f:
                bi=bin_f.read()
    temp = bi_to_np(bi, _type, endi=endi, feed_back=feed_back)
    temp=temp.reshape(size)
    mask=temp<0
    arr=np.full(temp.shape,np.nan,dtype=_type)*np.nan
    arr[~mask]=temp[~mask]
    if dataset=='gpcp':
        if feed_back:
            print('Cutting the Polar area!')
        arr=arr[13:60,:]
    if dataset=='st4':
        arr=arr/100
    if plot:
        if feed_back:
            print("note while ploting the array is rolled to present typical presentaion of world map, but when you get the array it starts from Lon=0")
        figsize=(21,7)
        fig, ax1 = plt.subplots(figsize=(figsize[0]+figsize[0]*.001,figsize[1]))
        if world:
            with open(_dict[dataset][4], 'rb') as file:
                cont = pickle.load(file)
            cont.plot(ax=ax1, color='None', edgecolor='k')
        if dataset!='st4':
            arr2=np.roll(arr,arr.shape[1]//2)
        else:
            arr2=arr.copy()
        plt.imshow(arr2,clim=clim,cmap=cmap)
        plt.colorbar(orientation='vertical',shrink=.8,pad=figsize[0]*.001)
        plt.title(os.path.basename(path))
        plt.grid(True,alpha=0)
        ax1.set_xticks([])
        ax1.set_yticks([])
        fig.tight_layout()
    if roll:
        arr=np.roll(arr,arr.shape[1]//2) 
    if get_pic:
        return arr,fig
    else:
        return arr


# In[ ]:


def gps_to_index(pos,dataset):
    lat,lon=pos[0],pos[1]
    reset_text = '\033[0m'
    print(f"\033[1mthis function works with roll=False{reset_text}")
    _dict={'cdr':[(-60,60),(-180,180),(480,1440)],'perssian':[(-60,60),(-180,180),(3000,9000)],           'st4':[(10,50),(-135,-65),(1000,1750)],}#'st4_phu':[(20.64,50),(-125,-67),(734,1444)]}
    if not (mine.is_between(lat, _dict[dataset][0]) and mine.is_between(lon, _dict[dataset][1])):
        raise ValueError(f"\033[1mGiven Cordinate is not accepted{reset_text}")
    a=_dict[dataset][2][0]/(_dict[dataset][0][0]-_dict[dataset][0][1])
    b=-a*_dict[dataset][0][1]
    res=a*lat+b
    y=int(res)
    a=_dict[dataset][2][1]/(_dict[dataset][1][1]-_dict[dataset][1][0])
    b=a*125
    res=a*lon+b
    x=int(res)
    if dataset=='st4':
        a=1000/-40
        b=-50*a
        y=lat*a+b
        a=1750/70
        b=135*a
        x=lon*a+b
#     if x!=res:
#         raise ValueError(f"\033[1mGetting not int index!{reset_text}")
    return (y,x)

