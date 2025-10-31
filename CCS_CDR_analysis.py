#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import mine
import time
import gzip
import pickle
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from alive_progress import alive_bar
from scipy import stats
import sys
import seaborn as sb
import shapely
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio 
from rasterio.transform import from_origin
from shapely.geometry import Point
import multiprocessing
import cmocean
import pickle
from sklearn.metrics import mean_squared_error
import sklearn
from scipy.ndimage import zoom
import matplotlib.patches as mpatches
import statsmodels.api as sm
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr,spearmanr,kendalltau
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib as mpl


# In[ ]:


chrs_path=dict_path ## make a nested dictionary of stage IV, PCC-B1, and PCC-CPC paths and load it here, the keys should be datetime objects, and values should be file paths in .bin format


# In[ ]:


shapefile=shapefile_path ## load your shapefile as a numpy mask


# In[ ]:


def cutting(arr,mask=mask):
    temp=np.zeros([3000,9000])*np.nan
    temp[mask]=arr[mask]
    non_blank_indices = np.where(~np.isnan(temp))
    try:
        min_x, min_y = np.min(non_blank_indices, axis=1)
        max_x, max_y = np.max(non_blank_indices, axis=1)
    except ValueError:
        min_x, min_y=776 ,2465
        max_x, max_y=1282, 2719
        print(min_x, min_y)
        print(max_x, max_y)
#     min_x, min_y=655, 2347
#     max_x, max_y=1259, 2719
    finally:
        print(min_x, min_y)
        print(max_x, max_y)
#         return temp[min_x:max_x, min_y:max_y]


# In[ ]:


def cutting(arr,mask=mask):
    temp=np.zeros([3000,9000])*np.nan
    temp[mask]=arr[mask]
    non_blank_indices = np.where(~np.isnan(temp))
#     try:
#         min_x, min_y = np.min(non_blank_indices, axis=1)
#         max_x, max_y = np.max(non_blank_indices, axis=1)
#     except ValueError:
#         min_x, min_y=776 ,2465
#         max_x, max_y=1282, 2719
    min_x, min_y=306, 6567
    max_x, max_y=532, 6852
#     finally:
    return temp[min_x:max_x, min_y:max_y]


# # Climatalogy

# ## ploting region of study

# In[ ]:


gdf = gpd.read_file("/zfs_data2/bolbolim/projects/FILES/shapefiles/us_state/cb_2018_us_state_5m.shp")
path="/zfs_data2/bolbolim/projects/cccdr/new/extrem_stats_US/upper_missi/upper_missi_shapefile_folder"
shp=gpd.read_file(path)
shp=shp[shp['HYBAS_ID']==7040569650] ## for upper missi

shp=shp.iloc[0,:]


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 10))  # Adjust figsize as needed
sb.set_theme(style='white',palette='deep')
gdf.plot(ax=ax,color='silver',alpha=.7,edgecolor='white', linewidth=3)
# gdf.plot(ax=ax,facecolor='none',edgecolor='black', linewidth=1)
for geom in shp.geometry.geoms:
        gpd.GeoSeries([geom]).plot(ax=ax, edgecolor='#800020', facecolor='none', linewidth=.5, hatch='//')
marker={"Michael":'o'}#,'Ida':'^',"Laura":'s'}
# for name,yr in zip(['Michael','Ida','Laura'],[2018,2021,2020]):
for name,yr in zip(['Michael'],[2018]):
    df=storm_info_04deg(name,yr,land=False)
    sc=sb.scatterplot(data=df,x='Longitude',y='Latitude',hue='hours_since_start',legend=False,                      palette='magma_r',marker=marker[name],ax=ax,s=100,)
norm = plt.Normalize(df['hours_since_start'].min(), df['hours_since_start'].max())
sm1 = plt.cm.ScalarMappable(cmap='magma_r', norm=norm)
sm1.set_array([])
cbar=plt.colorbar(sm1, label='Hours Since Hurricane Genesis',fraction=.02,ax=ax)
cbar.set_label('Hours Since Hurricane Genesis', fontsize=20)
handles = [mlines.Line2D([], [], color='purple', marker=marker[name], linestyle='None', markersize=15, label=f'Hurricane {name}')
           for name in marker]
hatched_rect = mpatches.Patch(edgecolor='#800020', facecolor='none', hatch='//', label='Upper Missisipi Basin',)

# Append the hatched rectangle to the handles
handles.append(hatched_rect)
ax.legend(handles=handles,loc='lower right',fontsize=15)
ax.set_xlim(-125, -65)
ax.set_ylim(25, 50)
# plt.title('Region of Study',fontsize=30,weight='bold')
plt.xlabel('Longitude',fontsize=20)
plt.ylabel('Latitude',fontsize=20)
plt.tick_params(axis='both', labelsize=15)
cbar.ax.tick_params(labelsize=15)
plt.show()
# mine.save_fig('region_of_study')


# In[ ]:


# plt.figure(figsize=(20,10))
# cmap = mcolors.ListedColormap(['none', 'silver'])
# sb.set_theme(style='white',palette='deep')
# plt.imshow(conus_st4,cmap=cmap)
# marker={"Michael":'o','Ida':'^',"Laura":'s'}
# for name,yr in zip(['Michael','Ida','Laura'],[2018,2021,2020]):
#     df=storm_info_04deg(name,yr,land=False)
#     sc=sb.scatterplot(data=df,x='x_st4',y='y_st4',hue='hours_since_start',legend=False,\
#                       palette='magma_r',marker=marker[name])
# norm = plt.Normalize(df['hours_since_start'].min(), df['hours_since_start'].max())
# sm1 = plt.cm.ScalarMappable(cmap='magma_r', norm=norm)
# sm1.set_array([])
# plt.colorbar(sm1, label='Hours since start',fraction=.03)
# handles = [mlines.Line2D([], [], color='purple', marker=marker[name], linestyle='None', markersize=10, label=name)
#            for name in marker]
# plt.legend(handles=handles,loc='lower right')
# plt.xlim([200,None])
# plt.ylim([700,None])


# ## cutting the files

# ### CHRS files

# In[ ]:


state='upper_missi'
version='cpc_based'
print(state,",",version)


# In[ ]:


shapefile=SHAPEFILE_PATH ## load your BASIN shapefile as a numpy mask
mask= shapefile.astype(bool)
plt.imshow(mask)
plt.show()


# In[ ]:


def cutting(arr,mask=mask):
    arr[~mask]=np.nan
    non_blank_indices = np.where(mask)
    try:
        min_x, min_y = np.min(non_blank_indices, axis=1)
        max_x, max_y = np.max(non_blank_indices, axis=1)
#         print(min_x, min_y,max_x, max_y)
    except ValueError:
#         print('The code needs a HAUWWK TUAAH!!')
        min_x, min_y =42 ,221 
        max_x, max_y= 138, 451
    finally:
        return arr[min_x:max_x, min_y:max_y]


# In[ ]:


def making_yearly_raw_data(yr,chrs_path=chrs_path,version=version,                           fl=OUTPUT_PATH):
    dataset='perssian'     
    v={"cpc_based":"ccs_cdr_cpc_daily","b1_based":"ccs_cdr_b1_daily","cdr_based":"cdr_daily"}
#     print(v[version])
    fn=str(yr)
    temp0=np.zeros([3000,9000])
    if version=='cdr_based':
        dataset='cdr'
        temp0=np.zeros([480,1440])
    arr=cutting(temp0)
    days=[q for q in chrs_path[v[version]].keys() if q.year==yr]
    print(f'starting with {yr}')
    for day in days:
#         print(day)
        try:
            temp=mine.load_map(chrs_path[v[version]][day],feed_back=False,dataset=dataset,roll=True)
            cutted=cutting(temp)
            arr=np.dstack((arr,cutted))
        except:
            try:
                temp=mine.load_gz(chrs_path[v[version]][day],feed_back=False)#,dataset=dataset,roll=True)
                temp=np.roll(temp,int(temp.shape[1]/2))
                cutted=cutting(temp)
                arr=np.dstack((arr,cutted))
            except:
                print(f'skipping {day}')
    arr=arr[:,:,1:]
    mine.write_gz(arr,os.path.join(fl,fn))
    


# In[ ]:


data_list = [[y] for y in range(2003,2025)]
data_list=[[2024]]
with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(making_yearly_raw_data, data_list)
print("All Tasks are Done!")


# In[ ]:


print('done')


# ### ST4 Files

# In[ ]:


def cutting(arr,mask=mask):
    arr[~mask]=np.nan
    non_blank_indices = np.where((mask))
    try:
        min_x, min_y = np.min(non_blank_indices, axis=1)
        max_x, max_y = np.max(non_blank_indices, axis=1)
    except ValueError:
        PRINT('eRROR')
    finally:
        return arr[min_x:max_x, min_y:max_y]


# In[ ]:


def making_yearly_raw_data(yr,chrs_path=chrs_path,version=version,                           fl=OUTPUT_PATH"):
    v={"cpc_based":"ccs_cdr_cpc_day","b1_based":"ccs_cdr_b1_daily","st4_based":'st4_daily'}
    fn=str(yr)
    temp0=np.zeros([1000,1750])
    arr=cutting(temp0)
    days=[q for q in chrs_path[v[version]].keys() if q.year==yr]
    print(f'starting with {yr}')
    for day in days:
#         print(day)
        temp=mine.load_gz(chrs_path[v[version]][day],feed_back=False)
        cutted=cutting(temp)
        arr=np.dstack((arr,cutted))
    arr=arr[:,:,1:]
    mine.write_gz(arr,os.path.join(fl,fn))
#     return arr


# In[ ]:


years=range(2002,2024)
len(years)


# In[ ]:


pool = multiprocessing.Pool(processes=10)
# chosen_years=[2016]#,2016,2017]
# data_list = [[y] for y in range(1983,2023) if y in chosen_years]
data_list = [[y] for y in years]
pool.starmap(making_yearly_raw_data, data_list)
pool.close()
pool.join()
print("All Tasks are Done!")


# In[ ]:


print('Done')


# ## doing the analysis

# In[ ]:


version='cpc_based'


# In[ ]:


def cutting_mask(mask):
    mask=mask.astype(float)
    mask[mask==0]=np.nan
    non_blank_indices = np.where(~np.isnan(mask))
    min_x, min_y = np.min(non_blank_indices, axis=1)
    max_x, max_y = np.max(non_blank_indices, axis=1)
    mask[np.isnan(mask)]=0
    mask=mask.astype(bool)
    return mask[min_x:max_x, min_y:max_y]


# In[ ]:


shapefile=SHAPAFILE_PATH ## LOAD SHAPEFILE AS NUMPY ARRAY
plt.imshow(shapefile)


# In[ ]:


def cdd_cwd(percip,y,x):
    max_dry,max_wet,dry,wet=0,0,0,0
    for day in range(percip.shape[2]):
        if percip[y,x,day]<1:
            dry+=1
            wet=0
            max_dry=max(dry,max_dry)
        elif percip[y,x,day]>=1:
            wet+=1
            max_wet=max(wet,max_wet)
            dry=0   
    return max_dry,max_wet


# In[ ]:


def extreme_stat_fun(path,fl=FILE_PATHS,shapefile=shapefile):
    percip=mine.load_gz(path,feed_back=False)
    fn=path[-14:-10]
    print(f'starting with {fn}')
    sample=np.zeros([percip.shape[0],percip.shape[1]])
    sdii=np.zeros_like(sample)
    r10=np.zeros_like(sample)
    r10tot=np.zeros_like(sample)
    cdd=np.zeros_like(sample)
    cwd=np.zeros_like(sample)
    r95ptot=np.zeros_like(sample)
    r99ptot=np.zeros_like(sample)
    prcptot=np.zeros_like(sample)
    prcptot[~shapefile]=np.nan
    nan_sum=np.zeros_like(sample)
    nan_sum[~shapefile]=np.nan
    r99ptot[~shapefile]=np.nan
    r95ptot[~shapefile]=np.nan
    cwd[~shapefile]=np.nan
    cdd[~shapefile]=np.nan
    r10tot[~shapefile]=np.nan
    r10[~shapefile]=np.nan
    sdii[~shapefile]=np.nan
    del sample
    prcptot=np.nanmean(percip,axis=2)
    prcptot*=365
    prcptot[~shapefile]=np.nan
    nan_sum=np.isnan(percip).sum(axis=2)
    p95=np.nanpercentile(percip, 95,axis=2)
    p95[~shapefile]=np.nan
    p99=np.nanpercentile(percip, 99,axis=2)
    p99[~shapefile]=np.nan
    for y in range(percip.shape[0]):
        for x in range(percip.shape[1]):
            if shapefile[y,x]:
                cdd[y,x],cwd[y,x]=cdd_cwd(percip,y,x)
                temp=percip[y,x,:]
                temp1=temp[temp>1]
                sdii[y,x]=np.nanmean(temp1)
                temp1=temp[temp>10]
                r10[y,x]=len(temp1)
                r10tot[y,x]=np.nansum(temp1)
                temp1=temp[temp>p95[y,x]]
                r95ptot[y,x]=np.nansum(temp1)
                temp1=temp[temp>p99[y,x]]
                r99ptot[y,x]=np.nansum(temp1)
    prcptot[~shapefile]=np.nan
    nan_sum=nan_sum.astype(float)
    nan_sum[~shapefile]=np.nan
    r99ptot[~shapefile]=np.nan
    r95ptot[~shapefile]=np.nan
    cwd[~shapefile]=np.nan
    cdd[~shapefile]=np.nan
    r10tot[~shapefile]=np.nan
    r10[~shapefile]=np.nan
    sdii[~shapefile]=np.nan
    extrem_stat={"cdd":cdd,"cwd":cwd,"sdii":sdii,"r10":r10,"r10tot":r10tot,"r95ptot":r95ptot,"r99ptot":r99ptot,                    "prcptot":prcptot,"nan_sum":nan_sum}
    mine.write_gz(extrem_stat,os.path.join(fl,fn))


# In[ ]:


fl=FILES_PATH ## GIVE THE PATH TO FILES FROM PREV. SECTION
fns=os.listdir(fl)
fns=sorted(fns)


# In[ ]:


pool2 = multiprocessing.Pool(processes=20)
data_list=[[os.path.join(fl,fn)] for fn in fns]
pool2.starmap(extreme_stat_fun, data_list)
pool2.close()
pool2.join()
print("All Tasks are Done!")


# In[ ]:


print('done')


# ## visualisation

# ### making the file files

# In[50]:


state='upper_missi'
version='cpc_based'
res='0.025deg'
print(state,",",version)


# In[51]:


fl=FILES_PATH ## GIVE THE PATH TO FILES FROM PREV. SECTION
fns=os.listdir(fl)
fns=sorted(fns)


# In[ ]:


cdd, cwd, sdii, r10, r10tot, r95ptot, r99ptot, prcptot, nan_sum={},{},{},{},{},{},{},{},{},


# In[ ]:


for fn in fns:
    dict1=mine.load_gz(os.path.join(fl,fn))
    yr=fn[:4]
    for _dict,var in zip([cdd, cwd, sdii, r10, r10tot, r95ptot, r99ptot, prcptot, nan_sum],                         ['cdd','cwd','sdii','r10','r10tot','r95ptot','r99ptot','prcptot','nan_sum']):
#         print(var)
        _dict[yr]=dict1[var]
    


# In[ ]:


fl=FILES_PATH ## GIVE THE PATH TO FILES FROM PREV. SECTION
for _dict,fn in zip([cdd, cwd, sdii, r10, r10tot, r95ptot, r99ptot, prcptot, nan_sum],                     ['cdd','cwd','sdii','r10','r10tot','r95ptot','r99ptot','prcptot','nan_sum']):
    mine.write_gz(_dict,os.path.join(fl,fn))


# In[ ]:


# path="/zfs_data2/bolbolim/projects/!FILES/shapefiles/"
# shapefile=mine.load_gz(os.path.join(path,'mask',f'{state}_st4.pickle.gz'),feed_back=False)
# shapefile=cutting_mask(shapefile)
# mask= shapefile.astype(bool)
# plt.imshow(mask)


# ## PLOTS

# In[ ]:





# In[52]:


state='upper_missi'
# state='conus'
# state='colorado'
res='0.025deg'
state


# In[53]:


def cutting_mask(mask):
    mask=mask.astype(float)
    mask[mask==0]=np.nan
    non_blank_indices = np.where(~np.isnan(mask))
    min_x, min_y = np.min(non_blank_indices, axis=1)
    max_x, max_y = np.max(non_blank_indices, axis=1)
    mask[np.isnan(mask)]=0
    mask=mask.astype(bool)
    return mask[min_x:max_x, min_y:max_y]


# In[54]:


# sb.set_theme(style=None)
shapefile=SHAPEFILE ## load your shapefile as numpy mask 
shapefile=cutting_mask(shapefile)
mask= shapefile.astype(bool)
plt.imshow(mask)


# ### plotting 0.025 deg

# In[55]:


st4_fl=PATH_TO_STAGE_IV_FILES
b1_fl=PATH_TO_B1_FILES
cpc_fl=PATH_TO_CPC_FILES


# In[56]:


# sb.set_theme(style=None)
shapefile=SHAPEFILE ## load your shapefile as numpy mask 
shapefile=cutting_mask(shapefile)
mask= shapefile.astype(bool)
plt.imshow(mask)


# #### ploting bar cpc , b1

# In[57]:


fns=sorted(os.listdir(b1_fl))
fns=fns[:-1]
cdd, cwd, sdii, r10, r10tot, r95ptot, r99ptot, prcptot, nan_sum={},{},{},{},{},{},{},{},{},
for fn in fns:
    dict_b1=mine.load_gz(os.path.join(b1_fl,fn),feed_back=False)
    try:
        dict_cpc=mine.load_gz(os.path.join(cpc_fl,fn),feed_back=False)
    except:
        ww=0
    yr=fn[:4]
    for _dict,var in zip([cdd, cwd, sdii, r10, r10tot, r95ptot, r99ptot, prcptot, nan_sum],                         ['cdd','cwd','sdii','r10','r10tot','r95ptot','r99ptot','prcptot','nan_sum']):
        try:
            _dict.update({yr:[dict_b1[var],dict_cpc[var]]})
        except:
            _dict.update({yr:[dict_b1[var],np.nan]})


# In[58]:


_dict={'cdd':cdd,'cwd':cwd,'sdii':sdii,'r10':r10,'r10tot':r10tot,'r95ptot':r95ptot,       'r99ptot':r99ptot,'prcptot':prcptot,'nan_sum':nan_sum}
for key in _dict.keys():
    for yr in range(1983,2020):
        for i in range(2):
            _dict[key][str(yr)][i][~mask]=np.nan


# In[60]:


def plot_comparision_chrs_all_vars(_dict=_dict,state=state):
    sb.set_theme(style='darkgrid')
    unit={'CDD':" (Days)",'CWD':" (Days)",'SDII':" (mm/day)",'R10':" (Days)",'R10TOT':" (mm)",'R95PTOT':" (mm)",           'R99PTOT':" (mm)",'PRCPTOT':" (mm)",'NAN_SUM':" "}
    custom_colors = {
            "PCC-B1 : 1983-2000": sb.color_palette()[-1],  # Light Blue
            "PCC-B1 : 2001-2024": sb.color_palette()[0], # Dark Blue
#             'Pre 2000 CPC': sb.color_palette()[-2],  
            "PCC-CPC : 2001-2024": sb.color_palette()[1]   
        }
    axes=[]
    fig=plt.figure(figsize=(30,25))
    for title,_ in zip(_dict.keys(),range(len(_dict.keys()))):
        if title=='nan_sum':
            continue
        row_id=_//3
        col_id=_%3
        vals_b1,vals_cpc,key1=[],[],[]
        axes.append(plt.subplot2grid((3,3), (row_id, col_id)))
        for key in _dict[title].keys():
            vals_b1.append(np.nanmean(_dict[title][key][0]))
            vals_cpc.append(np.nanmean(_dict[title][key][1]))
            key1.append(key)
        title=title.upper()
        title2=f'{title} abs. Error (%)'
        title+=unit[title]
        df=pd.DataFrame([key1,vals_b1,vals_cpc])
        df=df.T
        df.columns=['Year','B1','CPC']
        df=df.astype(np.float32)
        df["Year"]=df["Year"].astype(int)
        df['Group']=1990
        for i in range(len(df)):
            if df.iloc[i,0]>2000:
                df.iloc[i,3]=2000 
        df["Group"]=df["Group"].astype(str)
        _max=int(np.nanmax(df.iloc[:,1:3].values)*1.1+1)
        _min=max(0,int(np.nanmin(df.iloc[:,1:3].values))*.9)
        df_melted = pd.melt(df, id_vars=["Year","Group"], var_name="Data Base", value_name=title)
        df_melted['Data Set']=df_melted['Group']+df_melted['Data Base']
        converter={"1990B1":"PCC-B1 : 1983-2000","2000B1":"PCC-B1 : 2001-2024","2000CPC":"PCC-CPC : 2001-2024"}#,"1990CPC":"Pre 2000 CPC"}
        df_melted=df_melted[df_melted['Data Set']!="1990CPC"]
        for i in range(len(df_melted)):
            df_melted.iloc[i,4]=converter[df_melted.iloc[i,4]]
        boxplot =sb.boxplot(data=df_melted,x='Data Set',y=title,ax=axes[_],palette=custom_colors)
        boxplot.set_ylabel(title,fontsize=25)
        save_fn=title+'_CHRS_'+state+"_all_vars"
    for ax in axes:
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.tick_params(axis='both', which='major', labelsize=20)
    handles = [mpatches.Patch(color=color, label=label) for label, color in custom_colors.items()]
    fig.legend(handles=handles, title="Legend",loc='lower right', bbox_to_anchor=(.8, .2),fontsize=20, title_fontsize=20)


# In[61]:


plot_comparision_chrs_all_vars()


# #### ploting bar st4_cpc_b1

# In[62]:


fns=sorted(os.listdir(st4_fl))
fns
cdd, cwd, sdii, r10, r10tot, r95ptot, r99ptot, prcptot, nan_sum={},{},{},{},{},{},{},{},{},
for fn in fns:
    dict_st4=mine.load_gz(os.path.join(st4_fl,fn),feed_back=False)
    dict_b1=mine.load_gz(os.path.join(b1_fl,fn),feed_back=False)
    dict_cpc=mine.load_gz(os.path.join(cpc_fl,fn),feed_back=False)
    yr=fn[:4]
    for _dict,var in zip([cdd, cwd, sdii, r10, r10tot, r95ptot, r99ptot, prcptot, nan_sum],                         ['cdd','cwd','sdii','r10','r10tot','r95ptot','r99ptot','prcptot','nan_sum']):
        _dict.update({yr:[dict_b1[var],dict_st4[var],dict_cpc[var]]})


# In[63]:


_dict={'cdd':cdd,'cwd':cwd,'sdii':sdii,'r10':r10,'r10tot':r10tot,'r95ptot':r95ptot,       'r99ptot':r99ptot,'prcptot':prcptot,'nan_sum':nan_sum}
for key in _dict.keys():
    for yr in range(2002,2020):
        for i in range(3):
            _dict[key][str(yr)][i][~mask]=np.nan


# In[64]:


def plot_comparison_all_var(_dict=_dict,state=state):
    sb.set_theme(style='darkgrid')
    col = {
            'PCC-B1': sb.color_palette()[0], # Dark Blue
            'PCC-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
        }
    unit={'CDD':" (Days)",'CWD':" (Days)",'SDII':" (mm/day)",'R10':" (Days)",'R10TOT':" (mm)",'R95PTOT':" (mm)",           'R99PTOT':" (mm)",'PRCPTOT':" (mm)",'NAN_SUM':" "}
    sb.set_theme(style='darkgrid')
    axes=[]
    fig=plt.figure(figsize=(30,25),dpi=300)
    for title,_ in zip(_dict.keys(),range(len(_dict.keys()))):
        if title=='nan_sum':
            continue
        row_id=_//3
        col_id=_%3
        vals_b1,vals_cpc,key1=[],[],[]
        axes.append(plt.subplot2grid((3, 3), (row_id, col_id)))
        vals_st4,vals_b1,vals_cpc,key1=[],[],[],[]
        for key in _dict[title].keys():
            vals_b1.append(np.nanmean(_dict[title][key][0]))
            vals_st4.append(np.nanmean(_dict[title][key][1]))
            vals_cpc.append(np.nanmean(_dict[title][key][2]))
            key1.append(key)
        title=title.upper()
        title2=f'{title} abs. Error (%)'
        title+=unit[title]
        df=pd.DataFrame([key1,vals_b1,vals_st4,vals_cpc])
        df=df.T
        df.columns=['Year','PCC-B1','STAGE IV','PCC-CPC']
        df=df.astype(np.float32)
        df["Year"]=df["Year"].astype(int)
        _max=int(np.nanmax(df.iloc[:,1:].values)*1.1+1)
        _min=max(0,int(np.nanmin(df.iloc[:,1:].values))*.9)
        df_melted = pd.melt(df, id_vars=["Year"], var_name="Data Base", value_name=title)
        df_dif=pd.DataFrame()
        df_dif['Year']=df['Year'].astype(int)
        df_dif["Year"]=pd.to_datetime(df_dif["Year"], format='%Y')
        df_dif["PCC-B1"]=abs(df['STAGE IV']-df['PCC-B1'])/abs(df['STAGE IV']+np.finfo(float).eps)*100
        df_dif["PCC-CPC"]=abs(df['STAGE IV']-df['PCC-CPC'])/abs(df['STAGE IV']+np.finfo(float).eps)*100
        _max2=int(np.nanmax(df_dif.iloc[:,1:].values)*1.1+1)
        _min2=max(0,int(np.nanmin(df_dif.iloc[:,1:].values))*.9)
        df_dif_melted = pd.melt(df_dif, id_vars=["Year"], var_name="Data Base", value_name=title2)
        boxplot=sb.boxplot(data=df_melted,x='Data Base',y=title,ax=axes[_],palette=col)
        boxplot.set_ylabel(title,fontsize=25)
    for ax in axes:
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.tick_params(axis='both', which='major', labelsize=20)
    handles = [mpatches.Patch(color=color, label=label) for label, color in col.items()]
    fig.legend(handles=handles, title="Legend",loc='lower right', bbox_to_anchor=(.8, .2),fontsize=20,               title_fontsize=20)
    


# In[65]:


plot_comparison_all_var()


# # Huriicane events

# In[6]:


conus_per=SHAPEFILE
conus_cdr=CDR_SHAPE_FILE
conus_st4=STAGE_IV_CDR_SHAPE_FILE


# In[7]:


ibtracs=LOAD_ibtracs_CSV_FILE
ibtracs=ibtracs.iloc[1:,:]
ibtracs['SEASON']=ibtracs["SEASON"].astype(int)
ibtracs=ibtracs[ibtracs['SEASON']>=2000]
ibtracs.head(2)


# ## 0.025  deg daily analysis

# ### functions

# In[8]:


def gps_to_index(pos,dataset):
    lat,lon=pos[0],pos[1]
    reset_text = '\033[0m'
    _dict={'cdr':[(-60,60),(-180,180),(480,1440)],'perssian':[(-60,60),(-180,180),(3000,9000)],           'st4':[(10,50),(-135,-65),(1000,1750)],}#'st4_phu':[(20.64,50),(-125,-67),(734,1444)]}
    if not (mine.is_between(lat, _dict[dataset][0]) and mine.is_between(lon, _dict[dataset][1])):
        raise ValueError(f"\033[1mGiven Cordinate is not accepted{reset_text}")
    a=_dict[dataset][2][0]/(_dict[dataset][0][0]-_dict[dataset][0][1])
    b=-a*_dict[dataset][0][1]
    res=a*lat+b
    y=int(res)
    b=0
    a=_dict[dataset][2][1]/2/-180
    res=a*lon+b
#     print(res)
    if lon<0:
        res=_dict[dataset][2][1]-res
    x=int(res)
    if dataset=='st4':
        a=1000/-40
        b=-50*a
        y=lat*a+b
        a=1750/70
        b=135*a
        x=lon*a+b
    return (y,x)


# In[9]:


def storm_info_04deg(name,year,ibtracs=ibtracs,land=True):
    name=name.upper()
    df=ibtracs[(ibtracs['NAME']==name) & (ibtracs['SEASON']==year)]
    df=df.loc[:,['LAT',"LON",'USA_R34_NE','USA_R34_NW','USA_R34_SE','USA_R34_SW','ISO_TIME',"STORM_SPEED",'LANDFALL']]
    names=['USA_R34_NE','USA_R34_NW','USA_R34_SE','USA_R34_SW',"STORM_SPEED",'LANDFALL']
    for i in names:
        df[i]=pd.to_numeric(df[i], errors='coerce')
    for i in ['LAT','LON']:
        df[i]=pd.to_numeric(df[i], errors='coerce')    
    df=df[(df['LAT'] > 10) & (df['LAT'] < 50) & (df['LON'] > -135) & (df['LON'] < -65)]
    df['rad']=df.iloc[:,2:6].mean(axis=1)/60
    lat,lon,date,speed,rad,landfall=df['LAT'].values,df['LON'].values,df['ISO_TIME'].values,df['STORM_SPEED'].values    ,df['rad'].values,df["LANDFALL"].values
    lat=np.array(lat, dtype=float)
    lon=np.array(lon, dtype=float)
    pos_=[]
    for _ , __ in zip(lat,lon):
        p=(_,__)
        pos_.append(p)
    track,track2,track3=[],[],[]
    for p in pos_:
        new=gps_to_index(p,'st4')
        new2=gps_to_index(p,'perssian')
        new3=gps_to_index(p,'cdr')
        track.append(new)
        track2.append(new2)
        track3.append(new3)
    track=np.array(track)
    track2=np.array(track2)
    track3=np.array(track3)
    storm_df=pd.DataFrame(columns=["x_st4",'y_st4',"x_per",'y_per',"x_cdr",'y_cdr','date','speed'])
    storm_df['Latitude']=lat
    storm_df['Longitude']=lon
    storm_df["x_st4"]=track[:,1]
    storm_df["y_st4"]=track[:,0]
    storm_df["x_per"]=track2[:,1]
    storm_df["y_per"]=track2[:,0]
    storm_df["x_cdr"]=track3[:,1]
    storm_df["y_cdr"]=track3[:,0]
    storm_df['date']=date
    storm_df['speed']=speed
    storm_df['radius']=rad
    storm_df['date'] = pd.to_datetime(storm_df['date'])
    storm_df['Landfall']=landfall
    reference_time = storm_df['date'].min()
    storm_df['hours_since_start'] = (storm_df['date'] - reference_time) / pd.Timedelta(hours=1)
    if land==True:
        first_index = storm_df[storm_df['Landfall'] == 0].index[0]
        storm_df=storm_df.iloc[first_index:,:]
    return storm_df


# In[10]:


def cutting_map(arr):
    non_blank_indices = np.where(~np.isnan(arr))
    min_x, min_y = np.min(non_blank_indices, axis=1)
    max_x, max_y = np.max(non_blank_indices, axis=1)
    return arr[min_x:max_x, min_y:max_y]


# In[11]:


def images(lv1_data,df,lim=(0,120),_cmap='jet',res='day',rad_factor=1,give_me_cor=False,folder=None):
    num=3
    sb.set_theme(style='white',palette='pastel')
    fig=plt.figure(figsize=(6,2*len(lv1_data.keys())))
    if res=='3hr':
        fig=plt.figure(figsize=(6,2*len(lv1_data.keys())))
    if res=='cdr':
        fig=plt.figure(figsize=(8,2*len(lv1_data.keys())))
        num=4
    pl=0
    axes=[]
    for day,i in zip(lv1_data.keys(),range(len(lv1_data.keys()))):
        for dataset in lv1_data[day].keys():
            if res=='3hr':
                latitude_values = df[df['date'] == day]['Latitude'].values
                longitude_values = abs(df[df['date'] == day]['Longitude'].values)
            else:
                rad=df['radius'].max()
                latitude_values = df[df['date'].dt.day == day.day]['Latitude'].values
                lat_min,lat_max=min(latitude_values)-rad_factor*rad,max(latitude_values)+rad_factor*rad
                latitude_values=[np.mean([lat_min,lat_max])]
                longitude_values = abs(df[df['date'].dt.day == day.day]['Longitude'].values)
                lon_min,lon_max=min(longitude_values)-rad_factor*rad,max(longitude_values)+rad_factor*rad
                longitude_values=[np.mean([lon_min,lon_max])]
            axes.append(plt.subplot2grid((len(lv1_data.keys()),len(lv1_data[day].keys())),(i,pl%num)))  
            cax=axes[-1].imshow(lv1_data[day][dataset],cmap=_cmap,clim=lim)
            axes[-1].set_yticks([lv1_data[day][dataset].shape[0]/2])
            axes[-1].set_xticks([lv1_data[day][dataset].shape[1]/2])
            font_properties = {'fontweight': 'ultralight', 'fontstyle': 'italic'}
            axes[-1].set_yticklabels([f'{lat} N' for lat in latitude_values],rotation=90,**font_properties)
            axes[-1].set_xticklabels([f'{lat} W' for lat in longitude_values],**font_properties)
            axes[-1].yaxis.set_tick_params(pad=0)
            axes[-1].xaxis.set_tick_params(pad=3)
            axes[-1].tick_params('both',direction='inout',which='major', length=5, width=2, colors='k',                                     grid_color='k', grid_alpha=1, bottom=True, left=True)
            if pl//len(lv1_data[day].keys())==0:
                plt.title(dataset,weight="heavy")
            if dataset=='STAGE IV':
                label=day
                if res=='3hr':
                    label=label.strftime('%b-%d-%H')
                    plt.ylabel(label+" hr")
                else:
                    label=label.strftime('%b-%d')
                    plt.ylabel(label)
            pl+=1
    if not give_me_cor:
        for ax in axes:
            ## ax lat lon only works with cutting=True
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(bottom=False, left=False)
    cbar = fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('mm')
    if res=='3hr':
        cbar.set_label('mm')


# In[12]:


def rmse_fun(lv1_data,ref):
    rmse={}
    for date in lv1_data.keys(): 
        obs=lv1_data[date][ref].copy().flatten()
#         print(obs.shape)
        obs_mask=np.isnan(obs)
        rmse[date]={}
        for dataset in [f for f in lv1_data[date].keys() if f!=ref]:
            temp=lv1_data[date][dataset].copy().flatten()
#             print(dataset,temp.shape)
            common=obs_mask | np.isnan(temp)
            rmse[date][dataset]=np.sqrt(mean_squared_error(obs[~common], temp[~common]))#,squared=False)
    return rmse


# In[13]:


def skill_score_fun(lv1_data,ref,thres):
    skill={}
    for date in lv1_data.keys(): 
        obs=lv1_data[date][ref].copy().flatten()
        obs_mask=np.isnan(obs)
        obs[obs<thres]=-1
        obs[obs>=thres]=1
        obs[obs_mask]=np.nan
        skill[date]={}
        for dataset in [f for f in lv1_data[date].keys() if f!=ref]:
            temp=lv1_data[date][dataset].copy().flatten()
            mask=np.isnan(temp)
            temp[temp<thres]=-1
            temp[temp>=thres]=1
            temp[mask]=np.nan
            common=obs_mask | mask
            try:
                tn, fp, fn, tp=confusion_matrix(obs[~common], temp[~common]).ravel()
                skill[date][dataset]={}
                skill[date][dataset]['POD'] = tp / (tp + fn)  # Probability of Detection
                skill[date][dataset]['FAR'] = fp / (tp + fp)  # False Alarm Ratio
                skill[date][dataset]['CSI'] = tp / (tp + fn + fp)  # Critical Success Index
                numerator = 2 * (tp * tn - fn * fp)
                denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
                if denominator != 0:
                    skill[date][dataset]['HSS'] = numerator / denominator  # Heidke Skill Score
                else:
                    skill[date][dataset]['HSS'] = 0  # Handle division by zero
            except:
                skill[date][dataset]={}
                skill[date][dataset]['POD'] = np.nan  # Probability of Detection
                skill[date][dataset]['FAR'] = np.nan  # False Alarm Ratio
                skill[date][dataset]['CSI'] = np.nan  # Critical Success Index
                skill[date][dataset]['HSS'] = np.nan
    return skill


# In[14]:


def mssim_fun(rr,ref):
    lv1_data=rr.copy()
    mssim={}
    for date in lv1_data.keys(): 
        obs=lv1_data[date][ref].copy()#.flatten()
        obs_mask=np.isnan(obs)
        mssim[date]={}
        for dataset in [f for f in lv1_data[date].keys() if f!=ref]:
            obs_copy=obs.copy()
            pre=lv1_data[date][dataset].copy()
            mask=np.isnan(obs_copy) | np.isnan(pre)
            obs_copy[mask]=-1
            pre[mask]=-1
            try:
                mssim[date][dataset]=ssim(obs_copy, pre, full=True)[0]
            except:
                try:
                    mssim[date][dataset] = ssim(obs_copy, pre, win_size=5, full=True)[0]
                except:
                    mssim[date][dataset] = ssim(obs_copy, pre, win_size=3, full=True)[0]
    return mssim


# In[15]:


def total_precipitation(lv1_data):
    total={}
    for date in lv1_data.keys():
        total[date]={}
        for dataset in lv1_data[date].keys():
#             total[date][dataset]=np.nansum(lv1_data[date][dataset])
            total[date][dataset]=np.nanmean(lv1_data[date][dataset])/3
    return total


# In[16]:


def percent(lv1_data):
    percent={}
    for date in lv1_data.keys():
        percent[date]={10:{},50:{},95:{},99:{}}
        for dataset in lv1_data[date].keys():
            percent[date][10][dataset]=np.nanpercentile(lv1_data[date][dataset], 10)
            percent[date][50][dataset]=np.nanpercentile(lv1_data[date][dataset], 50)
            percent[date][95][dataset]=np.nanpercentile(lv1_data[date][dataset], 95)
            percent[date][99][dataset]=np.nanpercentile(lv1_data[date][dataset], 99)
    return percent


# In[17]:


def cor_fun(lv1_data,ref):
    cor={}
    for date in lv1_data.keys(): 
        obs=lv1_data[date][ref].copy().flatten()
        obs_mask=np.isnan(obs)
        cor[date]={}
        for dataset in [f for f in lv1_data[date].keys() if f!=ref]:
            temp=lv1_data[date][dataset].copy().flatten()
            mask=np.isnan(temp)
            common=obs_mask | np.isnan(temp)
            cor[date][dataset],_=pearsonr(obs[~common], temp[~common])
    return cor


# In[18]:


def cor_plot(lv1_data,ref,res='daily',folder=None):
    num=2
    sb.set_theme(style='darkgrid',palette='deep')
    sb.set_context("paper",font_scale =2)
    col = {
            'PCC-B1': sb.color_palette()[0], # Dark Blue
            'PCC-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
            'P-CDR': sb.color_palette()[3],
        }
    _max=0
    string=' (mm)'
    fig,axes,pl=plt.figure(figsize=(5*len(lv1_data.keys()),10)),[],0
    if res=='cdr':
        num=3
        fig=plt.figure(figsize=(5*len(lv1_data.keys()),15))
    if res=='3hr':
        fig=plt.figure(figsize=(5*len(lv1_data.keys()),15))
        string=' (mm)'
    for day,i in zip(lv1_data.keys(),range(len(lv1_data.keys()))):
        obs=lv1_data[day][ref].copy().flatten()
        obs_mask=np.isnan(obs)
        for dataset in [f for f in lv1_data[day].keys() if f!=ref]:
#             print(day,dataset,i,pl%2)
            temp=lv1_data[day][dataset].copy().flatten()
            mask=np.isnan(temp)
            common=obs_mask | np.isnan(temp)
            axes.append(plt.subplot2grid((num,len(lv1_data.keys())),(pl%num,i)))
            _max=max(np.nanmax(np.concatenate((temp, obs))),_max)
            axes[-1].scatter(temp[~common],obs[~common], color=col[dataset],alpha=.1)
#             axes[-1].set_title(f'{ref} vs. {dataset}')
            axes[-1].set_xlabel(dataset+string)
            axes[-1].set_ylabel(ref+string) 
            if i!=0:
                axes[-1].set_ylabel("")
            if dataset == 'PCC-CPC':
                label=day
                label=label.strftime('%b-%d')
                axes[-1].set_title(label,fontsize=25)
            pl+=1
    for ax in axes:
        ax.plot(np.linspace(0,_max+10,100),np.linspace(0,_max,100),c='r')
        ax.set_xlim([0, _max+10])
        ax.set_ylim([0, _max+10])
        ax.set_aspect('equal')


# In[19]:


def cdf_plot(lv1_data,ref,th1,th2,th3,folder=None):
    sb.set_theme(style='darkgrid',palette='deep')
    col = {
            'PCC-B1': sb.color_palette()[0], # Dark Blue
            'PCC-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
            'P-CDR': sb.color_palette()[3],
        }
    fig,axes,pl=plt.figure(figsize=(30,30)),[],0
    for day,i in zip(lv1_data.keys(),range(len(lv1_data.keys()))):
        obs=lv1_data[day][ref].copy().flatten()
        for pl,th in zip(range(len([th1,th2,th3])),[th1,th2,th3]):
            axes.append(plt.subplot2grid((len(lv1_data.keys()),3),(i,pl)))
            obs_ecdf = sm.distributions.ECDF(obs[obs>=th])
            axes[-1].plot(obs_ecdf.x,obs_ecdf.y, color=col['STAGE IV'],label=ref)
            for dataset in [f for f in lv1_data[day].keys() if f!=ref]:
                temp=lv1_data[day][dataset].copy().flatten()
                temp_ecdf = sm.distributions.ECDF(temp[temp>=th])
                axes[-1].plot(temp_ecdf.x,temp_ecdf.y,label=dataset,color=col[dataset])
                if pl%len(lv1_data.keys())==0:
                    label=day
                    label=label.strftime('%b-%d')
                    plt.ylabel(f'CDF for {label}')
    plt.legend()
    for ax,t in zip(axes,[th1,th2,th3]):
        ax.set_title(f'thresh hold is: >={t} mm')
    for ax in axes[-3:]:
         ax.set_xlabel('Precipitaion')


# In[20]:


def pixel_count_plot(lv1_data,bins,folder=None):
    sb.set_theme(style='darkgrid',palette='deep')
    sb.set_context("paper",font_scale =2.5)
    col = {
            'PCC-B1': sb.color_palette()[0], # Dark Blue
            'PCC-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
            'P-CDR': sb.color_palette()[3],
        }
    BIG_data={}
    BIG_data=pd.DataFrame()
    fig,axes,pl=plt.figure(figsize=(30,30)),[],0
    for day,i in zip(lv1_data.keys(),range(len(lv1_data.keys()))):
        axes.append(plt.subplot2grid((len(lv1_data.keys()),3),(i,0)))
        axes.append(plt.subplot2grid((len(lv1_data.keys()),3),(i,1)))
        axes.append(plt.subplot2grid((len(lv1_data.keys()),3),(i,2)))
        for dataset in [f for f in lv1_data[day].keys()]:
            BIG_data[dataset],_=np.histogram(lv1_data[day][dataset],bins=bins)
            BIG_data['bins']=bins[:-1]
            BIGGER_data = pd.melt(BIG_data, id_vars=["bins"], var_name="Data Base", value_name="Pixel Count")
        sb.barplot(x='bins',y='Pixel Count',hue="Data Base",data=BIGGER_data[BIGGER_data['bins']<.1],                   ax=axes[-3],palette=col)
        if pl%3==0:
            label=day
            label=label.strftime('%b-%d')
            axes[-3].set_ylabel(f'Pixel Count for {label}',fontsize=25)
        sb.barplot(x='bins',y='Pixel Count',hue="Data Base",                   data=BIGGER_data[(BIGGER_data['bins']>0) & (BIGGER_data['bins']<bins[8])],ax=axes[-2],palette=col)
        sb.barplot(x='bins',y='Pixel Count',hue="Data Base",                   data=BIGGER_data[BIGGER_data['bins']>bins[7]],ax=axes[-1],palette=col)
    for ax,num in zip(axes,range(len(axes))):
        if num!=1:
            ax.get_legend().remove()
        ax.set_xlabel('Precipitation (mm)',fontsize=25)
        ax.set_xticklabels(ax.get_xticklabels(),fontsize=20)
#         ax.set_yticklabels(ax.get_yticklabels(),fontsize=20)
        if num%3!=0:
            ax.set_ylabel('')
    for ax in axes[:-3]:
        ax.set_xlabel('')
    sb.reset_defaults()


# In[21]:


def percentile_plot(percent_val,res='day',folder=None):
    sb.set_context("paper",font_scale =3)
    sb.set_theme(style='darkgrid',palette='deep')
    col = {
            'PCC-B1': sb.color_palette()[0], # Dark Blue
            'PCC-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
            'P-CDR': sb.color_palette()[3],
        }
    fig,axes,pl,_=plt.figure(figsize=(17,5)),[],0,True
    if res=='3hr':
        fig,axes,_=plt.figure(figsize=(17,6)),[],True
    axes.append(plt.subplot2grid((1,3),(0,0)))
    axes.append(plt.subplot2grid((1,3),(0,1)))
    axes.append(plt.subplot2grid((1,3),(0,2)))
    for per,i in zip([50,95,99],range(3)):
        cpc,b1,st4,days,cdr=[],[],[],[],[]
        for day in percent_val.keys():
            days.append(day)
            cpc.append(percent_val[day][per]['PCC-CPC'])
            b1.append(percent_val[day][per]['PCC-B1'])
            st4.append(percent_val[day][per]['STAGE IV'])
            try:
                cdr.append(percent_val[day][per]['P-CDR'])
            except:
                alaki=1
        axes[i].set_title(f'{per}$^{{th}}$ Percentile',fontsize=25)
        axes[i].plot(days,cpc,color=col['PCC-CPC'],label='PCC-CPC',marker='o')
        axes[i].plot(days,st4,color=col['STAGE IV'],label='STAGE IV',marker='o')
        axes[i].plot(days,b1,color=col['PCC-B1'],label='PCC-B1',marker='o')
        try:
            axes[i].plot(days,cdr,color=col['P-CDR'],label='P-CDR',marker='o')
        except:
            alaki=1
        axes[i].set_xticks(days)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        axes[i].set_ylabel('mm',fontsize=25)
        if res=='3hr':
            axes[i].set_ylabel('mm',fontsize=25)
            axes[i].set_xticks(days[::5])
            axes[i].set_xticklabels(days[::5])
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%H'))
            axes[i].tick_params(axis='x', rotation=90)
        axes[i].tick_params(axis='both', labelsize=20)
        
    handles1, labels1 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles1,labels=labels1)  


# In[22]:


def metrics2(total,res='daily',folder=None):
    sb.set_context("paper",font_scale =3)
    sb.set_theme(style='darkgrid',palette='deep')
    col = {
            'PCC-B1': sb.color_palette()[0], # Dark Blue
            'PCC-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
            'P-CDR': sb.color_palette()[3],
        }
    fig,axes=plt.figure(figsize=(20,5)),[]
    axes.append(plt.subplot2grid((1,2),(0,0)))
    axes.append(plt.subplot2grid((1,2),(0,1)))
    st4,cpc,b1,days,cdr=[],[],[],[],[]
    for day in total.keys():
        days.append(day)
        st4.append(total[day]['STAGE IV'])
        cpc.append(total[day]['PCC-CPC'])
        b1.append(total[day]['PCC-B1'])
        try:
            cdr.append(total[day]['P-CDR'])
        except:
            alaki=0
    st4,cpc,b1=np.cumsum(st4),np.cumsum(cpc),np.cumsum(b1)
    for y,label in zip([st4,cpc,b1],["STAGE IV","PCC-CPC","PCC-B1"]):
            axes[0].plot(days,y,label=label,color=col[label],marker='o')
    if len(cdr)>0:
        cdr= np.cumsum(cdr)
        axes[0].plot(days,cdr,label='P-CDR',color=col['P-CDR'],marker='o')
    axes[0].legend()
    axes[0].set_xticks(days)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    if res=='3hr':
        axes[0].set_xticks(days[::5])
        axes[0].set_xticklabels(days[::5])
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%H'))
    ##########
    st4,cpc,b1,days,cdr=[],[],[],[],[]
    axes[0].set_title('Accumulated Precipitation',fontsize=25)
    axes[0].set_ylabel('mm',fontsize=20)
    for day in total.keys():
        days.append(day)
        st4.append(total[day]['STAGE IV'])
        cpc.append(total[day]['PCC-CPC'])
        b1.append(total[day]['PCC-B1'])
        try:
            cdr.append(total[day]['P-CDR'])
        except:
            alaki=0
#     st4,cpc,b1=np.cumsum(st4),np.cumsum(cpc),np.cumsum(b1)
    for y,label in zip([st4,cpc,b1],["STAGE IV","PCC-CPC","PCC-B1"]):
            axes[1].plot(days,y,label=label,color=col[label],marker='o')
    if len(cdr)>0:
        cdr= np.cumsum(cdr)
        axes[1].plot(days,cdr,label='P-CDR',color=col['P-CDR'],marker='o')
    axes[1].legend()
    axes[1].set_xticks(days)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    if res=='3hr':
        axes[1].set_xticks(days[::5])
        axes[1].set_xticklabels(days[::5])
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%H'))
    axes[0].tick_params(axis='both', labelsize=20)
    axes[1].tick_params(axis='both', labelsize=20)


# In[23]:


def metrics1(rmse,cor_value,mssim,res='daily',folder=None):
    mssim=rmse.copy()
    sb.set_context("paper",font_scale =3)
    sb.set_theme(style='darkgrid',palette='deep')
    varss={"RMSE":rmse,"C.C.":cor_value}#,"MSSIM":mssim}
    col = {
            'PCC-B1': sb.color_palette()[0], # Dark Blue
            'PCC-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
            'P-CDR': sb.color_palette()[3],
        }
    fig,axes=plt.figure(figsize=(25,5)),[]
    for var,pl in zip(varss.keys(),range(len(varss.keys()))):
        axes.append(plt.subplot2grid((1,3),(0,pl)))
        cpc,b1,days,cdr=[],[],[],[]
        for day in varss[var].keys():
            days.append(day)
            cpc.append(varss[var][day]['PCC-CPC'])
            b1.append(varss[var][day]['PCC-B1'])
            try:
                cdr.append(varss[var][day]['P-CDR'])
            except:
                alaki=0
        axes[-1].plot(days,cpc,marker='o',color=col['PCC-CPC'],label='PCC-CPC')
        axes[-1].plot(days,b1,marker='o',color=col['PCC-B1'],label='PCC-B1')
        if len(cdr)>0:
            axes[-1].plot(days,cdr,marker='o',color=col['P-CDR'],label='P-CDR')
        axes[-1].set_title(var,fontsize=25)
        axes[-1].set_xticks(days)
        if var=='RMSE':
            axes[-1].set_ylabel('mm',fontsize=20)
            if res=='3hr':
                axes[-1].set_ylabel('mm',fontsize=20)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        if res=='3hr':
            axes[-1].set_xticks(days[::5])
            axes[-1].set_xticklabels(days[::5])
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%H'))
        axes[-1].tick_params(axis='both', labelsize=20)
    axes[0].legend()   


# In[24]:


def skill_fig(skill,res='daily',folder=None):
#     sb.set_context("paper",font_scale =3)
    sb.set_theme(style='darkgrid',palette='deep')
    _=False
    col = {
            'PCC-B1': sb.color_palette()[0], # Dark Blue
            'PCC-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
            'P-CDR': sb.color_palette()[3],
        }
    fig,axes=plt.figure(figsize=(25,5)),[]
    if res=='3hr':
        fig,axes,_=plt.figure(figsize=(25,7)),[],True
    for var,i in zip(['POD','FAR','CSI','HSS'],range(4)):
        axes.append(plt.subplot2grid((1,4),(0,i)))
        cpc,b1,cdr,days=[],[],[],[]
        for day in skill.keys():
            days.append(day)
            cpc.append(skill[day]['PCC-CPC'][var])
            b1.append(skill[day]['PCC-B1'][var])
            try:
                cdr.append(skill[day]['P-CDR'][var])
            except:
                alaki=0
#                 print('no cdr')
            axes[i].plot(days,cpc,color=col['PCC-CPC'],label='PCC-CPC',marker='o')
            axes[i].plot(days,b1,color=col['PCC-B1'],label='PCC-B1',marker='o')
            if len(cdr)>0:
                axes[i].plot(days,cdr,color=col['P-CDR'],label='P-CDR',marker='o')
            axes[i].set_title(var.upper(),fontsize=25)
            axes[i].set_xticks(days)
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            if res=='3hr':
                axes[i].set_xticks(days[::5])
                axes[i].set_xticklabels(days[::5])
                axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%H'))
                axes[i].tick_params(axis='x', rotation=90)
            axes[i].tick_params(axis='both', labelsize=20)
    handles1, labels1 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles1[:len(handles1)//(len(days))], labels=labels1[:len(labels1)//(len(days))])  


# In[25]:


def cut_storm_box_25deg_daily(df_raw,day,rad_factor=1,chrs_path=chrs_path,conus_st4=conus_st4,conus_per=conus_per,
                             folder=None):
    df=df_raw[df_raw['date'].dt.day==day]
    df = df.reset_index(drop=True)
    df=df.fillna(0,)
    date=df['date'][0]
    date=date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    radius=max(df['radius'])*rad_factor
    x_st4_min,x_st4_max=max(0,int(np.floor(min(df["x_st4"])-radius/.04))),int(np.ceil(max(df["x_st4"])+radius/.04))
    y_st4_min,y_st4_max=max(0,int(np.floor(min(df["y_st4"])-radius/.04))),int(np.ceil(max(df["y_st4"])+radius/.04))
    x_per_min,x_per_max=int(np.floor(min(df["x_per"])-radius/.04)),int(np.ceil(max(df["x_per"])+radius/.04))
    y_per_min,y_per_max=int(np.floor(min(df["y_per"])-radius/.04)),int(np.ceil(max(df["y_per"])+radius/.04))
#     print(x_st4_min,x_st4_max,y_st4_min,y_st4_max,x_per_min,x_per_max,y_per_min,y_per_max)
    st4=mine.load_gz(chrs_path['st4_daily'][date],feed_back=False)
    st4[~conus_st4]=np.nan
    cpc=mine.load_gz(chrs_path['ccs_cdr_cpc_daily'][date],feed_back=False)
    cpc[~conus_per]=np.nan
    b1=mine.load_gz(chrs_path['ccs_cdr_b1_daily'][date],feed_back=False)
    b1[~conus_per]=np.nan
    st4=st4[y_st4_min:y_st4_max,x_st4_min:x_st4_max]
    cpc=cpc[y_per_min:y_per_max,x_per_min:x_per_max]
    b1=b1[y_per_min:y_per_max,x_per_min:x_per_max]  
    if not (np.isnan(st4).all() and np.isnan(cpc).all() and np.isnan(b1).all()):
        b1,cpc,st4=cutting_map(b1),cutting_map(cpc),cutting_map(st4)
        if b1.shape!=st4.shape:
            b1=b1[:st4.shape[0],:st4.shape[1]]
        if cpc.shape!=st4.shape:
            cpc=cpc[:st4.shape[0],:st4.shape[1]]
        if cpc.shape!=st4.shape:
            st4=st4[:cpc.shape[0],:cpc.shape[1]]
        _dict={'STAGE IV':st4,'PCC-CPC':cpc,"PCC-B1":b1,'date':date}
        return _dict
    else:
        print('empty array')


# In[27]:


def ASLI(storm,year,force_date=False):
    lv1_data={}
    folder_n=storm+'_daily_0.25'
    os.makedirs(os.path.join("/nfs/chrs-data3/shared/Bol/Analysis/climate/Figs",folder_n),exist_ok=True)
    storm_df=storm_info_04deg(storm,year,land=False)
    date=storm_df['date'].dt.day.unique()
    if force_date:
        date=date[date>=force_date.day]
    for day in date:
        temp_dict=cut_storm_box_25deg_daily(storm_df,day,rad_factor=4)
        if temp_dict is not None:
            lv1_data[temp_dict['date']]={}
            lv1_data[temp_dict['date']].update({'STAGE IV':temp_dict['STAGE IV'],"PCC-CPC":temp_dict['PCC-CPC'],                                               "PCC-B1":temp_dict['PCC-B1']})
    rmse=rmse_fun(lv1_data,'STAGE IV')
    skill=skill_score_fun(lv1_data,'STAGE IV',0.1)
    cor_value=cor_fun(lv1_data,'STAGE IV')
    total=total_precipitation(lv1_data)
    percent_val=percent(lv1_data)
    aa=mine.c_map_maker(plt.get_cmap("tab20b"))
    snapshot=images(lv1_data,_cmap=aa,df=storm_df,rad_factor=4,folder=folder_n)
    cor_img=cor_plot(lv1_data,'STAGE IV',folder=folder_n)
    pixel_img=pixel_count_plot(lv1_data,bins=[0,10,20,30,40,50,60,70,80,90,100,120,150,200],folder=folder_n)
    metrics1_fig=metrics1(rmse,cor_value,0,folder=folder_n)
    total_fig=metrics2(total,folder=folder_n)
    skill_figure=skill_fig(skill,folder=folder_n)
    percent_figure=percentile_plot(percent_val,folder=folder_n)
    return lv1_data


# ### results

# In[38]:


data=ASLI('Michael',2018,force_date=datetime(2018,10,9))


# 
# ## .025hourly analysis

# ### functions

# In[42]:


def cut_storm_box_025deg_3hr(df_raw,day,rad_factor=1,chrs_path=chrs_path,conus_st4=conus_st4,conus_per=conus_per):
    df=df_raw[df_raw['date']==day]
    df = df.reset_index(drop=True)
    df=df.fillna(0,)
    date=day
    radius=max(df['radius'])*rad_factor
    x_st4_min,x_st4_max=int(np.floor(min(df["x_st4"])-radius/.04)),int(np.ceil(max(df["x_st4"])+radius/.04))
    y_st4_min,y_st4_max=int(np.floor(min(df["y_st4"])-radius/.04)),int(np.ceil(max(df["y_st4"])+radius/.04))
    x_per_min,x_per_max=int(np.floor(min(df["x_per"])-radius/.04)),int(np.ceil(max(df["x_per"])+radius/.04))
    y_per_min,y_per_max=int(np.floor(min(df["y_per"])-radius/.04)),int(np.ceil(max(df["y_per"])+radius/.04))
    try:
        st4=mine.load_gz(chrs_path['st4_3hr'][date],feed_back=False)
        st4[~conus_st4]=np.nan
        cpc=mine.load_gz(chrs_path['ccs_cdr_cpc_3hr'][date],feed_back=False)/100*3
        cpc[~conus_per]=np.nan
        b1=mine.load_gz(chrs_path['ccs_cdr_b1_3hr'][date],feed_back=False)/100*3
        b1[~conus_per]=np.nan
        st4=st4[y_st4_min:y_st4_max,x_st4_min:x_st4_max]
        cpc=cpc[y_per_min:y_per_max,x_per_min:x_per_max]
        b1=b1[y_per_min:y_per_max,x_per_min:x_per_max]  
        if not (np.isnan(st4).all() and np.isnan(cpc).all() and np.isnan(b1).all()):
            b1,cpc,st4=cutting_map(b1),cutting_map(cpc),cutting_map(st4)
            if b1.shape!=st4.shape:
                b1=b1[:st4.shape[0],:st4.shape[1]]
            if cpc.shape!=st4.shape:
                cpc=cpc[:st4.shape[0],:st4.shape[1]]
            if cpc.shape!=st4.shape:
                st4=st4[:cpc.shape[0],:cpc.shape[1]]
            if not (np.isnan(st4).all() and np.isnan(cpc).all() and np.isnan(b1).all()):
                _dict={'STAGE IV':st4,'PCC-CPC':cpc,"PCC-B1":b1,'date':date}
                return _dict
        else:
            print('empty array')
    except:
        print(f'{date} not available')


# In[45]:


def ASLI_025_3hourly(storm,year,land=True,force_date=False):
    folder_n=storm+'_3hr_0.25'
    os.makedirs(os.path.join("/nfs/chrs-data3/shared/Bol/Analysis/climate/Figs",folder_n),exist_ok=True)
    lv1_data={}
    storm_df=storm_info_04deg(storm,year,land=land)
    date=storm_df['date']#.dt.day.unique()
    if force_date:
        date=date[date>=force_date]
    for day in date:
        temp_dict=cut_storm_box_025deg_3hr(storm_df,day,rad_factor=4)
        if temp_dict is not None:
            lv1_data[temp_dict['date']]={}
            lv1_data[temp_dict['date']].update({'STAGE IV':temp_dict['STAGE IV'],"PCC-CPC":temp_dict['PCC-CPC'],                                               "PCC-B1":temp_dict['PCC-B1']})
    rmse=rmse_fun(lv1_data,'STAGE IV')
    skill=skill_score_fun(lv1_data,'STAGE IV',0.1)
    cor_value=cor_fun(lv1_data,'STAGE IV')
    total=total_precipitation(lv1_data)
    percent_val=percent(lv1_data)
    aa=mine.c_map_maker(plt.get_cmap("tab20b"))
    snapshot=images(lv1_data,_cmap=aa,res='3hr',df=storm_df,folder=folder_n)
    metrics1_fig=metrics1(rmse,cor_value,0,res='3hr',folder=folder_n)
    total_fig=metrics2(total,res='3hr',folder=folder_n)
    skill_figure=skill_fig(skill,res='3hr',folder=folder_n)
    percent_figure=percentile_plot(percent_val,res='3hr',folder=folder_n)


# ### results

# In[46]:


data2=ASLI_025_3hourly('Michael',2018,force_date=datetime(2018,10,9))


# ## 04 daily analysis

# ### functions

# In[28]:


def cut_storm_box_04deg_daily(df_raw,day,rad_factor=1,zoom_factor=1/6.25,chrs_path=chrs_path,conus_st4=conus_st4,                              conus_per=conus_per,conus_cdr=conus_cdr):
    df=df_raw[df_raw['date'].dt.day==day]
    df = df.reset_index(drop=True)
    df=df.fillna(0,)
    date=df['date'][0]
    date=date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    radius=max(df['radius'])*rad_factor
    x_st4_min,x_st4_max=max(0,int(np.floor(min(df["x_st4"])-radius/.04))),int(np.ceil(max(df["x_st4"])+radius/.04))
    y_st4_min,y_st4_max=max(0,int(np.floor(min(df["y_st4"])-radius/.04))),int(np.ceil(max(df["y_st4"])+radius/.04))
    x_per_min,x_per_max=int(np.floor(min(df["x_per"])-radius/.04)),int(np.ceil(max(df["x_per"])+radius/.04))
    y_per_min,y_per_max=int(np.floor(min(df["y_per"])-radius/.04)),int(np.ceil(max(df["y_per"])+radius/.04))
    x_cdr_min,x_cdr_max=int(np.floor(min(df["x_cdr"])-radius/.25)),int(np.ceil(max(df["x_cdr"])+radius/.25))
    y_cdr_min,y_cdr_max=int(np.floor(min(df["y_cdr"])-radius/.25)),int(np.ceil(max(df["y_cdr"])+radius/.25))
    st4=mine.load_gz(chrs_path['st4_daily'][date],feed_back=False)
    st4[~conus_st4]=np.nan
    cpc=mine.load_gz(chrs_path['ccs_cdr_cpc_daily'][date],feed_back=False)
    cpc[~conus_per]=np.nan
    b1=mine.load_gz(chrs_path['ccs_cdr_b1_daily'][date],feed_back=False)
    b1[~conus_per]=np.nan
    cdr=mine.load_map(chrs_path['cdr_daily'][date],feed_back=False,dataset='cdr')
    cdr[~conus_cdr]=np.nan
    st4=st4[y_st4_min:y_st4_max,x_st4_min:x_st4_max]
    cpc=cpc[y_per_min:y_per_max,x_per_min:x_per_max]
    b1=b1[y_per_min:y_per_max,x_per_min:x_per_max]
    cdr=cdr[y_cdr_min:y_cdr_max,x_cdr_min:x_cdr_max]
    if not (np.isnan(st4).all() and np.isnan(cpc).all() and np.isnan(b1).all()):
        b1,cpc,st4,cdr=cutting_map(b1),cutting_map(cpc),cutting_map(st4),cutting_map(cdr)
        if zoom_factor!=1:
            cpc=zoom(cpc, zoom_factor, order=1)
            b1=zoom(b1, zoom_factor, order=1)
            st4=zoom(st4, zoom_factor, order=1)
        if b1.shape!=cdr.shape:
            b1=b1[:cdr.shape[0],:cdr.shape[1]]
        if cpc.shape!=cdr.shape:
            cpc=cpc[:cdr.shape[0],:cdr.shape[1]]
        if st4.shape!=cdr.shape:
            st4=st4[:cdr.shape[0],:cdr.shape[1]]
        if cpc.shape!=cdr.shape:
            cdr=cdr[:cpc.shape[0],:cpc.shape[1]]
        _dict={'STAGE IV':st4,'PCC-CPC':cpc,"PCC-B1":b1,"P-CDR":cdr,'date':date}
        return _dict
    else:
        print('empty array')


# In[29]:


def ASLI_04_daily(storm,year,force_date=False,land=True):
    folder_n=storm+'_daily_0.4'
    os.makedirs(os.path.join("/nfs/chrs-data3/shared/Bol/Analysis/climate/Figs",folder_n),exist_ok=True)
    lv1_data={}
    storm_df=storm_info_04deg(storm,year,land=False)
    date=storm_df['date'].dt.day.unique()
    if force_date:
        date=date[date>=force_date.day]
    for day in date:
        temp_dict=cut_storm_box_04deg_daily(storm_df,day,rad_factor=4)
        if temp_dict is not None:
            lv1_data[temp_dict['date']]={}
            lv1_data[temp_dict['date']].update({'STAGE IV':temp_dict['STAGE IV'],"PCC-CPC":temp_dict['PCC-CPC'],                                               "PCC-B1":temp_dict['PCC-B1'],"P-CDR":temp_dict['P-CDR']})
    rmse=rmse_fun(lv1_data,'STAGE IV')
    skill=skill_score_fun(lv1_data,'STAGE IV',0.1)
    cor_value=cor_fun(lv1_data,'STAGE IV')
    total=total_precipitation(lv1_data)
    percent_val=percent(lv1_data)
    aa=mine.c_map_maker(plt.get_cmap("tab20b"))
    snapshot=images(lv1_data,_cmap=aa,res='cdr',df=storm_df,folder=folder_n)
    cor_img=cor_plot(lv1_data,'STAGE IV',res='cdr',folder=folder_n)
    cdf_img=cdf_plot(lv1_data,'STAGE IV',.1,1,10,folder=folder_n)
    pixel_img=pixel_count_plot(lv1_data,bins=[0,10,20,30,40,50,60,70,80,90,100,120,150,200],folder=folder_n)
    metrics1_fig=metrics1(rmse,cor_value,0,folder=folder_n)
    total_fig=metrics2(total,folder=folder_n)
    skill_figure=skill_fig(skill,folder=folder_n)
    percent_figure=percentile_plot(percent_val,folder=folder_n)
    return lv1_data


# ### results

# In[30]:


data3=ASLI_04_daily('Michael',2018,force_date=datetime(2018,10,9))


# In[33]:


plt.close('all')
print('Done')


# # plot REGION OF STUDY

# In[3]:


gdf = LOAD_hybas_SHAPE_FILE
shp=gpd.read_file(path)
shp=shp[shp['HYBAS_ID']==7040569650] ## for upper missi


# In[ ]:


gdf = gdf.to_crs(epsg=2163)
shp = shp.to_crs(epsg=2163)


# In[2]:


fig, ax = plt.subplots(figsize=(20, 10))
sb.set_theme(style='white',palette='deep')
ax.set_facecolor('#767175')
gdf.plot(ax=ax,color="#918D8E",edgecolor='None', linewidth=1)
shp.plot(ax=ax,color="#FFB4B5",alpha=.5,edgecolor='white', linewidth=1)
marker={"Michael":'o'}
for name,yr in zip(['Michael'],[2018]):
    df=storm_info_04deg(name,yr,land=False)
    gdf2 = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
    crs="EPSG:4326")
    gdf2 = gdf2.to_crs(epsg=2163)
    gdf2['Lat']=gdf2.geometry.y
    gdf2['Lon']=gdf2.geometry.x
    sc=sb.scatterplot(data=gdf2,x='Lon',y='Lat',hue='hours_since_start',legend=False,                      palette='magma_r',marker=marker[name],ax=ax,s=100,)

norm = plt.Normalize(65,140)
sm1 = plt.cm.ScalarMappable(cmap='magma_r', norm=norm)
sm1.set_array([])
cbar=plt.colorbar(sm1, label='Hours Since Hurricane Genesis',fraction=.02,ax=ax)
cbar.set_label('Hours Since Hurricane Genesis', fontsize=20)
handles = [mlines.Line2D([], [], color='purple', marker=marker[name], linestyle='None', markersize=15, label=f'Hurricane {name}')
           for name in marker]
hatched_rect = mpatches.Patch(edgecolor="white", facecolor='#FFB4B5', label='Upper Missisipi Basin',)

handles.append(hatched_rect)
ax.legend(handles=handles,loc='lower left',fontsize=15)
ax.set_xlim(-2.2e6, 2.6e6)
ax.set_ylim(-2.3e6, 1e6)
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])


# # Stuff

# In[ ]:


p1=Illinois_SHAPEFILE
p2=INDIANA_SHAPEFILE
p3=MICHIGAN_SHAPEFILE
p4=WISCONSIN_SHAPEFILE
mask_p=p1 | p2 | p3 | p4 
mask_p=np.roll(mask_p,int(mask_p.shape[1]/2))
plt.figure()
plt.imshow(mask_p)

S1=Illinois_SHAPEFILE_ST4
S2=INDIANA_SHAPEFILE_ST4
S3=MICHIGAN_SHAPEFILE_ST4
S4=WISCONSIN_SHAPEFILE_ST4
mask_st4=s1 | s2 | s3 | s4
plt.figure()
plt.imshow(mask_st4)

del p1,p2,p3,p4,s1,s2,s3,s4


# In[ ]:


st4_3hr=st4_3hr_DICT


# In[ ]:


def cutting(arr,mask):
    arr[~mask]=np.nan
    non_blank_indices = np.where(mask)
    try:
        min_x, min_y = np.min(non_blank_indices, axis=1)
        max_x, max_y = np.max(non_blank_indices, axis=1)
    except ValueError:
        min_x, min_y =42 ,221 
        max_x, max_y= 138, 451
    finally:
        return arr[min_x:max_x, min_y:max_y]


# In[ ]:


def cut_region_3hr(date,chrs_path=chrs_path,mask_st4=mask_st4,mask_per=mask_p,st4_3hr=st4_3hr):
#     print(st4_3hr[date])
    st4=mine.load_gz(st4_3hr[date],feed_back=False)
    st4=cutting(st4,mask_st4)
    st4=st4*3
    cpc=mine.load_gz(chrs_path['ccs_cdr_cpc_3hr'][date],feed_back=False)/100*3
    cpc=cutting(cpc,mask_per)
    b1=mine.load_gz(chrs_path['ccs_cdr_b1_3hr'][date],feed_back=False)/100*3
    b1=cutting(b1,mask_per)
#     print(st4.shape,b1.shape,cpc.shape)
#     if not (np.isnan(st4).all() and np.isnan(cpc).all() and np.isnan(b1).all()):
#         b1,cpc,st4=cutting_map(b1),cutting_map(cpc),cutting_map(st4)
#         if b1.shape!=st4.shape:
#             b1=b1[:st4.shape[0],:st4.shape[1]]
#         if cpc.shape!=st4.shape:
#             cpc=cpc[:st4.shape[0],:st4.shape[1]]
#         if cpc.shape!=st4.shape:
#             st4=st4[:cpc.shape[0],:cpc.shape[1]]
    _dict={'STAGE IV':st4,'PERSIANN-CCS-CDR-CPC':cpc,"PERSIANN-CCS-CDR-B1":b1,'date':date}
    return _dict
#     else:
#         print('empty array')
    


# In[ ]:


def metrics1(rmse,cor_value,mssim,res='daily',folder=None):
    mssim=rmse.copy()
    sb.set_context("paper",font_scale =3)
    sb.set_theme(style='darkgrid',palette='deep')
    varss={"RMSE":rmse,"C.C.":cor_value}#,"MSSIM":mssim}
    col = {
            'PERSIANN-CCS-CDR-B1': sb.color_palette()[0], # Dark Blue
            'PERSIANN-CCS-CDR-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
            'PERSIANN CDR': sb.color_palette()[3],
        }
    fig,axes=plt.figure(figsize=(10*1.5,5*1.5)),[]
    for var,pl in zip(varss.keys(),range(len(varss.keys()))):
        axes.append(plt.subplot2grid((1,2),(0,pl)))
        cpc,b1,days,cdr=[],[],[],[]
        for day in varss[var].keys():
            days.append(day)
            cpc.append(varss[var][day]['PERSIANN-CCS-CDR-CPC'])
            b1.append(varss[var][day]['PERSIANN-CCS-CDR-B1'])
            try:
                cdr.append(varss[var][day]['PERSIANN CDR'])
            except:
                alaki=0
        axes[-1].plot(days,cpc,marker='o',color=col['PERSIANN-CCS-CDR-CPC'],label='PERSIANN-CCS-CDR-CPC')
        axes[-1].plot(days,b1,marker='o',color=col['PERSIANN-CCS-CDR-B1'],label='PERSIANN-CCS-CDR-B1')
        if len(cdr)>0:
            axes[-1].plot(days,cdr,marker='o',color=col['PERSIANN CDR'],label='PERSIANN CDR')
        axes[-1].set_title(var,fontsize=25)
        axes[-1].set_xticks(days)
        if var=='RMSE':
            axes[-1].set_ylabel('mm',fontsize=20)
            if res=='3hr':
                axes[-1].set_ylabel('mm',fontsize=20)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        if res=='3hr':
            axes[-1].set_xticks(days[::5])
            axes[-1].set_xticklabels(days[::5])
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%H'))
            for label in axes[-1].get_xticklabels():
                label.set_rotation(90)
        axes[-1].tick_params(axis='both', labelsize=20)
    axes[0].legend()   
#     plt.show()
#     mine.save_fig(os.path.join(folder,"metrics1"))
    save_fig(os.path.join(folder,"metrics1"),tight_layout=True)
    return fig


# In[ ]:


def percentile_plot(percent_val,res='day',folder=None):
    sb.set_context("paper",font_scale =3)
    sb.set_theme(style='darkgrid',palette='deep')
    col = {
            'PERSIANN-CCS-CDR-B1': sb.color_palette()[0], # Dark Blue
            'PERSIANN-CCS-CDR-CPC': sb.color_palette()[1],
            'STAGE IV': sb.color_palette()[2],
            'PERSIANN CDR': sb.color_palette()[3],
        }
    fig,axes,pl,_=plt.figure(figsize=(17,5)),[],0,True
    if res=='3hr':
        fig,axes,pl,_=plt.figure(figsize=(10*1.5,5*1.5)),[],0,True
    axes.append(plt.subplot2grid((1,2),(0,0)))
    axes.append(plt.subplot2grid((1,2),(0,1)))
#     axes.append(plt.subplot2grid((1,3),(0,2)))
    for per,i in zip([95,99],range(3)):
        cpc,b1,st4,days,cdr=[],[],[],[],[]
        for day in percent_val.keys():
            days.append(day)
            cpc.append(percent_val[day][per]['PERSIANN-CCS-CDR-CPC'])
            b1.append(percent_val[day][per]['PERSIANN-CCS-CDR-B1'])
            st4.append(percent_val[day][per]['STAGE IV'])
            try:
                cdr.append(percent_val[day][per]['PERSIANN CDR'])
            except:
                alaki=1
        axes[i].set_title(f'{per}$^{{th}}$ Percentile',fontsize=25)
        axes[i].plot(days,cpc,color=col['PERSIANN-CCS-CDR-CPC'],label='PERSIANN-CCS-CDR-CPC',marker='o')
        axes[i].plot(days,st4,color=col['STAGE IV'],label='STAGE IV',marker='o')
        axes[i].plot(days,b1,color=col['PERSIANN-CCS-CDR-B1'],label='PERSIANN-CCS-CDR-B1',marker='o')
        try:
            axes[i].plot(days,cdr,color=col['PERSIANN CDR'],label='PERSIANN CDR',marker='o')
        except:
            alaki=1
        axes[i].set_xticks(days)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        axes[i].set_ylabel('mm',fontsize=25)
        if res=='3hr':
            axes[i].set_ylabel('mm',fontsize=25)
            axes[i].set_xticks(days[::5])
            axes[i].set_xticklabels(days[::5])
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%H'))
            axes[i].tick_params(axis='x', rotation=90)
        axes[i].tick_params(axis='both', labelsize=20)
        
    handles1, labels1 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles1,labels=labels1)  
#     plt.show()
    save_fig(os.path.join(folder,"percentile_plot"),tight_layout=_,folder=None)
#     return fig


# In[ ]:


def region_event_3hr(dates,name):
    lv1_data={}
    folder_n=name+'_daily_0.25'
    os.makedirs(os.path.join(PATH_TO_YOUR_FOLDER,folder_n),exist_ok=True)
    for day in dates:
            temp_dict=cut_region_3hr(day)
            if temp_dict is not None:
                lv1_data[temp_dict['date']]={}
                lv1_data[temp_dict['date']].update({'STAGE IV':temp_dict['STAGE IV'],"PERSIANN-CCS-CDR-CPC":temp_dict['PERSIANN-CCS-CDR-CPC'],                                                   "PERSIANN-CCS-CDR-B1":temp_dict['PERSIANN-CCS-CDR-B1']})
    rmse=rmse_fun(lv1_data,'STAGE IV')
    skill=skill_score_fun(lv1_data,'STAGE IV',0.1)
    cor_value=cor_fun(lv1_data,'STAGE IV')
    total=total_precipitation(lv1_data)
    percent_val=percent(lv1_data)
    images(lv1_data,res='3hr',folder=YOUR_FOLDER)
    metrics1_fig=metrics1(rmse,cor_value,0,res='3hr',folder=folder_n)
    total_fig=metrics2(total,res='3hr',folder=folder_n)
    skill_figure=skill_fig(skill,res='3hr',folder=folder_n)
    percent_figure=percentile_plot(percent_val,res='3hr',folder=folder_n)     
    return lv1_data


# In[ ]:


dates=[datetime(2024,7,day,hr) for day in range(13,17) for hr in range(0,24,3)]
dates=[date for date in dates if date<datetime(2024,7,16,21) and date>datetime(2024,7,13,14)]


# In[ ]:


data31=region_event_3hr(dates,'mid_3hr')


# In[ ]:


cc=mine.c_map_maker(plt.get_cmap('coolwarm'))
def images(lv1_data,lim=(0,50),_cmap=cc,res='day',rad_factor=1,give_me_cor=False,folder=None):
    num=3
    sb.set_theme(style='white',palette='pastel')
    fig=plt.figure(figsize=(6,2*len(lv1_data.keys())))
    if res=='3hr':
        fig=plt.figure(figsize=(6,2*len(lv1_data.keys())))
    pl=0
    axes=[]
    for day,i in zip(lv1_data.keys(),range(len(lv1_data.keys()))):
        for dataset in lv1_data[day].keys():
            axes.append(plt.subplot2grid((len(lv1_data.keys()),len(lv1_data[day].keys())),(i,pl%num)))  
            cax=axes[-1].imshow(lv1_data[day][dataset],cmap=_cmap,clim=lim)
            axes[-1].set_yticks([lv1_data[day][dataset].shape[0]/2])
            axes[-1].set_xticks([lv1_data[day][dataset].shape[1]/2])
            font_properties = {'fontweight': 'ultralight', 'fontstyle': 'italic'}
            if pl//len(lv1_data[day].keys())==0:
                plt.title(dataset,fontsize=8,weight="heavy")
            if dataset=='STAGE IV':
                label=day
                if res=='3hr':
                    label=label.strftime('%b-%d-%H')
                    plt.ylabel(label+" hr")
                else:
                    label=label.strftime('%b-%d')
                    plt.ylabel(label)
            pl+=1
    if not give_me_cor:
        for ax in axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(bottom=False, left=False)
    cbar = fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('mm')
    if res=='3hr':
        cbar.set_label('mm')
#     plt.show(fig)
    save_fig(os.path.join(folder,"snapshot_midwest"),folder=None,tight_layout=False)
#     return fig


# In[ ]:


images(data31,res='3hr',folder=YOUR_FOLDER)


# In[ ]:




