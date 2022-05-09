import pandas as pd
import numpy as np
import os
from dataTool import DataTool, SaveToLas

fileNames = {   "birdfountain_station1_xyz_intensity_rgb" : "birdfountain1",
                "castleblatten_station1_intensity_rgb" : "castleblatten1",
                "castleblatten_station5_xyz_intensity_rgb" : "castleblatten5",
                "marketplacefeldkirch_station1_intensity_rgb" : "marketsquarefeldkirch1",
                "marketplacefeldkirch_station4_intensity_rgb" : "marketsquarefeldkirch4",
                "marketplacefeldkirch_station7_intensity_rgb" : "marketsquarefeldkirch7",
                "sg27_station3_intensity_rgb" : "sg27_3",
                "sg27_station6_intensity_rgb" : "sg27_6",
                "sg27_station8_intensity_rgb" : "sg27_8",
                "sg27_station10_intensity_rgb" : "sg27_10",
                "sg28_station2_intensity_rgb" : "sg28_2",
                "sg28_station5_xyz_intensity_rgb" : "sg28_5",
                "stgallencathedral_station1_intensity_rgb" : "stgallencathedral1",
                "stgallencathedral_station3_intensity_rgb" : "stgallencathedral3",
                "stgallencathedral_station6_intensity_rgb" : "stgallencathedral6",}

path = "D:/semantic3d/las"
files = os.listdir(path)
# files = ["bildstein_station1_xyz_intensity_rgb.txt"]

for file in files:
    if file.endswith(".labels"):
        file = file.replace(".labels","")
        
        txtfile = os.path.join(path,file+".txt")
        lblfile = os.path.join(path,file+".labels")
        lasfile = os.path.join(path,file+".las")
        
        if(os.path.exists(lasfile)):
            continue
        
        pointcloud = np.array(pd.read_csv(txtfile, sep=" ", dtype=np.float64, header=None), dtype=np.float64)        
        # DataTool().VisualizePointCloud([pointcloud[:,:3]])
        
        labels = np.array(pd.read_csv(lblfile, header=None))

        SaveToLas(lasfile, pointcloud[:,:3], pointcloud[:,3], pointcloud[:,4:6], labels)