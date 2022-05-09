from configs.Config import Config
from configs.SDE import SDE
from imports import *
from dataTool import ChangeLabels

class Semantic3D(Config):
    pointComponents = 3
    featureComponents = 3 #rgb
    classCount = Label.Semantic3D.Count-1
    classNames = Label.Semantic3D.Names[1:]
    name = "Sem3D"
    Paths = Paths.Semantic3D
    storeFilesInRAM = True

    input_tile_count = 1
    validation_split = 0.2
    ValidationInterval = None
    TestInterval = None

    if(input_tile_count == 1):
        test_step = 6.0
    else:
        test_step = 22.0

    validClasses = [1, 2, 3, 4, 5, 6, 7, 8]  
      
    ValidationInterval = 1 # validate model after each N epochs
    TestInterval  = None # test model after each N epochs
    # ValidateOnOtherData = "SDE"

    testFiles = [
                # "bildstein_station1_xyz_intensity_rgb_voxels",
                # "domfountain_station1_xyz_intensity_rgb_voxels.npy",
                ]

    fileNames = {"birdfountain_station1_xyz_intensity_rgb" : "birdfountain1",
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
                "stgallencathedral_station6_intensity_rgb" : "stgallencathedral6",

                # "MarketplaceFeldkirch_Station4_rgb_intensity-reduced" : "marketsquarefeldkirch4-reduced",
                # "sg27_station10_rgb_intensity-reduced" : "sg27_10-reduced",
                # "sg28_Station2_rgb_intensity-reduced" : "sg28_2-reduced",
                # "StGallenCathedral_station6_rgb_intensity-reduced" : "stgallencathedral6-reduced",
                }

    def MapLabels(self, labels, label_type):
        
        if(label_type == "SDE"):
            labels = ChangeLabels(labels, {
                0 : 0, # "Never classified" -> Unclassified
                1 : 2, # "man-made terrain" -> "Ground"
                2 : 2, # "natural terrain" -> "Ground"
                3 : 5, # "high vegetation" -> "High vegetation"
                4 : 5, # "low vegetation" -> "High vegetation"
                5 : 6, # "buildings" -> "Building"
                6 : 0, #  "hard scape" -> "Never classified"
                7 : 0, #  "scanning artefacts" -> "Never classified"
                8 : 70, #  "cars" -> "Cars"
            })            
            labels = SDE.ChangeLabels(labels)
        elif(label_type == "Sem3D"):
            labels -= 1
        else:
            assert("Unsupported label type")
        
        return labels
    
    class_color = np.array([
            [220,220,220], #light gray
            [124,252,0], #lawngreen     
            [34,139,34], #forestgreen          
            [173,255,47], # greenyellow       
            [105,105,105], # dimgrey 
            [70,130,180], # steelblue
            [255,0,0], #red
            [255,215,0], # gold
        ]) / 255
    
    def GetClassColors(self):
        return self.class_color