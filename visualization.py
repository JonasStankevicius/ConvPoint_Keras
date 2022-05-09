import open3d as o3d
from dataTool import LoadRenderOptions, SaveRenderOptions, GetPointsIndexInBoundingBox, GetPointsInBoundingBox
from imports import *

class BoxesIterator:
    def __init__(self, boxes, points, colors, labels):
        # self.pc = o3d.geometry.PointCloud()
        # self.pc.points = o3d.utility.Vector3dVector(points)
        self.src_points = points
        self.src_colors = colors if np.max(colors) <= 1 else colors/255
        self.src_labels = labels
        self.dst_points = np.zeros((0, 3), dtype = np.float)
        self.dst_colors = np.zeros((0, 3), dtype = np.float)
        self.boxes = boxes
        self.i = 0
        # self.kdt = KDTree(points, leaf_size=20)    

        self.trajectory = None
        # if(os.path.exists("./data/camera_trajectory.json")):
        #     self.trajectory = o3d.io.read_pinhole_camera_trajectory("./data/camera_trajectory.json").parameters
        #     self.trajectory_i = 0
        #     self.trajectory_time = time.time()

        grey = np.array([128, 128, 128])/255
        red = np.array([136, 0, 1])/255
        mint = np.array([170, 255, 195])/255
        teal = np.array([0, 128, 128])/255
        green = np.array([60, 180, 75])/255
        verygreen = np.array([0, 255, 0])/255
        brown = np.array([170, 110, 40])/255
        # white = np.array([255, 255, 255])/255
        black = np.array([0, 0, 0])/255
        blue = np.array([0, 0, 255])/255    
        pink = np.array([255, 56, 152])/255    

        #NPM3D
        self.colors = []
        if(np.max(self.src_labels) == 9):
            self.colors = [grey, red, blue, teal, mint, brown, pink, black, green]
        #Semantic3D
        elif(np.max(self.src_labels) == 8):
            self.colors = [grey, verygreen, green, mint, red, blue, brown, black]
        
        self.pc = o3d.geometry.PointCloud()        
        self.pc.points = o3d.utility.Vector3dVector(self.src_points)

        self.box = o3d.geometry.LineSet()
        lines = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],[0, 4], [1, 5], [2, 6], [3, 7]])
        self.box.lines = o3d.utility.Vector2iVector(lines)
        self.box.colors = o3d.utility.Vector3dVector(np.array([[1,0,0] for _ in range(len(lines))]))

        self.initSet = False

    def ColorPtsByClass(self, pts, lbl):
        pts_colors = np.zeros((len(pts), 3), np.float)

        for i in range(0, len(self.colors)):
            indexes = np.where(lbl == i+1)[0]
            pts_colors[indexes] = self.colors[i]

        return pts_colors
    
    def BoxPts(self, bBox):
        box =  [[bBox[0], bBox[2], bBox[4]], 
                [bBox[1], bBox[2], bBox[4]], 
                [bBox[0], bBox[3], bBox[4]], 
                [bBox[1], bBox[3], bBox[4]],
                [bBox[0], bBox[2], bBox[5]], 
                [bBox[1], bBox[2], bBox[5]], 
                [bBox[0], bBox[3], bBox[5]], 
                [bBox[1], bBox[3], bBox[5]]]
        return np.array(box)

    def AnimationFunction(self, vis):
        # time.sleep(0.2)
        if(self.i < len(self.boxes)):        
            pts = self.src_points[:, :2]
            mask_x = np.logical_and(self.boxes[self.i][0]<pts[:,0], pts[:,0]<self.boxes[self.i][1])
            mask_y = np.logical_and(self.boxes[self.i][2]<pts[:,1], pts[:,1]<self.boxes[self.i][3])
            ptsIdx = np.where(np.logical_and(mask_x, mask_y))[0]
            randIdx = np.random.choice(ptsIdx, min(8192, len(ptsIdx)), replace=False)
    
            self.dst_points = np.concatenate((self.dst_points, self.src_points[randIdx]), axis = 0)
            self.dst_colors = np.concatenate((self.dst_colors, self.ColorPtsByClass(self.src_points[randIdx], self.src_labels[randIdx])), axis = 0)

            self.src_points = np.delete(self.src_points, randIdx, axis = 0)
            self.src_labels = np.delete(self.src_labels, randIdx, axis = 0)
            self.src_colors = np.delete(self.src_colors, randIdx, axis = 0)
            
            self.pc.points = o3d.utility.Vector3dVector(np.concatenate((self.src_points, self.dst_points), 0))
            self.pc.colors = o3d.utility.Vector3dVector(np.concatenate((self.src_colors, self.dst_colors), 0))

            self.box.points = o3d.utility.Vector3dVector(self.BoxPts(self.boxes[self.i]))

            vis.clear_geometries()
            vis.add_geometry(self.pc, False)
            vis.add_geometry(self.box, False)
                    
            self.i += 1 
            # print(f"{self.i}/{len(self.boxes)}", end="\r")
        else:
            print("Iteration over.")

        if(not os.path.exists("./data/camera_trajectory.json")):
            self.trajectory = None

        if(self.trajectory is None):
            # vis = LoadRenderOptions(vis, returnVis=True)
            if(os.path.exists("./data/camera_trajectory.json")):
                self.trajectory = o3d.io.read_pinhole_camera_trajectory("./data/camera_trajectory.json").parameters
                self.trajectory_i = 0
                self.trajectory_time = time.time()                        
        else:
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(self.trajectory[self.trajectory_i])
            if(self.trajectory_i < len(self.trajectory)-1): #and time.time() - self.trajectory_time > 1
                print(f"Trajectory: {self.trajectory_i}/{len(self.trajectory)}", end="\r")
                self.trajectory_i += 1
                self.trajectory_time = time.time()

        return False

def ShowSequenceBoxes(ptsFile, lblFile, consts):
    from dataTool import DataTool

    consts.test_step = 4
    seq = TestSequence(ptsFile, consts, windowsMachineCap=False)

    minZ = np.min(seq.xyzrgb[:,2])
    maxZ = np.max(seq.xyzrgb[:,2])

    boxes = []
    for pt in seq.pts:
        minX = pt[0] - consts.blocksize/2
        maxX = pt[0] + consts.blocksize/2
        
        minY = pt[1] - consts.blocksize/2
        maxY = pt[1] + consts.blocksize/2

        boxes.append([minX, maxX, minY, maxY, minZ, maxZ])

    dt = DataTool()
    # dt.VisualizePointCloud([seq.xyzrgb[:,:3]], [seq.xyzrgb[:,3:6]], bBoxes = boxes)
    boxesitr = BoxesIterator(boxes, seq.xyzrgb[:,:3], seq.xyzrgb[:,3:], np.squeeze(ReadLabels(lblFile),1))
    dt.VisualizePointCloud([seq.xyzrgb[:,:3]], animationFunction=boxesitr.AnimationFunction)
    # dt.VisualizePointCloud([seq.xyzrgb[:,:3]])