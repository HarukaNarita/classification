import sys
import itertools
import functools
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import numpy as np
from scipy import signal
from skimage import io
from skimage.feature import blob_log
from skimage.restoration import rolling_ball
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from PyQt5 import QtWidgets, QtCore

class MyQt(QtWidgets.QWidget):
    def __init__(self, roi_max, fig, my_funcs, parent=None):
        super(MyQt, self).__init__(parent)
        self.roi_max = roi_max
        self.fig = fig
        self.my_funcs = my_funcs
        self.layout = QtWidgets.QVBoxLayout()
        self.sp = QtWidgets.QSpinBox()
        self.sp.setMinimum(0)
        self.sp.setMaximum(self.roi_max)
        self.sp.setGeometry(QtCore.QRect(10, 10, 211, 291))
        self.FigureCanvas = FigureCanvas(self.fig) 
        self.ax1 = self.fig.add_subplot(4,2,1)
        self.ax2 = self.fig.add_subplot(4,2,2)
        self.ax3 = self.fig.add_subplot(4,2,3)
        self.ax4 = self.fig.add_subplot(4,2,4)
        self.ax5 = self.fig.add_subplot(2,1,2)

        self.layout.addWidget(self.sp)
        self.layout.addWidget(self.FigureCanvas)
        self.setLayout(self.layout)
        
        self.sp.valueChanged.connect(self.show_figure)
    
    def show_figure(self,e=None):
        roi_number = self.sp.value()
        self.update_Figure(roi_number)
        
    def update_Figure(self,r):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()
        
        i1 = self.my_funcs[0](r)[0]  # skelton
        i2 = self.my_funcs[0](r)[1]  # raw data
        i3 = self.my_funcs[2](r)     # skelton map
        plot = self.my_funcs[1](r,self.ax2)      # plot along
        peaks = self.my_funcs[3](r,self.ax2)  # peaks
        PCA = self.my_funcs[4](r,self.ax5)
        
        self.ax1.imshow(i1)             # skelton
        self.ax3.imshow(i3,"plasma")    # skelton_map
        self.ax4.imshow(i2)             # raw data
        # self.ax2.plot(plot)             # plot along
        # self.ax2.scatter(peaks[0],peaks[1],c="r")
        self.ax2.set(ylim=(0))
        self.fig.canvas.draw()


# def interact_(**kwargs):
#     def wrapper(func):
#         def foo():
#         return interact(func)
#     return wrapper

class PolymorphClassifiy:
    def __init__(self,roi_path,image_path):
        self.I = io.imread(image_path)
        self.rois_xs = self.get_rois(roi_path)[0]
        self.rois_ys = self.get_rois(roi_path)[1]
        self.rois_z = self.get_rois(roi_path)[2]
        self._roi_number = len(self.rois_z)
        self.rois_min_x = self.get_min_max_x()[0]
        self.rois_max_x = self.get_min_max_x()[1]
        self.rois_min_y = self.get_min_max_y()[0]
        self.rois_max_y = self.get_min_max_y()[1]
        self.radius = 0
        self.pca_data = None
        self.feature = None
        self.pca = self.analyze_pca()
        
    def get_rois(self,roi_path_)->list:
        with open(roi_path_, encoding='utf8', newline='') as f:
            csvreader = csv.reader(f)
            all_line = [row for row in csvreader]     
        rois_xs = [np.array([int(x) for x in xs]) for xs in all_line[0::3]]
        rois_xs = [roi_xs[np.nonzero(roi_xs)] for roi_xs in rois_xs]
        rois_ys = [np.array([int(y) for y in ys]) for ys in all_line[1::3]]
        rois_ys = [roi_ys[np.nonzero(roi_ys)] for roi_ys in rois_ys]
        rois_z = [int(_[0]) for _ in all_line[2::3]]
        rois_z = [i-1 for i in rois_z]
        return rois_xs,rois_ys,rois_z
    
    def get_min_max_x(self):
        return [min(roi_xs) for roi_xs in self.rois_xs], [max(roi_xs) for roi_xs in self.rois_xs]

    def get_min_max_y(self):
        return [min(roi_ys) for roi_ys in self.rois_ys], [max(roi_ys) for roi_ys in self.rois_ys]

    @property
    def roi_number(self):
        return self._roi_number
    
    def analyze_pca(self):
        peak = [self.peak_number(i)/self.length(i) for i in range(self.roi_number)]
        length = [self.length(i) for i in range(self.roi_number)]
        intensity = [self.intensities(i) for i in range(self.roi_number)]
        branch = [self.branch_point_number(i)/self.length(i) for i in range(self.roi_number)]
        rectangle = [self.rectangle_full(i) for i in range(self.roi_number)]
        main = [len(self.get_longest_path(i))/2*len(self.rois_xs[i]) for i in range(self.roi_number)]
        self.pca_data = pd.DataFrame({"peak":peak, 
                                      "length": length,
                                      "intensity": intensity,
                                      "branch point": branch,
                                      "rectangle full": rectangle,
                                      "main": main})
        # width = [self.xys_to_array(i)[0].shape[0] for i in range(self.roi_number)]
        # height = [self.xys_to_array(i)[0].shape[1] for i in range(self.roi_number)]
        # self.pca_data = pd.DataFrame({"length": length,
        #                               "height": height,
        #                              "width": width})
        self.pca_data = self.pca_data.apply(lambda x: (x-x.mean())/x.std(), axis=0)
        pca = PCA()
        pca.fit(self.pca_data)
        self.feature = pca.transform(self.pca_data)
        return pca
    
    def plot_contribution_ratio(self):
        plt.figure(figsize=(6, 6))
        for x, y, name in zip(self.pca.components_[0], self.pca.components_[1], self.pca_data.columns):
            plt.text(x, y, name)
        plt.scatter(self.pca.components_[0], self.pca.components_[1], alpha=0.8)
        plt.grid()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()
    
    def xys_to_array(self,r):
        roi_xs = self.rois_xs[r]
        roi_ys = self.rois_ys[r]
        frame = self.rois_z[r]
        
        x_min = self.rois_min_x[r]
        x_max = self.rois_max_x[r]
        y_min = self.rois_min_y[r]
        y_max = self.rois_max_y[r]
        
        image_arr = self.I[frame][y_min:y_max+1,x_min:x_max+1]
        arr = np.zeros((x_max-x_min+1,y_max-y_min+1))   
        for x,y in zip(roi_xs,roi_ys):
            x = x - x_min
            y = y - y_min
            arr[x,y] = 1
        arr_ = np.copy(arr)
        arr_ = signal.convolve2d(arr_, np.ones((3,3))) * np.pad(arr_,(1,1),"constant")
        arr_ = arr_[1:-1,1:-1]
        arr[np.where(arr_==9)] = 0
        return arr.T, image_arr
    
    def length(self,r):
        return len(self.rois_xs[r])
    
    def intensities(self,r):
#         i = []
#         arr = self.xys_to_array(r)[0]
        image_arr = self.xys_to_array(r)[1]
#         masked = list(zip(*np.where(arr==1)))
#         for _ in masked:
#             x = _[0]
#             y = _[1]
#             i.append(np.sum(image_arr[x-self.radius:x+self.radius+1,y-self.radius:y+self.radius+1]))
        return np.sum(image_arr)
    
    def edges(self,arr):
        arr_ = signal.convolve2d(arr, np.ones((3,3))) * np.pad(arr,(1,1),"constant")
        arr_ = arr_[1:-1,1:-1]
        return list(zip(*np.where(arr_==2)))
    
    def branch_points(self,arr):
        arr_ = signal.convolve2d(arr, np.ones((3,3))) * np.pad(arr,(1,1),"constant")
        arr_ = arr_[1:-1,1:-1]
        return list(zip(*np.where(arr_>3)))
    
    def get_path(self,arr,s,g):
        arr[g] = 3
        h,w = arr.shape[0],arr.shape[1]
        def solve(s):
            y = s[0]
            x = s[1]
            if arr[y,x] == 3:
                return [(y, x)]
            arr[y,x] = 0
            for (next_y, next_x) in [(y+1,x),(y,x+1),(y,x-1),(y-1,x),(y-1,x-1),(y-1,x+1),(y+1,x-1),(y+1,x+1)]:
                if (next_y < 0) or (next_y >= h) or (next_x < 0) or (next_x >= w) or (arr[next_y,next_x] == 0):
                    continue
                route=solve((next_y, next_x))
                if route:
                    return [(y, x)] + route
        return solve(s)
    
    def get_longest_path(self,r):
        arr = self.xys_to_array(r)[0]
        max_path_length = 0
        max_path = []
        for pair in itertools.combinations(self.edges(arr), 2):
            arr = self.xys_to_array(r)[0]
            cur = self.get_path(arr,pair[0],pair[1])
            if len(cur) > max_path_length:
                max_path = cur
                max_path_length = len(cur)
        return max_path
    
    def get_skelton_map(self,r):
        along = self.get_longest_path(r)
        arr = self.xys_to_array(r)[0]
        c = 10
        for a in along:
            x = a[0]
            y = a[1]
            arr[x,y] = c
            c += 1
        return arr
    
    def intensities_along(self,r,ax=None):
        i = []
        x_min = self.rois_min_x[r]
        y_min = self.rois_min_y[r]
        z = self.rois_z[r]
        along = self.get_longest_path(r)
        for a in along:
            x = a[1] + x_min
            y = a[0] + y_min
            i.append(np.sum(self.I[z][y-self.radius:y+self.radius+1,x-self.radius:x+self.radius+1]))
        k = 10
        if len(i) > 0:
            v = np.ones(k)
            i = np.convolve(i, v, mode='same')
#             i = rolling_ball(i, radius=1)
        else:
            i = np.zeros(k)
        if not ax == None:
            ax.plot(i)
        return i
    
    def intensities_along_rel(self,r,ax=None):
        i = self.intensities_along(r)
        i_min = np.min(i)
        i_max = np.max(i)
        if i_max-i_min != 0:
            i_rel = (i - i_min) / (i_max - i_min)
            if not ax == None:
                ax.plot(i_rel)
        else:
            i_rel = i
            if not ax == None:
                ax.plot(i_rel)
        return i_rel
    
    def intensity_stds_along(self,r,ax=None):
        i = []
        x_min = self.rois_min_x[r]
        y_min = self.rois_min_y[r]
        z = self.rois_z[r]
        along = self.get_longest_path(r)
        for a in along:
            x = a[0] + x_min
            y = a[1] + y_min
            i.append(np.std(self.I[z][y-self.radius:y+self.radius+1,x-self.radius:x+self.radius+1]))
        if not ax == None:
            ax.plot(i)
        return i
    
    def peaks(self,r,ax=None):
#         intensity = self.intensities_along(r)
        intensity = self.intensities_along_rel(r)
#         ind_peaks = signal.argrelmax(intensity, order=4)[0]
#         ind_peaks = signal.find_peaks(intensity,width=0,distance=1,height=0.1)[0]
        LoG = blob_log(intensity, max_sigma=50, num_sigma=1, threshold=0.015)
        ind_peaks = LoG.T[0].astype(np.int64)
        peaks_ = [ind_peaks,[intensity[p] for p in ind_peaks]]
        if not ax == None:
            ax.scatter(peaks_[0], peaks_[1], c="r")
        return peaks_
    
    def peak_number(self,r):
        return len(self.peaks(r)[0])
    
    def branch_point_number(self,r):
        arr = self.xys_to_array(r)[0]
        bpn = self.branch_points(arr)
        return len(bpn)
    
    def rectangle_full(self,r):
        masked = np.copy(self.xys_to_array(r)[1])
        masked[np.where(masked<220)] = 0
        masked[np.where(masked!=0)] = 1
        return np.sum(masked)/self.xys_to_array(r)[1].size
        
    def PCA(self,r,ax=None):
        if not ax == None:
            ax.scatter(self.feature[:, 0], self.feature[:, 1], alpha=0.8)
            ax.scatter(self.feature[r, 0], self.feature[r, 1], c="red")
            ax.grid()
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")    
        
    def show_image(self):
        app = QtWidgets.QApplication(sys.argv)
        fig = plt.figure(figsize=(12, 36), dpi=100)

        form = MyQt(roi_max=len(self.rois_xs), 
                    fig=fig, 
                    my_funcs=[self.xys_to_array,
                              self.intensities_along_rel,
                              self.get_skelton_map,
                              self.peaks,
                              self.PCA])
        form.show()
        sys.exit(app.exec_())
    
    

