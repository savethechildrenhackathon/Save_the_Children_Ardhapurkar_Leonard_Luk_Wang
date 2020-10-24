"""
Code adapted from William Basener, UVA Faculty of the School of Data Science
Save the Children Hackathon 
"""
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QToolBar, QWidget, QVBoxLayout, QHBoxLayout
from scipy.stats import uniform
from osgeo import osr, gdal
import datetime
from PIL import Image, ImageQt
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import csv

import sys
import random
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2

##Changed min and max col to be pixels it can sample from
## took our RGB from Image mode
class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class QImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.fname_before = 'before.tiff'
        self.fname_after = 'after.tiff'
        self.dirname = r'C:\Users\Amy\Documents\Save the Children Hackathon\Pictures'
        os.chdir(self.dirname)
        self.datestamp = str(datetime.datetime.now()).replace(":", "_").replace(".", "_").replace(" ", "_")
        self.dirname_output = self.dirname+'output_'+self.datestamp+'/'
        # create this output directory if it does not exist
        if not os.path.exists(self.dirname_output):
            os.makedirs(self.dirname_output)
        self.fname_output = self.dirname_output+'chip_data_'+self.datestamp+'.csv'
        self.im_before = gdal.Open(self.fname_before)
        self.im_after = gdal.Open(self.fname_after)
        self.minCol, self.maxCol =1, 1466-257 #I think this is 0 indexed
        self.minRow, self.maxRow = 1, 832-257
        self.xsize, self.ysize = 256, 256
        self.data_output = list()

        self.canvas_before = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_after = MplCanvas(self, width=5, height=4, dpi=100)

        self.widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.canvas_before)
        layout.addWidget(self.canvas_after)
        self.widget.setLayout(layout)


        self.setCentralWidget(self.widget)

        self.createToolbar()

        self.setWindowTitle("Image Viewer")
        self.resize(1600, 800)

        self.pick_row_col()
        self.display_image()

    def createToolbar(self):

        self.toolbar = QToolBar("My main toolbar")
        self.toolbar.setIconSize(QSize(256, 256))
        self.addToolBar(self.toolbar)

        self.button_save_action = QAction("SAVE", enabled=True, triggered=self.save)
        self.button_ID0_action = QAction("ID 0", shortcut="0", enabled=True, triggered=self.ID0)
        self.button_ID1_action = QAction("ID 1", shortcut="1", enabled=True, triggered=self.ID1)
        self.button_ID2_action = QAction("ID 2", shortcut="2", enabled=True, triggered=self.ID2)
        self.button_ID3_action = QAction("ID 3", shortcut="3", enabled=True, triggered=self.ID3)
        self.button_ID4_action = QAction("ID 4", shortcut="4", enabled=True, triggered=self.ID4)
        self.button_ID5_action = QAction("ID 5", shortcut="5", enabled=True, triggered=self.ID5)
        self.button_clouds_action = QAction("Clouds", shortcut="c", enabled=True, triggered=self.clouds)
        self.button_natural_action = QAction("Natural", shortcut="n", enabled=True, triggered=self.natural)
        self.button_skip_action = QAction("Skip", shortcut="s", enabled=True, triggered=self.skip)
        self.toolbar.addAction(self.button_save_action)
        self.toolbar.addAction(self.button_ID0_action)
        self.toolbar.addAction(self.button_ID1_action)
        self.toolbar.addAction(self.button_ID2_action)
        self.toolbar.addAction(self.button_ID3_action)
        self.toolbar.addAction(self.button_ID4_action)
        self.toolbar.addAction(self.button_ID5_action)
        self.toolbar.addAction(self.button_clouds_action)
        self.toolbar.addAction(self.button_natural_action)
        self.toolbar.addAction(self.button_skip_action)

    def display_image(self):

        # get the chip from the full image
        chip_before = np.moveaxis(self.im_before.ReadAsArray(
            xoff=self.col, yoff=self.row, xsize=self.xsize, ysize=self.ysize
            ), 0, -1)
        # conver time PIL image
        im8_before = Image.fromarray(chip_before.astype('uint8'))
        # save the before chip image
        self.fname_before = 'X'+str(self.row)+'Y'+str(self.col)+'.tif'
        im8_before.save(self.dirname_output+self.fname_before, "TIFF")
        print(self.row)
        print(self.col)

        # use projections to get lat lon from _before and then row,col for this lat/lonin _after
        gt = self.im_before.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + 256 * gt[4] + 256 * gt[5]
        maxx = gt[0] + 256 * gt[1] + 256 * gt[2]
        maxy = gt[3]

        transf = self.im_after.GetGeoTransform()
        cols = self.im_after.RasterXSize
        rows = self.im_after.RasterYSize
        bands = self.im_after.RasterCount  # 1
        band = self.im_after.GetRasterBand(1)
        bandtype = gdal.GetDataTypeName(band.DataType)  # Int16
        driver = self.im_after.GetDriver().LongName  # 'GeoTIFF'
        transfInv = gdal.InvGeoTransform(transf)

        self.col_after, self.row_after = gdal.ApplyGeoTransform(transfInv, minx, miny)
        self.col_after = int(self.col_after)
        self.row_after = int(self.row_after)
        #before_cs = osr.SpatialReference()
        #before_cs.ImportFromWkt(self.im_before.GetProjectionRef())
        #after_cs = osr.SpatialReference()
        #after_cs.ImportFromWkt(self.im_after.GetProjectionRef())
        #transform = osr.CoordinateTransformation(before_cs, after_cs)
        #latlong = transform.TransformPoint(self.row, self.col)

        # hardcode for now since both the before and after image cover the same area and are the same pixel size 
        self.row_after = self.row
        self.col_after = self.col
        self.xsize_after = 256
        self.ysize_after = 256
        
        print(self.col_after+self.xsize_after, "is less than", self.im_after.RasterXSize)
        print(self.row_after+self.ysize_after, "is less than",self.im_after.RasterYSize )
    
        if ((self.col_after+self.xsize_after < self.im_after.RasterXSize) and
            (self.row_after+self.ysize_after < self.im_after.RasterYSize)):
   
            chip_after = np.moveaxis(self.im_after.ReadAsArray(
                xoff=self.col_after, yoff=self.row_after, xsize=self.xsize_after, ysize=self.ysize_after
                ), 0, -1)

            # conver time PIL image
            im8_after = Image.fromarray(chip_after.astype('uint8'))
            # save the before chip image
            self.fname_after = 'X'+str(self.row_after)+'Y'+str(self.col_after)+'.tif'
            im8_after.save(self.dirname_output + self.fname_after, "TIFF")

            self.canvas_before.axes.cla()  # Clear the canvas.
            self.canvas_before.axes.imshow(im8_before)
            # Trigger the canvas to update and redraw.
            self.canvas_before.draw()
            self.canvas_after.axes.cla()  # Clear the canvas.
            self.canvas_after.axes.imshow(im8_after)
            # Trigger the canvas to update and redraw.
            self.canvas_after.draw()

            self.data_dict_current = {
              "Damge Type": 'None',
              "Filename Before": self.fname_before,
              "Filename After": self.fname_after
            }

            # Compute features
            features = self.compute_features(chip_before, chip_after)
            for key in features.keys():
                self.data_dict_current[key] = features[key]
            self.data_output.append(self.data_dict_current)

    def compute_features(self, chip_before, chip_after):

        # compute eigenvalues and det of each chip
        # before image
        nc, nr, nb =np.shape(chip_before)
        chip_before_list = np.reshape(chip_before, [nc*nr, nb]).T
        cov_before = np.cov(chip_before_list)
        evals_before, _ = np.linalg.eig(cov_before)
        det_before = np.prod(evals_before)
        # after image
        nc, nr, nb =np.shape(chip_after)
        chip_after_list = np.reshape(chip_after, [nc*nr, nb]).T
        cov_after = np.cov(chip_before_list)
        evals_after, _ = np.linalg.eig(cov_after) # array of eigenvalues
        det_after = np.prod(evals_after)

        # change in pixel values
        pixel_change = chip_before - chip_after
        avg_pixel_change = np.sum(np.abs(chip_before - chip_after)) / (256 * 256)
        red_pixel_change = np.sum(np.abs(pixel_change[:,:,0])) / (256 * 256)
        green_pixel_change = np.sum(np.abs(pixel_change[:,:,1])) / (256 * 256)
        blue_pixel_change = np.sum(np.abs(pixel_change[:,:,2])) / (256 * 256)
        
        #change in edge count 
        
        edge_before = cv2.Canny(chip_before,10, 200 ) 
        edge_count_before = np.count_nonzero(edge_before)
        edge_after = cv2.Canny(chip_after,10, 200 ) 
        edge_count_after = np.count_nonzero(edge_after)
        edge_change = np.abs(edge_count_before-edge_count_after)
        
     

        plt.subplot(121),plt.imshow(edge_before, cmap = 'gray')
        plt.title('Edge After Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edge_after,cmap = 'gray')
        plt.title('Edge After Image'), plt.xticks([]), plt.yticks([])
        
        plt.show()
        
       
        
        features = {
            "Evals After": evals_before,
            "Det After": det_before,
            "Evals Before": evals_after,
            "Det After": det_after,
            "Avg Pixel Change": avg_pixel_change,
            "Avg Red Pixel Change": red_pixel_change,
            "Avg Green Pixel Change": green_pixel_change,
            "Avg Blue Pixel Change": blue_pixel_change,
            "Edge Count Change": edge_change
        }

        return features

    def pick_row_col(self):
        self.row = int(uniform.rvs(size=1, loc=self.minRow, scale=self.maxRow-self.minRow))
        self.col = int(uniform.rvs(size=1, loc=self.minCol, scale=self.maxCol-self.minCol))

    def ID0(self):
        print(len(self.data_output[-1]['Damge Type']))
        self.data_output[-1]['Damge Type'] = ('0')
        self.pick_row_col()
        self.display_image()

    def ID1(self):
        self.data_output[-1]['Damge Type'] = ('1')
        self.pick_row_col()
        self.display_image()

    def ID2(self):
        self.data_output[-1]['Damge Type'] = ('2')
        self.pick_row_col()
        self.display_image()

    def ID3(self):
        self.data_output[-1]['Damge Type'] = ('3')
        self.pick_row_col()
        self.display_image()

    def ID4(self):
        self.data_output[-1]['Damge Type'] = ('4')
        self.pick_row_col()
        self.display_image()

    def ID5(self):
        self.data_output[-1]['Damge Type'] = ('5')
        self.pick_row_col()
        self.display_image()

    def clouds(self):
        self.data_output[-1]['Damge Type'] = ('Clouds')
        self.pick_row_col()
        self.display_image()

    def natural(self):
        self.data_output[-1]['Damge Type'] = ('Natural')
        self.pick_row_col()
        self.display_image()

    def skip(self):
        self.data_output[-1]['Damge Type'] = ('skipped')
        self.pick_row_col()
        self.display_image()

    def save(self):
        print('save')

        # write
        with open(self.fname_output, 'w', encoding='utf8', newline='') as output_file:
            fc = csv.DictWriter(output_file,
                                fieldnames=self.data_output[0].keys(),
                                )
            fc.writeheader()
            fc.writerows(self.data_output)
            #fc.close()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
