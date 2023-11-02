""" 
GUI Class to implement and call all the functions of BRI Class 

Author: Amir Abbas Babasafari (AB)
Date: October 2023
Email: a.babasafari@yahoo.com   
"""

# Import Libraries
import os
import sys
import numpy as np

from bri_class import BRI

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
  QApplication,
  QHBoxLayout,
  QLabel,
  QMainWindow,
  QPushButton,
  QVBoxLayout,
  QWidget,
  QAction,
  QComboBox,
  QMessageBox,
  QFileDialog,
  QSpacerItem, 
  QSizePolicy,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot as plt


# Call BRI class 
class BRI_interface(BRI):

    def __init__(self):
        super(BRI, self).__init__()
        pass 
    

# Create a canvas for plot
class FigCanvas(FigureCanvas):
    def __init__(self, parent = None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom = 0.1, right = 0.95, top = 0.9, left = 0.05)


# Class for the main GUI window
class Main_Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.title = 'BRI'
        self.left = 10
        self.top = 10
        self.width = 1920
        self.height = 1080

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.statusBar().showMessage('Ready')
        self.setWindowTitle("BRI Application")

        # Create main layout and sub-layouts
        layout = QVBoxLayout()

        # Layout1 with corresponding widgets and functions to be triggered
        layout1 = QHBoxLayout()
        self.Button1 = QPushButton('Location Map')
        self.Button2 = QPushButton('BRI Spectral Enhancement')
        self.Combo = QComboBox()

        self.Button1.clicked.connect(self.plot_map)
        self.Button2.clicked.connect(self.run_bri)
        self.Button1.setEnabled(False)
        self.Button2.setEnabled(False)
        self.Combo.addItems(['Original Seismic Display','BRI Seismic Display'])
        self.Combo.setEnabled(False)
        self.Combo.currentTextChanged.connect(self.change_display)

        layout1.addWidget(self.Button1)
        layout1.addWidget(self.Button2)
        layout1.addWidget(self.Combo)
        layout.addLayout(layout1)

        # Layout2 with corresponding widgets and functions to be triggered
        layout2 = QHBoxLayout()
        label = QLabel("Quality Control Tools")
        self.setCentralWidget(label)
        label.setAlignment(Qt.AlignCenter)
        self.Button3 = QPushButton('Amplitude Spectrum')
        self.Button4 = QPushButton('Overlay Comparison')

        self.Button3.setEnabled(False)
        self.Button4.setEnabled(False)
        self.Button3.clicked.connect(self.amplitude_spectrum)
        self.Button4.clicked.connect(self.compare_traces)

        layout2.addWidget(label)
        layout2.addWidget(self.Button3)
        layout2.addWidget(self.Button4)
        layout.addLayout(layout2)

        # Layout3 for toolbar with corresponding widgets and functions to be triggered
        self.canvas = FigCanvas(self, width=10, height=8)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout3 = QVBoxLayout()
        layout3.addWidget(self.toolbar)
        layout3.addItem(QSpacerItem(1000, 10, QSizePolicy.Expanding))
        layout3.setAlignment(Qt.AlignCenter)

        layout.addLayout(layout3)

        # Layout4 for canvas with corresponding widgets and functions to be triggered
        layout4 = QVBoxLayout()
        layout4.addWidget(self.canvas)
        layout.addLayout(layout4)

        # Create a placeholder widget
        widget = QWidget()
        widget.setLayout(layout)

        # Menu-bar with corresponding widgets and functions to be triggered
        menu = self.menuBar()
        menu.setNativeMenuBar(False)
        file_menu = menu.addMenu("&Data")

        button_load = QAction( "Load Original Seismic", self)
        button_save = QAction( "Save BRI Seismic", self)
        exitButton = QAction("Exit", self)
        exitButton.setStatusTip('Exit application')

        file_menu.addAction(button_load)
        file_menu.addAction(button_save)
        file_menu.addAction(exitButton)

        button_load.triggered.connect(self.load_segy_file)
        button_save.triggered.connect(self.save_segy_file)
        exitButton.triggered.connect(self.close)

        self.setCentralWidget(widget)

        # Initialize input file
        self.filepath_in = None

        # BRI class inherited from parents
        self.bri_spectral_class = BRI_interface()

    # Function to load and read segy file
    def load_segy_file(self):

        self.update_canvas()

        try:
            self.fileName = QFileDialog.getOpenFileName(self,
            str("Open a file (2D/3D post-stack seismic data)"), os.getcwd(), 
            str("Segy File (*.sgy *.segy);;""All Files (*.*)"))
            self.filepath_in = str(self.fileName[0])
            print(self.filepath_in)
                        
            self.bri_spectral_class.initialize()
            self.bri_spectral_class.define_data_dimension_stack_display(self.filepath_in)
            self.bri_spectral_class.read_seismic_segy(self.filepath_in)
            self.bri_spectral_class.read_segy_byte_locations(self.filepath_in)
            
            self.Combo.setCurrentIndex(0)
            self.Button1.setEnabled(True)
            self.Button2.setEnabled(True)
            self.Combo.setEnabled(False)
            self.Button3.setEnabled(False)
            self.Button4.setEnabled(False)

            self.plot(self.bri_spectral_class.data_display)
            self.load_message_success()

        except:
            self.load_message_failure()

    # Function to update canvas
    def update_canvas(self):
            
            try:
                # self.canvas.fig.clf()
                self.canvas.ax.cla()
                self.cb.remove()
                # self.show()
                self.canvas.draw()
            except:
                pass

    # Function for data type information            
    def data_type_display(self):
        if self.bri_spectral_class.data_type == 'Post-stack 2D':
            id = 'All traces are displayed'
        if self.bri_spectral_class.data_type == 'Post-stack 3D':
            id = ("Traces of middle inline are displayed. "
                 "However BRI will be applied to the entire traces of 3D volume.")
        return id
    
    # Function to plot and update seismic data 
    def plot(self, data):
        self.image = self.canvas.ax.imshow(data, cmap = plt.cm.seismic, 
                aspect = 'auto', vmin = -np.max(data), vmax = np.max(data), 
                extent = (min(self.bri_spectral_class.cdp_no), max(self.bri_spectral_class.cdp_no), 
                          max(self.bri_spectral_class.twt), min(self.bri_spectral_class.twt)))

        self.canvas.ax.set_xlabel("CDP No.")
        self.canvas.ax.set_ylabel("Time (ms)")
        self.canvas.ax.grid(True)

        try:
            self.canvas.ax.set_title(
                "Seismic file name: {0} \n Seismic data type: {1} , No. Traces: {2} , No. Samples: {3} ,"  
                "Sample_rate: {4}ms , Start_time: {5}ms , Trace_length: {6}ms , Format: {7} \n {8}".format(
                os.path.splitext(os.path.basename(self.filepath_in))[0], self.bri_spectral_class.data_type, 
                self.bri_spectral_class.n_traces, self.bri_spectral_class.n_samples, self.bri_spectral_class.sample_rate, 
                self.bri_spectral_class.twt[0], max(self.bri_spectral_class.twt) - min(self.bri_spectral_class.twt), 
                self.bri_spectral_class.data_format, self.data_type_display()))
        except:
            pass


        try:
            self.cb.remove()
        except:
            pass

        self.cb = self.canvas.fig.colorbar(self.image)
        # self.show()
        self.canvas.draw()

    # Function to plot location map
    def plot_map(self):
        self.bri_spectral_class.xy_plot()

    # Function to run BRI
    def run_bri(self):
        self.bri_spectral_class.amplitude_spectrum_seismic = self.bri_spectral_class.calculate_amplitude_spectrum(
            self.bri_spectral_class.seismic_data, mode = 'seismic amplitude')
        self.bri_spectral_class.mean_spectrum_normalized = self.bri_spectral_class.mean_amplitude_spectrum(
            self.bri_spectral_class.amplitude_spectrum_seismic, kernel_length = 1)
        self.bri_spectral_class.get_rc_from_seismic(plot = False)
        self.bri_spectral_class.calculate_amplitude_spectrum(self.bri_spectral_class.rc_series, 
                                                             mode = 'rc amplitude')
        self.bri_spectral_class.find_f4_index(self.bri_spectral_class.mean_spectrum_normalized)
        self.bri_spectral_class.fit_curve_rc_spectrum(self.bri_spectral_class.mean_spectrum_normalized, 
                                                      method = 'calc intrcpt_grdnt', plot = False)
        self.bri_spectral_class.bri_operator()
        self.bri_spectral_class.bri_data = self.bri_spectral_class.convolve_seismic_operator(
            self.bri_spectral_class.seismic_data)
        self.Combo.setEnabled(True)
        self.Button3.setEnabled(True)
        self.Button4.setEnabled(True)

    # Function to update seismic plot once it changes
    def change_display(self):

        if self.Combo.currentIndex() == 0:
            self.plot(self.bri_spectral_class.data_display)

        else:
            self.bri_spectral_class.bri_data_display = self.bri_spectral_class.convolve_seismic_operator(
                self.bri_spectral_class.data_display)
            self.plot(self.bri_spectral_class.bri_data_display)

    # Function to calculate amplitude spectrum
    def amplitude_spectrum(self):
        self.bri_spectral_class.bri_amplitude_spectrum_seismic = self.bri_spectral_class.calculate_amplitude_spectrum(
            self.bri_spectral_class.bri_data, mode = 'seismic amplitude')
        self.bri_spectral_class.bri_mean_spectrum_normalized = self.bri_spectral_class.mean_amplitude_spectrum(
            self.bri_spectral_class.bri_amplitude_spectrum_seismic, kernel_length = 1)
        self.bri_spectral_class.plot_amplitude_spectrum(
            self.bri_spectral_class.mean_spectrum_normalized, normalized_spectrum_BRI = self.bri_spectral_class.bri_mean_spectrum_normalized)

    # Function for 1D overlay comparison
    def compare_traces(self):
        self.bri_spectral_class.compare_input_output_1d(
            self.bri_spectral_class.twt[0], self.bri_spectral_class.twt[-1])

    # Function to save BRI seismic segy
    def save_segy_file(self):
        try:
            self.bri_spectral_class.export_seismic_segy(self.filepath_in, self.bri_spectral_class.bri_data)
            self.bri_spectral_class.write_segy_byte_locations(self.bri_spectral_class.filepath_out)
            self.save_message_success()
        except:
            self.save_message_failure()

    # Function to show successful message after loading
    def load_message_success(self):

        QMessageBox.information(self,"Load File",
        "Seismic data was loaded successfully",
        buttons=QMessageBox.Close)

    # Function to show failed message after loading
    def load_message_failure(self):
        
        QMessageBox.critical(self,"Load File",
        "Please make sure that loaded data is 2D or 3D post-stack seismic data, pre-stack is not supported."
        " If data is post-stack, please check inline and crossline byte locations;" 
        " supported inline_xline byte locations are [189,193], [9,21], and [5,21]",
        buttons=QMessageBox.Close)

    # Function to show successful message after saving
    def save_message_success(self):
        button = QMessageBox.information(
        self,
        "Save File",
        "BRI seismic data was saved successfully",
        buttons=QMessageBox.Close)

    # Function to show failed message after saving
    def save_message_failure(self):
        button = QMessageBox.warning(
        self,
        "Save File",
        "Please run BRI Spectral Enhancement",
        buttons=QMessageBox.Close)


if __name__ == '__main__':

    print('BRI module has started')

    app = QApplication(sys.argv)
    mainWin = Main_Window()
    mainWin.show()
    sys.exit(app.exec_())
    
