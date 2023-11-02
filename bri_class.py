""" 
Class for Seismic Spectral Enhancement based on 
the Blueing Reflectivity Integration (BRI) Technique
proposed by Babasafari et al., 2016

Author: Amir Abbas Babasafari (AB)
Date: October 2023
Email: a.babasafari@yahoo.com   
"""

# Import Libraries
import os
import numpy as np
import segyio 
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.optimize import curve_fit


class BRI:
    def __init__(self):
        # initializing
        self.initialize()
        

    def initialize(self):
        
        # Initialize values
        self.seismic_data = None
        self.data_display = None
        self.data_type = None
        self.cdp_no = None
        self.twt = None
        self.n_traces = None
        self.n_samples = None
        self.sample_rate = None
        self.Source_X_plot = None
        self.Source_Y_plot = None
        self.CDP_X_plot = None
        self.CDP_Y_plot = None
        self.sample_rate_scaled = None
        self.nyquist_frequency = None
        self.amplitude_spectrum = None
        self.amplitude_spectrum_seismic = None
        self.mean_spectrum_normalized = None
        self.amplitude_spectrum_rc = None
        self.frequency_axis_interpolated = None
        self.rc_series = None
        self.frequency_axis_interpolated_upto_f4 = None
        self.amplitude_spectrum_rc_upto_f4 = None
        self.fitted_curve = None
        self.operator_time_axis = None
        self.bri_operator_amplitude = None
        self.bri_amplitude_spectrum_seismic = None
        self.bri_mean_spectrum_normalized = None
        self.bri_data = None
        self.bri_data_display = None
        self.filepath_out = None

        # Initialize byte locations
        # TRACE_SEQUENCE_FILE _ byte location:5
        self.TraceSequenceFile = []
        # FieldRecord _ byte location:9
        self.Field_Record = []
        # Trace_Field _ byte location:13
        self.Trace_Field = []
        # CDP _ byte location:21
        self.CDP = []
        # Trace_Number _ byte location:25
        self.Trace_Number = []
        # Scaler_coordinate _ byte location:71
        self.Scaler_coordinate = []
        # SourceX _ byte location:73
        self.Source_X = []
        # SourceY _ byte location:77
        self.Source_Y = []
        # CDP_X _ byte location:181
        self.CDP_X = []
        # CDP_Y _ byte location:185
        self.CDP_Y = []
        # INLINE_3D _ byte location:189
        self.Inline_3D = []
        # CROSSLINE_3D _ byte location:193
        self.Crossline_3D = []


    def define_data_dimension_stack_display(self, filepath_in):    
        """   
        Function to define data type as 2D or 3D and Post-Stack or Pre-Stack as well as 
        selecting data for display 
        
        Parameter:
        ----------
        filepath_in (str): file path of loaded segy file
        
        Returns:
        --------
        data_display (numpy.ndarray): Seismic amplitude traces to plot
        If data is 2D (arbitrary line, single inline, or single crossline), all traces are displayed
        If data is 3D, traces of middle inline are displayed. 
        However BRI is applied to the entire traces of 3D volume

        data_type (str): Define data as 2D or 3D and Post-Stack or Pre-Stack
        
        seismic_data_shape (tuple): Shape of loaded seismic data
        
        cdp_no (numpy.ndarray): array of CDP number for seismic data to display
        
        """

        # Supported inline and crossline byte locations
        inline_xline = [[189,193], [9,21], [5,21]]
        
        # Read segy data with the specified byte location of geometry 
        for byte_loc in inline_xline:

            try:
                with segyio.open(filepath_in, iline = byte_loc[0], xline = byte_loc[1], ignore_geometry = False) as f:
                    self.seismic_data_vol = segyio.tools.cube(f)
                    n_traces = f.tracecount    
                    # trc_no = f.bin[segyio.BinField.Traces]
                    trc_no = f.attributes(segyio.TraceField.TraceNumber)[-1]
                    if not isinstance(trc_no, int):
                        trc_no = f.attributes(segyio.TraceField.TraceNumber)[-2] + 1
                    trc_no = int(trc_no[0])
                    spec = segyio.spec()
                    spec.sorting = f.sorting
                    data_sorting = spec.sorting == segyio.TraceSortingFormat.INLINE_SORTING

            except:
                pass

        # Define sort of data    
        try:
            if data_sorting is True:
                print('Data is sorted by Inline')
            else:
                print('Data is sorted by Crossline')

        except:
            print('Error, Please check inline and crossline byte locations')
            raise  

            
        # Define data as 2D/3D and Post-stack/Pre-stack
        if len(self.seismic_data_vol.shape) == 3:
            if self.seismic_data_vol.shape[0] != 1:
                self.data_type = 'Post-stack 3D'
            else:
                if n_traces > trc_no > 1:   
                    self.data_type = 'Post-stack 3D'
                else:
                    self.data_type = 'Post-stack 2D'
                
        else:        
            if len(f.offsets) > 1:
                if self.seismic_data_vol.shape[0] == 1:
                    self.data_type = 'Pre-Stack 2D'
                else:
                    self.data_type = 'Pre-Stack 3D'    
            else:
                print('Error, Please check inline and crossline byte locations')

        print('Data Type: {0}'.format(self.data_type))
        print('Seismic Data Shape: {0}'.format(self.seismic_data_vol.shape))

        #  The application supports post-stack data
        if self.data_type == 'Post-stack 2D' or self.data_type == 'Post-stack 3D':
            pass
        else:
            print('Please make sure that data loaded is 2D or 3D post-stack seismic data, pre-stack is not supported') 
        
        
        # Specify data for display and CDP range based on type of data    
        inline, cdp, samples = self.seismic_data_vol.shape
        
        if self.data_type == 'Post-stack 2D':
            self.data_display = self.seismic_data_vol.reshape(cdp, samples).T
            self.cdp_no = np.arange(n_traces)    
        
        elif self.data_type == 'Post-stack 3D':
            if inline == 1 and trc_no > 1 and n_traces % trc_no == 0:  
                inline_no =  n_traces / trc_no
                inline_mid = int(inline_no / 2)
                cdp_range = np.arange(inline_mid * trc_no, inline_mid * trc_no + trc_no)
                self.data_display = self.seismic_data_vol.reshape(cdp, samples).T
                self.data_display = self.data_display[:, cdp_range]
                self.cdp_no = np.arange(trc_no)
                print('Seismic Data Shape after reshape: {0}'.format((int(inline_no), int(trc_no), int(samples))))

            else:  
                inline_mid = int(inline / 2)
                self.data_display = self.seismic_data_vol[inline_mid, :, :].T
                self.cdp_no = np.arange(cdp)
                

    def read_seismic_segy(self, filepath_in):
        """   
        Function to read 2D/3D post-stack seismic data (segy file) and specify data-related parameters
        
        Parameter:
        ----------
        filepath_in (str): file path of loaded segy file
        Note: Segy data should be in time domain and post-stack seismic data
        
        Returns:
        --------
        data (numpy.ndarray): All seismic amplitude traces in 2D array
        
        n_traces (int): Number of traces
        
        sample_rate (float): Sample rate of seismic data in ms
        
        twt (numpy.ndarray): array of TWT number
        
        """
        
        # Read segy data and specify data parameters
        try:
            with segyio.open(filepath_in, ignore_geometry=True) as f:
                # Get the attributes
                self.n_traces = f.tracecount
                self.sample_rate = segyio.tools.dt(f) / 1000
                self.n_samples = f.samples.size
                self.twt = f.samples
                self.seismic_data = f.trace.raw[:].T 
                spec = segyio.spec()
                self.data_format = f.format
                # print(f.bin)
                print('No. Traces: {0}, No. Samples: {1}, Sample_rate: {2}ms, Start_time: {3}ms, Trace_length: {4}ms, Format: {5}'
                    .format(self.n_traces, self.n_samples, self.sample_rate, self.twt[0], max(self.twt) - min(self.twt), self.data_format))

        except:
            print('Please make sure that inline and crossline byte locations are correct')
            print('In addition, please check that data loaded is 2D or 3D post-stack seismic data, pre-stack is not supported')
            raise
            

    def read_segy_byte_locations(self, filepath_in):  
        """   
        Function to extract header information of segy file (Geometry-related byte locations) 
        
        Parameter:
        ----------
        filepath_in (str): file path of loaded segy file
        Note: Segy data should be in time domain and post-stack seismic data
        
        Returns:
        --------
        TraceSequenceFile, Field_Record, CDP, Source_X, Source_Y, CDP_X, CDP_Y, Inline_3D, and Crossline_3D (list): 
        Geometry-related byte locations
        """
        
        # Extract standard geometry-related byte locations from segy header  
        with segyio.open(filepath_in, ignore_geometry=True) as f:
            
            # Get the Geometry-related attributes
            n_traces = f.tracecount

            for i in range(n_traces):
                trace_seq_no = f.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[i]; self.TraceSequenceFile.append(trace_seq_no)
                field_record = f.attributes(segyio.TraceField.FieldRecord)[i]; self.Field_Record.append(field_record)
                trace_field = f.attributes(segyio.TraceField.TraceNumber)[i]; self.Trace_Field.append(trace_field)
                cdp = f.attributes(segyio.TraceField.CDP)[i]; self.CDP.append(cdp)
                cdp_tr_no = f.attributes(segyio.TraceField.CDP_TRACE)[i]; self.Trace_Number.append(cdp_tr_no)
                scale = f.attributes(segyio.TraceField.SourceGroupScalar)[i]; self.Scaler_coordinate.append(scale)
                gx = f.attributes(segyio.TraceField.SourceX)[i]; self.Source_X.append(gx)
                gy = f.attributes(segyio.TraceField.SourceY)[i]; self.Source_Y.append(gy)
                cdp_x = f.attributes(segyio.TraceField.CDP_X)[i]; self.CDP_X.append(cdp_x)
                cdp_y = f.attributes(segyio.TraceField.CDP_Y)[i]; self.CDP_Y.append(cdp_y)
                inline = f.attributes(segyio.TraceField.INLINE_3D)[i]; self.Inline_3D.append(inline)
                xline = f.attributes(segyio.TraceField.CROSSLINE_3D)[i]; self.Crossline_3D.append(xline)


    def xy_plot(self):
        """   
        Function for scatter-plotting of X and Y Geometry
        
        Parameter:
        ----------
        Source_X, Source_Y, CDP_X, and CDP_Y (list)
        Geometry-related byte locations
        
        Returns:
        --------
        Scatter-plot of X and Y Coordinates    
        """

        if np.any(self.Scaler_coordinate) != 0:
            self.Scaler = np.asarray([np.abs(1/x) for x in self.Scaler_coordinate], dtype = float)
            self.Source_X_plot = self.Source_X * self.Scaler
            self.Source_Y_plot = self.Source_Y * self.Scaler
            self.CDP_X_plot = self.CDP_X * self.Scaler
            self.CDP_Y_plot = self.CDP_Y * self.Scaler
        else:
            self.Source_X_plot = self.Source_X
            self.Source_Y_plot = self.Source_Y
            self.CDP_X_plot = self.CDP_X
            self.CDP_Y_plot = self.CDP_Y

        # Plot data X and Y based on the standard byte locations (73,77) or (181,185)
        if np.any(self.Source_X_plot) != 0:
            plt.scatter(self.Source_X_plot, self.Source_Y_plot)
        else:
            plt.scatter(self.CDP_X_plot, self.CDP_Y_plot)
        
        # set the axis lables
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title("XY scatter-plot")
        plt.grid(True)
        plt.show()
        

    def display_seismic_data(self, filepath_in):    
        """   
        Function to display seismic data (segy file)
        
        Parameter:
        ----------
        seismic_data (numpy.ndarray): seismic amplitude traces selected to display
        
        cdp_no (numpy.ndarray): array of CDP number for seismic data to display

        twt (numpy.ndarray): array of TWT number
        
        Returns:
        --------
        Plot seismic traces        
        """
    
        # Plot seismic data 2D (entire line) and 3D (the mid inline number)
        plt.figure(figsize=(15,15))
        plt.imshow(self.data_display, interpolation = 'nearest', cmap = plt.cm.seismic, aspect = 'auto', 
                vmin = np.min(self.data_display), vmax = np.max(self.data_display), 
                extent = (min(self.cdp_no), max(self.cdp_no), max(self.twt), min(self.twt)))
        plt.xlabel("CDP No.")
        plt.ylabel("Time (ms)")
        plt.title("Seismic file name: {0}".format(os.path.splitext(os.path.basename(filepath_in))[0]))
        plt.grid(True)
        plt.colorbar()
        plt.show()
        

    def calculate_amplitude_spectrum(self, data, mode = 'seismic amplitude'):    
        
        """   
        Function to calculate amplitude spectrum
        
        Parameter:
        ----------
        seismic_data (numpy.ndarray): All seismic amplitude traces in 2D array
        sample_rate (float): Sample rate of seismic data in ms
        mode (str): 'seismic amplitude' , 'rc amplitude'

        Returns:
        --------
        amplitude_spectrum (numpy.ndarray): Amplitude spectrum of selected traces
        frequency_axis_interpolated (numpy.ndarray): Interpolated frequency axis
        nyquist_frequency (int): Nyquist frequency (HZ)
        
        """

        if mode == 'seismic amplitude':
            # Select a weight for skipping traces on a coarse grid
            coef = 0.1
            # Select every n trace for averaging the amplitude spectrum
            n = int(data.shape[1] * coef)

            if data.shape[1] > 1: 
                traces = data[:,::n]
            else:
                traces = data
        
        elif mode == 'rc amplitude':
            traces = data
            
        # Define sample rate and nyquist frequency
        self.sample_rate_scaled = self.sample_rate * 1e-3
        self.nyquist_frequency = int(1 / (2 * self.sample_rate_scaled))
        
        # Initialize amplitude based on nyquist frequency
        self.amplitude_spectrum = np.empty(shape=(self.nyquist_frequency, traces.shape[1]))

        # Calculate amplitude spectrum using fourier transform
        for i in range(traces.shape[1]):
            
            n_sample = len(traces[:,i])
            spectrum = np.fft.fft(traces[:,i] / n_sample)
            frequency = np.fft.fftfreq(n_sample)
            
            # Display positive frequency content with corresponding absolute and real amplitude
            spectrum_real = abs(spectrum[:int(n_sample / 2)].real)
            # scale frequency axis wih Nyquist frequency for positive axis only
            frequency_axis = frequency[:int(n_sample / 2)] * (1 / self.sample_rate_scaled)

            # Interpolate spectrum_real and freq_axis
            self.frequency_axis_interpolated = np.arange(0, self.nyquist_frequency, 1)
            spectrum_real_interpolated = np.interp(self.frequency_axis_interpolated, frequency_axis, spectrum_real)
            self.amplitude_spectrum[:,i] = spectrum_real_interpolated

        if mode == 'seismic amplitude':
            amplitude_spectrum_seismic = self.amplitude_spectrum
            # The output will be seismic or bri
            return amplitude_spectrum_seismic
        
        elif mode == 'rc amplitude':
            self.amplitude_spectrum_rc = self.amplitude_spectrum
        

    def mean_amplitude_spectrum(self, amplitude_spectrum_seismic, kernel_length = 1):
        
        """   
        Function to calculate mean amplitude spectrum
        
        Parameter:
        ----------
        spectrum (numpy.ndarray): Amplitude spectrum of selected traces
        kernel_length (int): Smoothing factor

        Returns:
        --------
        mean_spectrum_normalized (numpy.ndarray): Normalized average amplitude spectrum 
        
        """

        # Calculate average of amplitude spectrum for all selected traces
        mean_spectrum = np.mean(amplitude_spectrum_seismic, axis=1) 
        
        # Smooth the average amplitude spectrum if needed
        kernel = np.ones(kernel_length) / kernel_length
        smooth_mean_spectrum = np.convolve(mean_spectrum, kernel, mode='same')
        
        # Normalize the average amplitude spectrum 
        mean_spectrum_normalized = (smooth_mean_spectrum-min(smooth_mean_spectrum))\
        /(max(smooth_mean_spectrum)-min(smooth_mean_spectrum))
        
        return mean_spectrum_normalized
        

    def plot_amplitude_spectrum(self, mean_spectrum_normalized, normalized_spectrum_BRI = None):
        
        """   
        Function to plot amplitude spectrum
        
        Parameter:
        ----------
        normalized_spectrum (numpy.ndarray): Normalized average amplitude spectrum 
        frequency_axis_interpolated (numpy.ndarray): Interpolated frequency axis
        normalized_spectrum_BRI (numpy.ndarray): Normalized average amplitude spectrum BRI data

        Returns:
        --------
        Plot normalized amplitude spectrum      
        
        """

        # Plot the smooth average amplitude spectrum 
        plt.plot(self.frequency_axis_interpolated, mean_spectrum_normalized, color='orange', \
                label = 'Original Data', linestyle='solid', linewidth = 3)
        
        if not normalized_spectrum_BRI is None:
            plt.plot(self.frequency_axis_interpolated, normalized_spectrum_BRI, color='blue', \
                label = 'BRI Data', linestyle='solid', linewidth = 3) 
        plt.xlabel('Frequency (HZ)')
        plt.ylabel('Normalized Amplitude')
        plt.title('Amplitude Spectrum upto Nyquist Frequency')
        plt.grid(True)
        plt.legend(loc = 'upper right')
        plt.show()
        

    def get_rc_from_seismic(self, plot = True):
        
        """   
        Function to calculate pseudo Reflection Coefficient (RC) from seismic data
        
        'find min and max local minima along seismic traces by calculating zero amplitude of first derivative'
        'It is considered as reflection coefficient to be replaced with rc from well log'

        Parameter:
        ----------
        seismic_data (numpy.ndarray): All seismic amplitude traces in 2D array
        
        Returns:
        --------
        rc_series (numpy.ndarray): Calculated pseudo reflection coefficient from seismic data
            
        """

        # Select a weight for skipping traces on a coarse grid
        coef = 0.1
        # Select every n trace for averaging the amplitude spectrum
        n = int(self.seismic_data.shape[1] * coef)

        if self.seismic_data.shape[1] > 1: 
            traces = self.seismic_data[:,::n]
        else:
            traces = self.seismic_data

        # Initialize list of zero trace index    
        zero_trace_index = [] 
        
        # Calculate zero trace index and exclude zero traces from seismic data
        for ind in range(traces.shape[1]):
            if np.any(traces[:,ind]) == 0.0 or np.any(traces[:,ind]) == None:
                zero_trace_index.append(ind)
        traces = np.delete(traces, zero_trace_index, 1)
        # Initialize reflection coefficient 
        self.rc_series = np.zeros((traces.shape[0], traces.shape[1]))
        
        # Calculate reflection coefficient of selected traces
        for i in range(traces.shape[1]):
            n_sample = len(traces[:, i])
            sgn = np.sign(np.diff(traces[:, i]))
            for j in range(n_sample-2):
                if sgn[j] != sgn[j+1]:
                    self.rc_series[j+1, i] = traces[j+1, i]
        
        if plot:
            plt.plot(self.rc_series)
            plt.xlabel("RC series")
            plt.ylabel("Amplitude")
            plt.title("Pseudo Reflection Coefficient (RC) from seismic data")
            plt.show()
        

    def get_rc_from_well_log(self, filepath):
        
        """
        Function to calculate Pseudo Reflection Coefficient (RC) from well log

        If the acoustic impedance (AI) log (density * P-velocity ) is available, 
        amplitude spectrum of RC log can be calculated directly from AI log.
        RC calculated from well log produces higher resolution in final BRI output 
        than RC calculated from seismic data.
        
        
        Parameter:
        ----------
        filepath (str): file path of loaded AI file
        
        Returns:
        --------
        RC (numpy.ndarray): Calculated reflection coefficient from AI log
            
        """

        # Loading acoustic impedance log (AI) and assuming there is no header to skip.
        # Time and AI columns are first and second columns, respectively.
        
        log = np.loadtxt(filepath, skiprows=0, usecols=[0,1], converters=None)
        AI = log[:,1]
        n = len(AI)
        
        # Calculate RC from AI log
        def conv_matrx(c,n):
            col = np.hstack([c[0], np.zeros(n-1)])
            row = np.hstack([c, np.zeros(n-1)])
            return toeplitz(col, row)
        
        D = conv_matrx([-1, 1], n)    
        self.RC = D[:,:-1].dot(np.log(AI))    
        self.RC = self.RC.reshape(len(self.RC),1)    
        self.RC = self.RC[1:-1]
        

    def find_f4_index(self, mean_spectrum_normalized):
        
        """   
        Function to compute F4 Index
        
        Find index of min spectrum_real_interp in the second half of array 
        Assuming f1, f2, f3, and f4 as low-cut, low-pass, high-pass,
        and high-cut parameters for a trapezoid shape band-pass filter (ormsby)

        Parameter:
        ----------
        spectrum (numpy.ndarray): Normalized average amplitude spectrum 
        nyquist_frequency (int): Nyquist frequency (HZ)
        
        Returns:
        --------
        f4_ind (numpy.int64): Computed f4 index 
            
        """

        # select a treshold for f4 and calculate f4 index over amplitude spectrum
        try:
            f4 = 0.05
            min_amplitude_ind = np.argwhere(f4 * max(mean_spectrum_normalized) >= mean_spectrum_normalized)
            max_amplitude_ind = np.argwhere(max(mean_spectrum_normalized) == mean_spectrum_normalized)
            self.f4_ind = min_amplitude_ind[min_amplitude_ind > max_amplitude_ind][0]
        except:
            self.f4_ind = int(self.nyquist_frequency/2)
            

    def fit_curve_rc_spectrum(self, mean_spectrum_normalized, method = 'calc intrcpt_grdnt', plot = True):
            
        """   
        Function for fitting a curve on RC amplitude spectrum
        
        Three functions are available for calculating the fitted curve.
        In the first method (calc intrcpt_grdnt), the gradient is calculated based on the maximum amplitude observed in 
        RC amplitude spectrum to create designed operator as if RC wel log is used. Second (linear polyfit) and Third (linear curvefit) 
        methods produce the same results. The higher gradient of fitted curve, the more amplitude is boosted.

        Parameter:
        ----------
        amplitude_spectrum_rc (numpy.ndarray): Amplitude spectrum of RC traces
        mean_spectrum_normalized (numpy.ndarray): Normalized average amplitude spectrum 
        frequency_axis_interpolated (numpy.ndarray): Interpolated frequency axis
        f4_ind (numpy.int64): Computed f4 index 
        method (str): 'calc intrcpt_grdnt', 'linear polyfit', 'linear curvefit'

        Returns:
        --------
        fitted_curve (numpy.ndarray): Calculated fitted curve on RC amplitude spectrum
        
        """
            
        # Get data upto f4 index
        self.frequency_axis_interpolated_upto_f4 = self.frequency_axis_interpolated[1:self.f4_ind]
        self.amplitude_spectrum_rc_upto_f4 = self.amplitude_spectrum_rc[1:self.f4_ind,:]

        # Calculate a fitted curve based on intercept and gradient (recommended) 
        if method == 'calc intrcpt_grdnt':
            
            gradient_list = []
            intercept_list = []
            for i in range(self.amplitude_spectrum_rc_upto_f4.shape[1]):
                intercept = np.log(self.amplitude_spectrum_rc_upto_f4[0,i])
                intercept_list.append(intercept)
                gradient = (np.log(max(self.amplitude_spectrum_rc_upto_f4[:,i]))-np.log(self.amplitude_spectrum_rc_upto_f4[0,i]))\
                /np.log(self.frequency_axis_interpolated_upto_f4[-1]).ravel()
                gradient_list.append(gradient) 
            parameters = [np.mean(gradient_list), np.mean(intercept_list)]  

        # Calculate a fitted curve based on linear polyfit
        elif method == 'linear polyfit':
            
            fit_func_list = []
            n = 1
            for i in range(self.amplitude_spectrum_rc_upto_f4.shape[1]):
                fit = np.polyfit(np.log(self.frequency_axis_interpolated_upto_f4),np.log(self.amplitude_spectrum_rc_upto_f4[:,i]),n)
                fit_func_list.append(fit)
            parameters = np.mean(fit_func_list, axis=0)  
        
        # Calculate a fitted curve based on curve fit function
        elif method == 'linear curvefit':
            
            def func(x, gradient, intercept):
                return gradient*x + intercept
            
            gradient_list = []
            intercept_list = []
            for i in range(self.amplitude_spectrum_rc_upto_f4.shape[1]):
                x = np.log(self.frequency_axis_interpolated_upto_f4)
                y = np.log(self.amplitude_spectrum_rc_upto_f4[:,i])
                params, _ = curve_fit(func, x, y)
                gradient_list.append(params[0])
                intercept_list.append(params[1])
            parameters = [np.mean(gradient_list), np.mean(intercept_list)]   
                    
        print(parameters)
        fitted_curve_log = np.polyval(parameters, np.log(self.frequency_axis_interpolated_upto_f4))
        
        # Fitted curve upto f4 index converted back after logarithmic transformation 
        fit_curve1 = np.exp(fitted_curve_log)
        fit_curve1 = np.append(fit_curve1, fit_curve1[0])
        
        # Add amplitude spectrum from f4 index to nyquist frequency index
        fit_curve2 = mean_spectrum_normalized[self.f4_ind:] * max(fit_curve1)
        self.fitted_curve = np.hstack((fit_curve1, fit_curve2))
        
        if plot:
            # Plot fitted curve and RC amplitude spectrum upto f4 index
            plt.plot(np.log(self.frequency_axis_interpolated_upto_f4), fitted_curve_log, '--', color ='blue', label ="fitted curve")
            plt.plot(np.log(self.frequency_axis_interpolated_upto_f4), np.log(self.amplitude_spectrum_rc_upto_f4))
            plt.xlabel('Frequency (HZ) _ Logarithmic scale')
            plt.ylabel('Amplitude _ Logarithmic scale')
            plt.title('Amplitude Spectrum upto F4 index')
            plt.grid(True)
            plt.legend()
            plt.show()
        

    def bri_operator(self):
        
        """   
        Function to calculate BRI operator
        
        Parameter:
        ----------
        fitted_curve (numpy.ndarray): Calculated fitted curve on RC amplitude spectrum
        frequency_axis_interpolated (numpy.ndarray): Interpolated frequency axis
        nyquist_frequency (int): Nyquist frequency (HZ)

        
        Returns:
        --------
        bri_operator_amplitude (numpy.ndarray): BRI designed operator
        operator_time_axis (numpy.ndarray): Time axis of designed operator
            
        """

        # Flip amplitude spectrum and add it to the original amplitude spectrum
        self.operator_time_axis = np.concatenate((self.frequency_axis_interpolated, self.frequency_axis_interpolated + self.frequency_axis_interpolated[-1]))
        operator_amplitude_spectrum = np.concatenate((self.fitted_curve, self.fitted_curve[::-1]))
        
        # Implement inverse fft 
        operator_amplitude_time = np.fft.ifft(operator_amplitude_spectrum)
        
        # Take fftshift to compute the designed operator
        self.bri_operator_amplitude = np.fft.fftshift(operator_amplitude_time)
        
        # # Limit time axis for display
        # a = int(self.nyquist_frequency - self.nyquist_frequency/2)
        # b = int(self.nyquist_frequency + self.nyquist_frequency/2)
    
        # # Plot designed operator
        # plt.plot(self.operator_time_axis[a:b], self.bri_operator_amplitude[a:b])
        # plt.xlabel('Time(ms)')
        # plt.ylabel('Amplitude')
        # plt.title('BRI Operator Amplitude')
        # plt.grid(True)
        # plt.show()
        

    def convolve_seismic_operator(self, input_seismic_data):

        """   
        Function to convolve seismic traces and BRI operator
        
        Parameter:
        ----------
        seismic_data (numpy.ndarray): All seismic amplitude traces in 2D array
        bri_operator_amplitude (numpy.ndarray): BRI designed operator
        operator_time_axis (numpy.ndarray): Time axis of designed operator

        
        Returns:
        --------
        bri_data (numpy.ndarray): BRI seismic data based on Blueing Reflectivity Integration Technique  
        
        """

        # Initialize BRI output
        bri_data = np.empty((input_seismic_data.shape[0], input_seismic_data.shape[1]))
        
        # Convolve the designed operator to the seismic data to produce final BRI output
        for i in range(input_seismic_data.shape[1]):
            
            if np.any(input_seismic_data[:,i]) != 0.0: 
                
                if len(self.bri_operator_amplitude) > len(input_seismic_data[:, i]):
                    index = int((len(self.bri_operator_amplitude) - len(input_seismic_data[:, i]))/2)
                    bri_temp = np.convolve(self.bri_operator_amplitude[index:-index] / len(self.operator_time_axis[index:-index]), input_seismic_data[:, i],'same').real
                    try:
                        bri = bri_temp[1:] 
                        # Normalize the BRI trace's amplitude with seismic trace amplitude
                        bri_data[:, i] = bri * (max(input_seismic_data[:,i]) / max(bri))
                    except:
                        bri = bri_temp
                        bri = np.delete(bri, 0)
                        bri = np.append(bri, bri[-1]) 
                        # Normalize the BRI trace's amplitude with seismic trace amplitude
                        bri_data[:, i] = bri * (max(input_seismic_data[:,i]) / max(bri))
                
                else:
                    bri = np.convolve(self.bri_operator_amplitude / len(self.operator_time_axis), input_seismic_data[:, i], 'same').real
                    bri = np.delete(bri,0)
                    bri = np.append(bri,bri[-1])  
                    # Normalize the BRI trace's amplitude with seismic trace amplitude
                    bri_data[:,i] = bri * (max(input_seismic_data[:,i]) / max(bri))
            
            else: 
                bri_data[:,i] = input_seismic_data[:,i]
                
        return bri_data
    

    def compare_input_output_1d(self, start_time, end_time):
        
        """   
        Function to compare seismic and BRI data at one location
        A simple overlay comparison between original seismic and BRI at selected location
        Start_time and end_time are in ms

        Parameter:
        ----------
        data_display (numpy.ndarray): Original seismic amplitude traces for display
        bri_data_display (numpy.ndarray): BRI seismic amplitude traces for display
        start_time (numpy.float64): Selected starting time; Default is time of first sample
        end_time (numpy.float64): Selected ending time; Default is time of last sample
        trace_no (int): Selected trace number; Default is trace in the middle of 2D line or center of 3D survey

        Returns:
        --------
        Plot selected seismic amplitude traces (Original and BRI)
        """

        try:
            start_index = int((start_time - self.twt[0]) / self.sample_rate)
            end_index = int((end_time - self.twt[0]) / self.sample_rate)
            Time_axis = np.linspace(start_time, end_time, end_index - start_index)
            trace_no = int(np.median(self.cdp_no))

            plt.plot(self.data_display[start_index:end_index, trace_no], Time_axis, label='Original', color = 'orange')
            plt.plot(self.bri_data_display[start_index:end_index, trace_no], Time_axis, label='BRI', color = 'blue')
            plt.gca().invert_yaxis()
            plt.xlim([-max(self.data_display[:, trace_no]), max(self.data_display[:, trace_no])])
            plt.xlabel('Amplitude')
            plt.ylabel('Time axis (ms)')
            plt.title('Overlay comparison of seismic amplitude traces in the middle of 2D line\n'
                       'or center of 3D survey')
            plt.grid(True)
            plt.legend(loc = 'upper right')
            plt.show()
            
        except:
            print('Please make sure start_time and end_time are selected from {0} to {1} ms, also trace_no is between {2} and {3}'
                .format(self.twt[0], self.twt[-1], self.cdp_no[0], self.cdp_no[-1]))
            

    def export_seismic_segy(self, filepath_in, segyout):
        
        """   
        Function to export 2D/3D post-stack BRI data (segy file)
        
        Parameter:
        ----------
        filepath_in (str): file path of loaded segy file
        seismic_data (numpy.ndarray): BRI seismic data 
        sample_rate (float): sample rate of seismic data in ms
        twt (numpy.ndarray): array of TWT number
        
        Returns:
        --------
        filepath_out (str): file path of export BRI segy file
        """

        # Specify output name with the same parameters of input segy 
        base_filepath = os.path.split(filepath_in)[0]
        base_filename = os.path.splitext(os.path.basename(filepath_in))[0]                
        self.filepath_out = base_filepath + '/' + base_filename + '_BRI.sgy'
        segyio.tools.from_array(self.filepath_out, segyout.T, dt = self.sample_rate * 1e3, delrt = min(self.twt), format = segyio.SegySampleFormat(1))


    def write_segy_byte_locations(self, filepath_out):
        
        """   
        Function for writing header information on segy output (Geometry-related byte locations)
        
        Parameter:
        ----------
        filepath_in (str): file path of loaded segy file
        filepath_out (str): file path of export BRI segy file
        
        Returns:
        --------
        Update header information and Geometry of export BRI segy file based on loaded segy file   
        """
        # Write the same input geometry over the output segy
        with segyio.open(filepath_out, mode='r+') as f:
            try:
                for i, val in enumerate(f.header):
                    val.update({segyio.TraceField.TRACE_SEQUENCE_FILE: int(self.TraceSequenceFile[i].item())})
                    val.update({segyio.TraceField.FieldRecord: int(self.Field_Record[i].item())})
                    val.update({segyio.TraceField.TraceNumber: int(self.Trace_Field[i].item())})
                    val.update({segyio.TraceField.CDP: int(self.CDP[i].item())})
                    val.update({segyio.TraceField.CDP_TRACE: int(self.Trace_Number[i].item())})
                    val.update({segyio.TraceField.SourceGroupScalar: int(self.Scaler_coordinate[i].item())})
                    val.update({segyio.TraceField.SourceX: int(self.Source_X[i].item())})
                    val.update({segyio.TraceField.SourceY: int(self.Source_Y[i].item())})
                    val.update({segyio.TraceField.CDP_X: int(self.CDP_X[i].item())})
                    val.update({segyio.TraceField.CDP_Y: int(self.CDP_Y[i].item())})
                    val.update({segyio.TraceField.INLINE_3D: int(self.Inline_3D[i].item())})
                    val.update({segyio.TraceField.CROSSLINE_3D: int(self.Crossline_3D[i].item())})
            except ValueError:
                print('something went wrong during header writing')
        

if __name__ == '__main__':

    bri_spectral_class = BRI()
    print('BRI spectral enhancement started')


     