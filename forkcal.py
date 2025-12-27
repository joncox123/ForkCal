#!/usr/bin/env python3
"""
ForkCal Tuning Fork Watch Timegrapher

For the adjustment of tuning fork watches, such as the Bulova Accutron and Omega f300 Hz watches.
Developed by joncox123, all rights reserved.

Main GUI application using tkinter and matplotlib
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator
import matplotlib.pyplot as plt

# Import analysis module (to be created)
from analysis import AudioAnalyzer


class SpectrumAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ForkCal Tuning Fork Watch Timegrapher")
        self.root.geometry("1200x1080")

        # Audio analyzer instance
        self.analyzer = None
        self.is_running = False

        # Data for plotting
        self.frequencies = None
        self.psd_db = None

        # Timegrapher data (initialize with 100 zeros)
        self.time_data = list(range(100))
        self.deviation_data = [0.0] * 100
        self.deviation_data_raw = [0.0] * 100  # Raw data before moving average
        self.current_time = 0.0
        self.current_freq = None
        self.last_timegrapher_freq = None  # Track last reading to detect new data
        self.data_index = 0  # Current position in rolling buffer

        # Filter parameters for timegrapher
        self.filter_lowcut = None
        self.filter_highcut = None
        self.filter_center = None

        # Debug plot window
        self.debug_window = None
        self.debug_fig = None
        self.debug_canvas = None
        self.debug_enabled = False  # Track if debug plots are active
        self.first_plot_after_start = False  # Track if this is first plot after acquisition starts

        # Create GUI elements
        self.create_controls()
        self.create_plot()
        self.create_info_display()

        # Initialize device list
        self.update_device_list()

    def create_controls(self):
        """Create control panel with all settings"""
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Microphone selection
        ttk.Label(control_frame, text="Microphone:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(control_frame, textvariable=self.mic_var, width=40, state='readonly')
        self.mic_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.mic_combo.bind('<<ComboboxSelected>>', self.on_device_changed)

        # Sampling rate selection
        ttk.Label(control_frame, text="Sample Rate (Hz):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.sample_rate_var = tk.StringVar(value="48000")
        sample_rates = ["8000", "16000", "22050", "44100", "48000", "96000", "192000"]
        self.sample_rate_combo = ttk.Combobox(control_frame, textvariable=self.sample_rate_var,
                                              values=sample_rates, width=20, state='readonly')
        self.sample_rate_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Acquisition period selection
        ttk.Label(control_frame, text="Acquisition Period:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.acq_period_var = tk.StringVar(value="250 ms")
        acq_periods = [
            "250 ms", "500 ms", "1 s", "2 s", "5 s", "10 s", "20 s", "60 s"
        ]
        self.acq_period_combo = ttk.Combobox(control_frame, textvariable=self.acq_period_var,
                                             values=acq_periods, width=20, state='readonly')
        self.acq_period_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Reference frequency selection
        ttk.Label(control_frame, text="Reference Frequency (Hz):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.ref_freq_var = tk.StringVar(value="720")
        ref_freqs = ["300", "360", "600", "720"]
        self.ref_freq_combo = ttk.Combobox(control_frame, textvariable=self.ref_freq_var,
                                           values=ref_freqs, width=20, state='readonly')
        self.ref_freq_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        self.ref_freq_combo.bind('<<ComboboxSelected>>', self.on_ref_freq_changed)

        # Frequency estimation method selection
        ttk.Label(control_frame, text="Frequency Estimation:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.freq_method_var = tk.StringVar(value="Instantaneous phase fit")
        freq_methods = ["Sine best fit", "Instantaneous phase fit"]
        self.freq_method_combo = ttk.Combobox(control_frame, textvariable=self.freq_method_var,
                                              values=freq_methods, width=20, state='readonly')
        self.freq_method_combo.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)

        # Resolution Bandwidth (right side, row 0)
        ttk.Label(control_frame, text="Resolution Bandwidth:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.rbw_var = tk.StringVar(value="RBW: N/A")
        self.rbw_label = ttk.Label(control_frame, textvariable=self.rbw_var, font=('TkDefaultFont', 10, 'bold'))
        self.rbw_label.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Estimated Frequency (right side, row 1)
        ttk.Label(control_frame, text="Estimated Frequency:").grid(row=1, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.freq_var = tk.StringVar(value="Freq: N/A")
        self.freq_label = ttk.Label(control_frame, textvariable=self.freq_var, font=('TkDefaultFont', 10, 'bold'))
        self.freq_label.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Timegrapher Statistics (right side, row 2)
        ttk.Label(control_frame, text="Timegrapher Stats:").grid(row=2, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.stats_var = tk.StringVar(value="N/A")
        self.stats_label = ttk.Label(control_frame, textvariable=self.stats_var, font=('TkDefaultFont', 10, 'bold'))
        self.stats_label.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)

        # Start/Stop button
        self.start_stop_btn = ttk.Button(control_frame, text="Start", command=self.toggle_acquisition, width=15)
        self.start_stop_btn.grid(row=5, column=0, columnspan=2, pady=10)

        # Debug plots button
        self.debug_btn = ttk.Button(control_frame, text="Show Debug Plots", command=self.toggle_debug_window, width=15)
        self.debug_btn.grid(row=5, column=2, padx=10, pady=10)

        # Status label
        self.status_var = tk.StringVar(value="Status: Stopped")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, foreground="red", font=('TkDefaultFont', 10))
        self.status_label.grid(row=6, column=0, columnspan=4, pady=5)

    def create_plot(self):
        """Create matplotlib plot with spectrum analyzer and timegrapher"""
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create figure with two subplots side by side
        self.fig = Figure(figsize=(14, 6), dpi=100)
        self.ax_spectrum = self.fig.add_subplot(121)
        self.ax_time = self.fig.add_subplot(122)

        # Configure spectrum analyzer plot (semi-log: dB vs log Hz)
        self.ax_spectrum.set_xscale('log')
        self.ax_spectrum.set_xlabel('Frequency (Hz)')
        self.ax_spectrum.set_ylabel('Power Spectral Density (dB)')
        self.ax_spectrum.set_title('Real-time Spectrum Analyzer')

        # Enable fine grid for spectrum plot
        self.ax_spectrum.grid(True, which='major', alpha=0.5, linewidth=0.8)
        self.ax_spectrum.grid(True, which='minor', alpha=0.2, linewidth=0.5)

        # Custom formatter for x-axis to show Hz without scientific notation
        def freq_formatter(x, pos):
            return f'{x:.1f}'

        self.ax_spectrum.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
        self.ax_spectrum.xaxis.set_minor_formatter(FuncFormatter(freq_formatter))

        # Use LogLocator with more subdivisions for better tick placement
        self.ax_spectrum.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
        self.ax_spectrum.xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=15))

        # Rotate x-axis labels by 60 degrees to save horizontal space (both major and minor)
        plt.setp(self.ax_spectrum.xaxis.get_majorticklabels(), rotation=60, ha='right')
        plt.setp(self.ax_spectrum.xaxis.get_minorticklabels(), rotation=60, ha='right')

        # Initialize spectrum line
        self.line_spectrum, = self.ax_spectrum.plot([], [], 'b-', linewidth=1.5)

        # Initialize filter cutoff lines (will be updated from timegrapher data)
        self.vline_lowcut = self.ax_spectrum.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0, label='Low cutoff')
        self.vline_highcut = self.ax_spectrum.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0, label='High cutoff')
        self.vline_center = self.ax_spectrum.axvline(x=0, color='g', linestyle='--', linewidth=1, alpha=0, label='Center')

        # Set initial limits for spectrum
        self.ax_spectrum.set_xlim(10, 24000)
        self.ax_spectrum.set_ylim(-100, -20)

        # Configure timegrapher plot (linear)
        self.ax_time.set_xlabel('Time [s]')
        self.ax_time.set_ylabel('Deviation [spd]')
        self.ax_time.set_title('Timegrapher')

        # Enable grid for timegrapher plot
        self.ax_time.grid(True, which='major', alpha=0.5, linewidth=0.8)
        self.ax_time.grid(True, which='minor', alpha=0.2, linewidth=0.5)

        # Initialize timegrapher scatter plot and trend line
        self.line_time = self.ax_time.scatter([], [], c='red', s=30, alpha=0.8)
        self.line_trend, = self.ax_time.plot([], [], 'b-.', linewidth=2, label='Trend')

        # Initialize group average annotations (5 groups)
        self.group_annotations = []
        for i in range(5):
            ann = self.ax_time.text(0, 0, '', ha='center', va='top', fontsize=9,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            ann.set_visible(False)
            self.group_annotations.append(ann)

        # Set initial limits for timegrapher
        self.ax_time.set_xlim(0, 100)
        self.ax_time.set_ylim(-10, 10)

        # Adjust subplot spacing
        self.fig.tight_layout()

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()

        # Add matplotlib toolbar for zoom/pan
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Pack canvas after toolbar to get correct layout
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_info_display(self):
        """Create info display - placeholder for compatibility"""
        pass

    def update_device_list(self):
        """Update the list of available audio devices"""
        try:
            from analysis import get_audio_devices
            devices = get_audio_devices()
            self.mic_combo['values'] = devices
            if devices:
                self.mic_combo.current(0)
                # Update sample rates for initial device
                self.update_sample_rates()
        except Exception as e:
            self.mic_combo['values'] = [f"Error: {str(e)}"]
            self.mic_combo.current(0)

    def on_device_changed(self, event):
        """Called when device selection changes"""
        self.update_sample_rates()

    def on_ref_freq_changed(self, event):
        """Called when reference frequency selection changes"""
        if self.analyzer:
            ref_freq = float(self.ref_freq_var.get())
            self.analyzer.set_reference_frequency(ref_freq)

    def update_sample_rates(self):
        """Update sample rate dropdown based on selected device"""
        device_name = self.mic_var.get()
        if not device_name or device_name.startswith("Error:"):
            return

        try:
            from analysis import get_supported_sample_rates

            # Get supported sample rates for this device
            supported_rates = get_supported_sample_rates(device_name)

            if supported_rates:
                # Update combobox with supported rates (as strings)
                rate_strings = [str(rate) for rate in supported_rates]
                self.sample_rate_combo['values'] = rate_strings

                # Try to keep current selection if it's still valid
                current_rate = self.sample_rate_var.get()
                if current_rate in rate_strings:
                    self.sample_rate_var.set(current_rate)
                else:
                    # Default to 48000 if available, otherwise first in list
                    if "48000" in rate_strings:
                        self.sample_rate_var.set("48000")
                    else:
                        self.sample_rate_var.set(rate_strings[0])
            else:
                # No supported rates found, keep default list
                self.sample_rate_combo['values'] = ["8000", "16000", "22050", "44100", "48000", "96000", "192000"]

        except Exception as e:
            print(f"Error updating sample rates: {e}")

    def parse_acquisition_period(self):
        """Parse acquisition period string to seconds"""
        period_str = self.acq_period_var.get()
        value, unit = period_str.split()
        value = float(value)

        if unit == 'us':
            return value * 1e-6
        elif unit == 'ms':
            return value * 1e-3
        elif unit == 's':
            return value
        else:
            return 0.1  # Default 100ms

    def toggle_acquisition(self):
        """Start or stop the acquisition"""
        if not self.is_running:
            self.start_acquisition()
        else:
            self.stop_acquisition()

    def start_acquisition(self):
        """Start audio acquisition and analysis"""
        try:
            # Get parameters
            device_name = self.mic_var.get()
            sample_rate = int(self.sample_rate_var.get())
            acq_period = self.parse_acquisition_period()
            ref_freq = float(self.ref_freq_var.get())

            # Map GUI selection to method name
            freq_method_gui = self.freq_method_var.get()
            freq_method = 'sine_fit' if freq_method_gui == 'Sine best fit' else 'phase_fit'

            # Create analyzer instance (no averaging)
            self.analyzer = AudioAnalyzer(
                device_name=device_name,
                sample_rate=sample_rate,
                acquisition_period=acq_period,
                num_averages=1,
                reference_freq=ref_freq,
                freq_estimation_method=freq_method
            )

            # Start analyzer
            self.analyzer.start()

            # Set debug mode if debug window is open
            if self.debug_enabled:
                self.analyzer.set_debug_mode(True)

            # Reset timegrapher data (keep 100 zero initialization)
            # Initialize time axis with proper spacing based on acquisition period
            self.time_data = [i * acq_period for i in range(100)]
            self.deviation_data = [0.0] * 100
            self.deviation_data_raw = [0.0] * 100
            self.current_time = 0.0
            self.last_timegrapher_freq = None
            self.data_index = 0
            self.acq_period = acq_period  # Store for later use

            # Update UI
            self.is_running = True
            self.start_stop_btn.config(text="Stop")
            self.status_var.set("Status: Running")
            self.status_label.config(foreground="green")

            # Disable controls during acquisition
            self.mic_combo.config(state='disabled')
            self.sample_rate_combo.config(state='disabled')
            self.acq_period_combo.config(state='disabled')
            self.ref_freq_combo.config(state='disabled')
            self.freq_method_combo.config(state='disabled')

            # Set flag to configure xlims on first plot update
            self.first_plot_after_start = True

            # Start update loop
            self.update_plot()

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.status_label.config(foreground="red")

    def stop_acquisition(self):
        """Stop audio acquisition and analysis"""
        try:
            if self.analyzer:
                self.analyzer.stop()
                self.analyzer = None

            # Update UI
            self.is_running = False
            self.start_stop_btn.config(text="Start")
            self.status_var.set("Status: Stopped")
            self.status_label.config(foreground="red")

            # Re-enable controls
            self.mic_combo.config(state='readonly')
            self.sample_rate_combo.config(state='readonly')
            self.acq_period_combo.config(state='readonly')
            self.ref_freq_combo.config(state='readonly')
            self.freq_method_combo.config(state='readonly')

        except Exception as e:
            self.status_var.set(f"Error stopping: {str(e)}")

    def update_plot(self, N_moving_avg=10):
        """Update the plot with new data"""
        if not self.is_running:
            return

        try:
            # Get spectrum data from analyzer
            frequencies, psd_db = self.analyzer.get_spectrum()

            if frequencies is not None and psd_db is not None:
                # Update spectrum line data
                self.line_spectrum.set_data(frequencies, psd_db)

                # Store for click events
                self.frequencies = frequencies
                self.psd_db = psd_db

                # Update RBW display
                rbw = self.analyzer.compute_rbw()
                if rbw is not None:
                    if rbw >= 1000:
                        self.rbw_var.set(f"RBW: {rbw/1000:.2f} kHz")
                    else:
                        self.rbw_var.set(f"RBW: {rbw:.2f} Hz")

            # Get timegrapher data from analyzer
            freq_estimate, deviation_spd = self.analyzer.get_timegrapher_data()

            # Only update if we have NEW data (check if frequency changed)
            if freq_estimate is not None and deviation_spd is not None:
                # Check if this is new data by comparing to last reading
                if freq_estimate != self.last_timegrapher_freq:
                    # Update timegrapher data using rolling buffer
                    self.time_data[self.data_index] = self.current_time
                    self.deviation_data_raw[self.data_index] = deviation_spd

                    # Apply N-point backward-looking moving average ONLY to the new point
                    # Average this point with the previous 9 points (backward-looking)
                    window_values = []
                    for j in range(N_moving_avg):
                        idx = (self.data_index - j) % 100
                        window_values.append(self.deviation_data_raw[idx])
                    avg = np.mean(window_values)
                    self.deviation_data[self.data_index] = avg

                    # Increment index (wrap around at 100)
                    self.data_index = (self.data_index + 1) % 100

                    # Update timegrapher scatter plot
                    self.line_time.set_offsets(np.column_stack((self.time_data, self.deviation_data)))

                    # Compute averages for 5 groups of 20 points each
                    for group_idx in range(5):
                        start_idx = group_idx * 20
                        end_idx = start_idx + 20
                        data = self.deviation_data[start_idx:end_idx]
                        # ignore non-populated data
                        if np.any(data == 0.0):
                            group_avg = np.nan
                        else:
                            group_avg = np.mean(data)

                        # Update annotation for this group
                        # Position at center of group window, near top of plot
                        center_time_idx = start_idx + 10
                        x_pos = self.time_data[center_time_idx]

                        # Get current y-axis limits to position annotation near top
                        y_min, y_max = self.ax_time.get_ylim()
                        y_pos = y_max * 0.95  # 95% up from bottom

                        # Update annotation text and position
                        if not np.isnan(group_avg):
                            self.group_annotations[group_idx].set_text(f'{group_avg:.1f}')
                            self.group_annotations[group_idx].set_position((x_pos, y_pos))
                            self.group_annotations[group_idx].set_visible(True)
                        else:
                            self.group_annotations[group_idx].set_visible(False)

                    # Clear any existing trend line
                    self.line_trend.set_data([], [])

                    # Update x-axis limits for rolling display
                    if len(self.time_data) > 0:
                        time_range = max(self.time_data) - min(self.time_data)
                        if time_range > 0:
                            self.ax_time.set_xlim(min(self.time_data), max(self.time_data))

                    # Auto-scale y-axis to fixed ranges centered at zero
                    if len(self.deviation_data) > 0:
                        y_max_abs = max(abs(min(self.deviation_data)), abs(max(self.deviation_data)))
                        # Choose smallest range that fits the data
                        ranges = [5, 10, 25, 50, 100, 200]
                        y_range = 200  # Default to largest
                        for r in ranges:
                            if y_max_abs <= r:
                                y_range = r
                                break
                        self.ax_time.set_ylim(-y_range, y_range)

                    # Update frequency display and tracking variables
                    self.freq_var.set(f"Freq: {freq_estimate:.6f} Hz")
                    self.current_freq = freq_estimate
                    self.last_timegrapher_freq = freq_estimate
                    self.current_time += self.parse_acquisition_period()

                    # Calculate and update timegrapher statistics from raw data
                    # Only include non-zero entries (populated data points)
                    raw_data = np.array(self.deviation_data_raw)
                    valid_data = raw_data[raw_data != 0.0]

                    if len(valid_data) > 0:
                        mean_spd = np.mean(valid_data)
                        std_spd = np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0.0
                        self.stats_var.set(f"{mean_spd:.2f} ± {std_spd:.2f} spd")
                    else:
                        self.stats_var.set("N/A")

            # Get filter parameters and update spectrum plot x-limits
            if self.analyzer:
                lowcut, highcut, center = self.analyzer.get_filter_params()
                if center is not None:
                    # Update vertical line positions and make visible
                    self.vline_lowcut.set_xdata([lowcut])
                    self.vline_lowcut.set_alpha(0.6)
                    self.vline_highcut.set_xdata([highcut])
                    self.vline_highcut.set_alpha(0.6)
                    self.vline_center.set_xdata([center])
                    self.vline_center.set_alpha(0.6)

                    # Set spectrum plot x-limits only on first plot after acquisition starts
                    # Range: 0.4 * f_c to 2.5 * f_c
                    if self.first_plot_after_start:
                        self.ax_spectrum.set_xlim(0.4 * center, 2.5 * center)

            # Check signal quality and update plot titles and status
            if self.analyzer:
                signal_quality_good = self.analyzer.get_signal_quality()
                if signal_quality_good:
                    # Good signal - normal titles and status
                    self.ax_spectrum.set_title("Spectrum Analyzer", color='black', fontweight='normal')
                    self.ax_time.set_title("Timegrapher", color='black', fontweight='normal')
                    self.status_var.set("Status: Running")
                    self.status_label.config(foreground="green", font=('TkDefaultFont', 10))
                else:
                    # Poor signal - red bold titles and status
                    self.ax_spectrum.set_title("Spectrum Analyzer", color='red', fontweight='bold')
                    self.ax_time.set_title("Timegrapher", color='red', fontweight='bold')
                    self.status_var.set("Status: Running; Signal not found!")
                    self.status_label.config(foreground="red", font=('TkDefaultFont', 10, 'bold'))

            # Check for debug plot data
            if self.analyzer:
                debug_data = self.analyzer.get_debug_plot_data()
                if debug_data is not None:
                    self.update_debug_plots(debug_data)

            # Redraw canvas
            self.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating plot: {e}")

        # Schedule next update (approximately 30 fps)
        if self.is_running:
            self.root.after(33, self.update_plot)

    def toggle_debug_window(self):
        """Toggle debug plots window"""
        if self.debug_window is None or not tk.Toplevel.winfo_exists(self.debug_window):
            self.create_debug_window()
            self.debug_enabled = True
            # Enable debug mode in analyzer if it's running
            if self.analyzer:
                self.analyzer.set_debug_mode(True)
        else:
            self.debug_window.destroy()
            self.debug_window = None
            self.debug_fig = None
            self.debug_canvas = None
            self.debug_enabled = False
            # Disable debug mode in analyzer if it's running
            if self.analyzer:
                self.analyzer.set_debug_mode(False)

    def create_debug_window(self):
        """Create debug plots window"""
        self.debug_window = tk.Toplevel(self.root)
        self.debug_window.title("Timegrapher Debug Plots")
        self.debug_window.geometry("1400x900")

        # Create figure with three subplots
        self.debug_fig = Figure(figsize=(14, 9), dpi=100)

        # Filter response subplot (top)
        self.ax_filter = self.debug_fig.add_subplot(311)
        self.ax_filter.set_title('Bandpass Filter Frequency Response')
        self.ax_filter.set_xlabel('Frequency [Hz]')
        self.ax_filter.set_ylabel('Magnitude [dB]')
        self.ax_filter.grid(True, alpha=0.3)

        # Filtered signal subplot (middle)
        self.ax_filtered = self.debug_fig.add_subplot(312)
        self.ax_filtered.set_title('Bandpass Filtered Signal (Time Domain)')
        self.ax_filtered.set_xlabel('Time [s]')
        self.ax_filtered.set_ylabel('Amplitude')
        self.ax_filtered.grid(True, alpha=0.3)

        # Phase residuals subplot (bottom) - only used for phase_fit method
        self.ax_phase = self.debug_fig.add_subplot(313)
        self.ax_phase.set_title('Phase Fit Residuals')
        self.ax_phase.set_xlabel('Time [s]')
        self.ax_phase.set_ylabel('Phase Residual [rad]')
        self.ax_phase.grid(True, alpha=0.3)

        self.debug_fig.tight_layout()

        # Create canvas
        self.debug_canvas = FigureCanvasTkAgg(self.debug_fig, master=self.debug_window)
        self.debug_canvas.draw()
        self.debug_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.debug_canvas, self.debug_window)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_debug_plots(self, debug_data):
        """Update debug plots with new data"""
        if self.debug_window is None or not tk.Toplevel.winfo_exists(self.debug_window):
            return

        try:
            # Clear previous plots
            self.ax_filter.clear()
            self.ax_filtered.clear()
            self.ax_phase.clear()

            # Plot 1: Filter frequency response
            if 'filter_freq' in debug_data and 'filter_h' in debug_data:
                w = debug_data['filter_freq']
                h = debug_data['filter_h']
                lowcut = debug_data.get('lowcut', 0)
                highcut = debug_data.get('highcut', 0)
                center_freq = debug_data.get('center_freq', 0)

                self.ax_filter.plot(w, 20 * np.log10(abs(h)), 'b', linewidth=1.5)
                self.ax_filter.axvline(lowcut, color='r', linestyle='--', label=f'Low cutoff: {lowcut:.2f} Hz')
                self.ax_filter.axvline(highcut, color='r', linestyle='--', label=f'High cutoff: {highcut:.2f} Hz')
                self.ax_filter.axvline(center_freq, color='g', linestyle='--', label=f'Center: {center_freq:.2f} Hz')
                self.ax_filter.set_title('Bandpass Filter Magnitude Response')
                self.ax_filter.set_xlabel('Frequency [Hz]')
                self.ax_filter.set_ylabel('Magnitude [dB]')
                self.ax_filter.grid(True, alpha=0.3)
                self.ax_filter.legend()

                # Set x-limits only on first plot after acquisition starts
                # Range: 0.4 * f_c to 2.5 * f_c
                if self.first_plot_after_start:
                    self.ax_filter.set_xlim(0.4 * center_freq, 2.5 * center_freq)

            # Plot 2: Filtered signal in time domain with fitted sine wave overlay
            if 'time' in debug_data and 'x_filtered' in debug_data:
                t = debug_data['time']
                x_filtered = debug_data['x_filtered']

                self.ax_filtered.plot(t, x_filtered, 'b', linewidth=0.8, label='Full Signal')

                # Overlay fitted signal if available
                if 't_cropped_fit' in debug_data and 'fitted_signal' in debug_data:
                    t_crop = debug_data['t_cropped_fit']
                    fitted_signal = debug_data['fitted_signal']
                    estimated_freq = debug_data.get('estimated_freq', 0)
                    deviation_spd = debug_data.get('deviation_spd', 0)
                    fit_method = debug_data.get('fit_method', 'sine_fit')

                    self.ax_filtered.plot(t_crop, fitted_signal, 'r-.', linewidth=2, label='Fitted Signal')

                    # Add parameter text box - content depends on fit method
                    if fit_method == 'sine_fit':
                        A_fit = debug_data.get('A_fit', 0)
                        f_fit = debug_data.get('f_fit', 0)
                        phi_fit = debug_data.get('phi_fit', 0)

                        param_text = f'Sine Fit Parameters:\n'
                        param_text += f'A = {A_fit:.6f}\n'
                        param_text += f'f = {f_fit:.6f} Hz\n'
                        param_text += f'φ = {phi_fit:.6f} rad ({np.degrees(phi_fit):.2f}°)\n'
                        param_text += f'\nEstimated Frequency = {estimated_freq:.6f} Hz\n'
                        param_text += f'Deviation = {deviation_spd:.3f} s/day'
                    else:  # phase_fit
                        param_text = f'Phase Fit Parameters:\n'
                        param_text += f'Method: Instantaneous Phase\n'
                        param_text += f'\nEstimated Frequency = {estimated_freq:.6f} Hz\n'
                        param_text += f'Deviation = {deviation_spd:.3f} s/day'

                    self.ax_filtered.text(0.02, 0.98, param_text, transform=self.ax_filtered.transAxes,
                                         verticalalignment='top', fontfamily='monospace',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                title = 'Bandpass Filtered Signal with Sine Fit' if debug_data.get('fit_method') == 'sine_fit' else 'Bandpass Filtered Signal with Phase Fit'
                self.ax_filtered.set_title(title)
                self.ax_filtered.set_xlabel('Time [s]')
                self.ax_filtered.set_ylabel('Amplitude')
                self.ax_filtered.grid(True, alpha=0.3)
                self.ax_filtered.legend()

            # Plot 3: Phase residuals (only for phase_fit method)
            fit_method = debug_data.get('fit_method', 'sine_fit')
            if fit_method == 'phase_fit' and 'phase_residuals' in debug_data and 't_cropped_fit' in debug_data:
                t_crop = debug_data['t_cropped_fit']
                phase_residuals = debug_data['phase_residuals']

                # Plot phase residuals
                self.ax_phase.plot(t_crop, phase_residuals, 'r-', linewidth=1.0, label='Phase Residuals')
                self.ax_phase.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

                # Calculate RMS of residuals for display
                rms_residual = np.sqrt(np.mean(phase_residuals**2))

                # Add statistics text box
                stats_text = f'Phase Fit Quality:\n'
                stats_text += f'RMS Residual = {rms_residual:.6f} rad\n'
                stats_text += f'RMS Residual = {np.degrees(rms_residual):.4f}°'

                self.ax_phase.text(0.98, 0.98, stats_text, transform=self.ax_phase.transAxes,
                                  verticalalignment='top', horizontalalignment='right',
                                  fontfamily='monospace',
                                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

                self.ax_phase.set_title('Phase Fit Residuals (Unwrapped Phase - Linear Fit)')
                self.ax_phase.set_xlabel('Time [s]')
                self.ax_phase.set_ylabel('Phase Residual [rad]')
                self.ax_phase.grid(True, alpha=0.3)
                self.ax_phase.legend()
            else:
                # For sine_fit method, show a message or leave blank
                self.ax_phase.text(0.5, 0.5, 'Phase residuals only available\nfor Instantaneous Phase Fit method',
                                  transform=self.ax_phase.transAxes,
                                  horizontalalignment='center', verticalalignment='center',
                                  fontsize=12, style='italic', color='gray')
                self.ax_phase.set_title('Phase Fit Residuals')
                self.ax_phase.set_xlabel('Time [s]')
                self.ax_phase.set_ylabel('Phase Residual [rad]')
                self.ax_phase.grid(True, alpha=0.3)

            self.debug_fig.tight_layout()
            self.debug_canvas.draw_idle()

            # Reset flag after first plot update
            if self.first_plot_after_start:
                self.first_plot_after_start = False

        except Exception as e:
            print(f"Error updating debug plots: {e}")

    def on_close(self):
        """Clean up when closing the window"""
        if self.is_running:
            self.stop_acquisition()
        if self.debug_window is not None:
            self.debug_window.destroy()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SpectrumAnalyzerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
