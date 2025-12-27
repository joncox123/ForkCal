"""
Audio Analysis Module

ForkCal Tuning Fork Watch Timegrapher

For the adjustment of tuning fork watches, such as the Bulova Accutron and Omega f300 Hz watches.
Developed by joncox123, all rights reserved.
"""

import numpy as np
import pyaudio
from scipy import signal
from scipy.optimize import curve_fit
from collections import deque
import threading
import queue


class PoorFitError(Exception):
    """Exception raised when sine wave fit quality is too poor"""
    pass


def get_audio_devices():
    """
    Get list of available audio input devices
    Returns list of strings with device names
    """
    p = pyaudio.PyAudio()
    devices = []

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # Only add input devices (with input channels > 0)
        if info['maxInputChannels'] > 0:
            device_str = f"{i}: {info['name']} ({info['maxInputChannels']} ch, {int(info['defaultSampleRate'])} Hz)"
            devices.append(device_str)

    p.terminate()
    return devices


def get_supported_sample_rates(device_name, test_rates=None):
    """
    Test which sample rates are supported by a given device

    Parameters:
    -----------
    device_name : str
        Name of the audio device (from get_audio_devices)
    test_rates : list of int, optional
        List of sample rates to test. If None, uses default list.

    Returns:
    --------
    supported_rates : list of int
        List of supported sample rates
    """
    if test_rates is None:
        test_rates = [8000, 16000, 22050, 44100, 48000, 96000, 192000]

    # Extract device index from device name string "index: name ..."
    device_index = int(device_name.split(':')[0])

    p = pyaudio.PyAudio()
    supported_rates = []

    for rate in test_rates:
        try:
            # Try to open stream with this sample rate
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
                start=False  # Don't start the stream
            )
            # If successful, close immediately and add to supported list
            stream.close()
            supported_rates.append(rate)
        except Exception:
            # This sample rate is not supported
            pass

    p.terminate()
    return supported_rates


class AudioAnalyzer:
    """
    Real-time audio analyzer using PyAudio and scipy.signal.welch
    """

    def __init__(self, device_name, sample_rate, acquisition_period, num_averages, reference_freq=360.0, freq_estimation_method='sine_fit'):
        """
        Initialize the audio analyzer

        Parameters:
        -----------
        device_name : str
            Name of the audio device (from get_audio_devices)
        sample_rate : int
            Sampling rate in Hz
        acquisition_period : float
            Acquisition period in seconds
        num_averages : int
            Number of spectra to average (moving average)
        reference_freq : float
            Reference frequency in Hz (default 360.0)
        freq_estimation_method : str
            Method for frequency estimation: 'sine_fit' or 'phase_fit' (default 'sine_fit')
        """
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.acquisition_period = acquisition_period
        self.num_averages = num_averages
        self.reference_freq = reference_freq
        self.freq_estimation_method = freq_estimation_method

        # Extract device index from device name string "index: name ..."
        self.device_index = int(device_name.split(':')[0])

        # Calculate buffer parameters
        self.chunk_size = int(self.sample_rate * self.acquisition_period)

        # Ensure chunk size is reasonable (at least 256 samples, power of 2 preferred)
        if self.chunk_size < 256:
            self.chunk_size = 256

        # PyAudio instance and stream
        self.p = None
        self.stream = None

        # Data buffers
        self.audio_buffer = []
        self.spectrum_buffer = deque(maxlen=num_averages)

        # Current spectrum data
        self.frequencies = None
        self.psd_db = None
        self.nperseg = None  # Store nperseg for RBW calculation
        self.lock = threading.Lock()

        # Timegrapher data
        self.timegrapher_freq = None
        self.timegrapher_deviation = None
        self.signal_quality_good = True  # Track if last fit was good

        # Running flag
        self.running = False
        self.analysis_thread = None

        # Debug plot queue (thread-safe communication to GUI)
        self.debug_plot_queue = queue.Queue(maxsize=2)  # Keep only latest debug data

        # Debug mode flag
        self.debug_mode = False

        # Cached filter coefficients for timegrapher
        self.fir_coeff = None
        self.filter_lowcut = None
        self.filter_highcut = None
        self.filter_center_freq = None

    def start(self):
        """Start audio capture and analysis"""
        if self.running:
            return

        # Reset averaging buffer to clear prior measurements
        self.spectrum_buffer.clear()

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )

        # Start the stream
        self.stream.start_stream()
        self.running = True

        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def stop(self):
        """Stop audio capture and analysis"""
        if not self.running:
            return

        self.running = False

        # Wait for analysis thread to finish
        if self.analysis_thread:
            self.analysis_thread.join(timeout=1.0)

        # Stop and close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Terminate PyAudio
        if self.p:
            self.p.terminate()
            self.p = None

        # Clear buffers
        self.audio_buffer = []
        self.spectrum_buffer.clear()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")

        # Convert byte data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Store in buffer
        self.audio_buffer.append(audio_data)

        return (None, pyaudio.paContinue)

    def _analysis_loop(self):
        """Analysis loop running in separate thread"""
        while self.running:
            # Check if we have data to process
            if len(self.audio_buffer) > 0:
                # Get audio data from buffer
                with self.lock:
                    if len(self.audio_buffer) > 0:
                        audio_data = self.audio_buffer.pop(0)
                    else:
                        continue

                # Perform spectrum analysis
                self._analyze_spectrum(audio_data)

                # Perform timegrapher analysis
                self._analyze_timegrapher(audio_data)
            else:
                # Small sleep to avoid busy waiting
                threading.Event().wait(0.001)

    def _analyze_spectrum(self, audio_data, fft_interp_factor=1):
        """
        Analyze spectrum using scipy.signal.welch

        Parameters:
        -----------
        audio_data : numpy array
            Audio samples (int16)
        """
        # Convert to float and normalize
        audio_float = audio_data.astype(np.float64) / 32768.0

        # Apply window to reduce spectral leakage
        # Use Welch's method for power spectral density estimation
        try:
            # Use all available data for maximum frequency resolution
            # nperseg = length of data means single segment (no averaging within Welch)
            nperseg = len(audio_float)//4

            frequencies, psd = signal.welch(
                audio_float,
                fs=self.sample_rate,
                window='hann',
                nperseg=nperseg,
                noverlap=None,
                scaling='density',
                nfft=fft_interp_factor*nperseg # zero pad to interpolate and improve frequency resolution
            )

            # Convert to dB (with floor to avoid log(0))
            psd_db = 10 * np.log10(psd + 1e-12)

            # Add to spectrum buffer for averaging
            self.spectrum_buffer.append(psd_db)

            # Compute moving average
            if len(self.spectrum_buffer) > 0:
                avg_psd_db = np.mean(self.spectrum_buffer, axis=0)

                # Update current spectrum data (thread-safe)
                with self.lock:
                    self.frequencies = frequencies
                    self.psd_db = avg_psd_db
                    self.nperseg = nperseg

        except Exception as e:
            print(f"Error in spectrum analysis: {e}")

    def compute_rbw(self):
        """
        Compute the resolution bandwidth (RBW) of the Welch spectrum

        Returns:
        --------
        rbw : float
            Resolution bandwidth in Hz
        """
        with self.lock:
            if self.nperseg is not None:
                # For Hann window, the equivalent noise bandwidth (ENBW) factor is 1.5
                # RBW = (sample_rate / nperseg) * ENBW_factor
                enbw_factor = 1.5
                rbw = (self.sample_rate / self.nperseg) * enbw_factor
                return rbw
            else:
                return None

    def get_spectrum(self):
        """
        Get current spectrum data

        Returns:
        --------
        frequencies : numpy array
            Frequency values in Hz
        psd_db : numpy array
            Power spectral density in dB
        """
        with self.lock:
            if self.frequencies is not None and self.psd_db is not None:
                return self.frequencies.copy(), self.psd_db.copy()
            else:
                return None, None

    def sine_best_fit(self, x_cropped, t_cropped, center_freq, residual_threshold, debug_data, debug):
        # Step 3: Fit a sine wave directly to the cropped filtered signal
        # Model: A * sin(2*pi*f*t + phi)

        # Define the sine model function
        def sine_model(t, A, f, phi):
            return A * np.sin(2 * np.pi * f * t + phi)

        # Initial parameter guesses
        # Amplitude: estimate from signal std
        A_guess = np.std(x_cropped) * np.sqrt(2)
        # Frequency: should be close to center_freq (360 Hz)
        f_guess = center_freq
        # Phase: estimate from first few samples
        phi_guess = 0.0

        # Perform curve fitting
        try:
            # Initial guess
            p0 = [A_guess, f_guess, phi_guess]

            # Fit the sine model directly to the cropped filtered signal
            popt, pcov = curve_fit(sine_model, t_cropped, x_cropped, p0=p0,
                                    method='lm',
                                    ftol=1e-15,      # Tighter function tolerance
                                    xtol=1e-15,      # Tighter parameter tolerance
                                    gtol=1e-15)      # Tighter gradient tolerance


            # Extract fitted parameters
            A_fit, f_fit, phi_fit = popt
            f_fit = np.abs(f_fit)

            # The fitted frequency is the actual frequency (no baseband offset)
            estimated_freq = f_fit

            # Calculate deviation in seconds per day
            # deviation (s/day) = (f_error / f_nominal) * 86400
            freq_error = estimated_freq - self.reference_freq
            deviation_spd = (freq_error / self.reference_freq) * 86400.0

            # Generate fitted sine wave and compute residuals
            fitted_signal = sine_model(t_cropped, A_fit, f_fit, phi_fit)
            residuals = x_cropped - fitted_signal

            # Check fit quality using RMS residual relative to signal amplitude
            rms_residual = np.sqrt(np.mean(residuals**2))
            signal_amplitude = A_fit

            # Debug plot 3: Cropped signal and fitted sine wave
            # ALWAYS update debug plots, even if fit is poor
            if debug:
                # Store debug data
                debug_data['x_cropped_fit'] = x_cropped
                debug_data['t_cropped_fit'] = t_cropped
                debug_data['fitted_signal'] = fitted_signal
                debug_data['residuals'] = residuals
                debug_data['A_fit'] = A_fit
                debug_data['f_fit'] = f_fit
                debug_data['phi_fit'] = phi_fit
                debug_data['estimated_freq'] = estimated_freq
                debug_data['deviation_spd'] = deviation_spd
                debug_data['fit_method'] = 'sine_fit'

                # Put debug data in queue (non-blocking, discard if full)
                try:
                    self.debug_plot_queue.put_nowait(debug_data)
                except queue.Full:
                    # Queue is full, discard oldest and try again
                    try:
                        self.debug_plot_queue.get_nowait()
                        self.debug_plot_queue.put_nowait(debug_data)
                    except:
                        pass  # If still fails, just skip this update

            # Check fit quality and raise exception AFTER debug data is queued
            # Threshold: RMS residual should be < threshold of signal amplitude
            if signal_amplitude > 0:
                relative_residual = rms_residual / signal_amplitude
                if relative_residual > residual_threshold:
                    raise PoorFitError(f"Poor fit quality: RMS residual {relative_residual:.2%} exceeds threshold {residual_threshold:.2%}")

            # Update timegrapher data (only if fit quality is good)
            with self.lock:
                self.timegrapher_freq = estimated_freq
                self.timegrapher_deviation = deviation_spd
                self.signal_quality_good = True

        except PoorFitError as pf_error:
            # Poor fit detected - skip this update silently (or log if desired)
            # Don't update timegrapher_freq or timegrapher_deviation
            # This prevents the GUI from detecting "new" data
            print(f"Skipping update: {pf_error}")
            with self.lock:
                self.signal_quality_good = False

        except Exception as fit_error:
            print(f"Curve fitting error: {fit_error}")

    def instantaneous_phase_fit(self, x_cropped, t_cropped, center_freq, residual_threshold, debug_data, debug):
        """
        Estimate frequency using instantaneous phase from Hilbert transform

        This method:
        1. Computes the analytic signal using Hilbert transform
        2. Extracts and unwraps the instantaneous phase
        3. Fits a line to the unwrapped phase
        4. Extracts instantaneous frequency from the slope

        Parameters:
        -----------
        x_cropped : numpy array
            Cropped filtered signal
        t_cropped : numpy array
            Time vector for cropped signal
        center_freq : float
            Center frequency of the bandpass filter
        residual_threshold : float
            Threshold for fit quality (normalized residual)
        debug_data : dict
            Dictionary to store debug plot data
        debug : bool
            Whether to store debug data
        """
        try:
            # Compute analytic signal using Hilbert transform
            analytic_signal = signal.hilbert(x_cropped)

            # Extract instantaneous phase
            instantaneous_phase = np.angle(analytic_signal)

            # Unwrap phase to remove 2π discontinuities
            unwrapped_phase = np.unwrap(instantaneous_phase)

            # Fit a linear model to the unwrapped phase: φ(t) = 2πf*t + φ₀
            # Using polyfit for robust linear regression
            coeffs = np.polyfit(t_cropped, unwrapped_phase, 1)
            slope = coeffs[0]  # This is 2πf
            intercept = coeffs[1]  # This is φ₀

            # Extract frequency from slope: f = slope / (2π)
            estimated_freq = slope / (2 * np.pi)
            estimated_freq = np.abs(estimated_freq)

            # Calculate deviation in seconds per day
            freq_error = estimated_freq - self.reference_freq
            deviation_spd = (freq_error / self.reference_freq) * 86400.0

            # Generate fitted phase line for quality assessment
            fitted_phase = slope * t_cropped + intercept
            phase_residuals = unwrapped_phase - fitted_phase

            # Reconstruct signal from fitted phase for visualization
            amplitude = np.max(np.abs(analytic_signal))
            fitted_signal = amplitude * np.cos(fitted_phase)
            signal_residuals = x_cropped - fitted_signal

            # Check fit quality using RMS residual of phase fit
            rms_phase_residual = np.sqrt(np.mean(phase_residuals**2))
            phase_range = np.ptp(unwrapped_phase)  # Peak-to-peak range

            # Also check signal reconstruction quality
            rms_signal_residual = np.sqrt(np.mean(signal_residuals**2))
            signal_amplitude = amplitude

            # Store debug data (always, even if fit is poor)
            if debug:
                debug_data['x_cropped_fit'] = x_cropped
                debug_data['t_cropped_fit'] = t_cropped
                debug_data['fitted_signal'] = fitted_signal
                debug_data['residuals'] = signal_residuals
                debug_data['estimated_freq'] = estimated_freq
                debug_data['deviation_spd'] = deviation_spd
                # Phase-specific debug data
                debug_data['unwrapped_phase'] = unwrapped_phase
                debug_data['fitted_phase'] = fitted_phase
                debug_data['phase_residuals'] = phase_residuals
                debug_data['analytic_amplitude'] = np.abs(analytic_signal)
                debug_data['instantaneous_freq'] = np.diff(unwrapped_phase) / (2 * np.pi * np.diff(t_cropped))
                debug_data['fit_method'] = 'phase_fit'

                # Put debug data in queue (non-blocking, discard if full)
                try:
                    self.debug_plot_queue.put_nowait(debug_data)
                except queue.Full:
                    # Queue is full, discard oldest and try again
                    try:
                        self.debug_plot_queue.get_nowait()
                        self.debug_plot_queue.put_nowait(debug_data)
                    except:
                        pass  # If still fails, just skip this update

            # Check fit quality using signal reconstruction
            if signal_amplitude > 0:
                relative_residual = rms_signal_residual / signal_amplitude
                if relative_residual > residual_threshold:
                    raise PoorFitError(f"Poor phase fit quality: RMS residual {relative_residual:.2%} exceeds threshold {residual_threshold:.2%}")

            # Update timegrapher data (only if fit quality is good)
            with self.lock:
                self.timegrapher_freq = estimated_freq
                self.timegrapher_deviation = deviation_spd
                self.signal_quality_good = True

        except PoorFitError as pf_error:
            # Poor fit detected - skip this update
            print(f"Skipping update: {pf_error}")
            with self.lock:
                self.signal_quality_good = False

        except Exception as fit_error:
            print(f"Phase fitting error: {fit_error}")
            with self.lock:
                self.signal_quality_good = False

    def _analyze_timegrapher(self, audio_data, residual_threshold=0.05):
        """
        Analyze timegrapher data using bandpass filter and sine fitting

        Parameters:
        -----------
        audio_data : numpy array
            Audio samples (int16)
        residual_threshold : float
            Threshold for RMS residual relative to signal amplitude (default 0.05)
        """
        try:
            # Check if debug mode is enabled (thread-safe)
            with self.lock:
                debug = self.debug_mode
            # Convert to float and normalize
            audio_float = audio_data.astype(np.float64) / 32768.0

            # Step 1: Get or compute bandpass filter
            fir_coeff, lowcut, highcut, center_freq = self._compute_bandpass_filter(len(audio_float))

            # Apply FIR filter (already linear phase, but use filtfilt for zero phase)
            x_filtered = signal.filtfilt(fir_coeff, 1.0, audio_float)

            # Debug plot 1: Filter frequency response
            debug_data = {}
            if debug:
                w, h = signal.freqz(fir_coeff, worN=16384, fs=self.sample_rate)
                debug_data['filter_freq'] = w
                debug_data['filter_h'] = h
                debug_data['lowcut'] = lowcut
                debug_data['highcut'] = highcut
                debug_data['center_freq'] = center_freq

            # Step 2: Crop the filtered signal to remove filter transients
            # Remove first 10% and last 10% of the signal
            t_full = np.arange(len(audio_float)) / self.sample_rate
            crop_start = int(len(x_filtered) * 0.15)
            crop_end = int(len(x_filtered) * 0.85)

            x_cropped = x_filtered[crop_start:crop_end]
            t_cropped = t_full[crop_start:crop_end]

            # Debug plot 2: Filtered signal in time domain (show full signal)
            if debug:
                debug_data['time'] = t_full
                debug_data['x_filtered'] = x_filtered
                debug_data['time_cropped'] = t_cropped
                debug_data['x_cropped'] = x_cropped

            # Select the appropriate frequency estimation routine based on user selection
            if self.freq_estimation_method == 'sine_fit':
                print("sine_best_fit", flush=True)
                self.sine_best_fit(x_cropped, t_cropped, center_freq, residual_threshold, debug_data, debug)
            elif self.freq_estimation_method == 'phase_fit':
                print("instantaneous_phase_fit", flush=True)
                self.instantaneous_phase_fit(x_cropped, t_cropped, center_freq, residual_threshold, debug_data, debug)
            else:
                print(f"Unknown frequency estimation method: {self.freq_estimation_method}")

        except Exception as e:
            print(f"Error in timegrapher analysis: {e}")

    def get_timegrapher_data(self):
        """
        Get current timegrapher data

        Returns:
        --------
        freq_estimate : float
            Estimated frequency in Hz
        deviation_spd : float
            Deviation in seconds per day
        """
        with self.lock:
            if self.timegrapher_freq is not None and self.timegrapher_deviation is not None:
                return self.timegrapher_freq, self.timegrapher_deviation
            else:
                return None, None

    def get_signal_quality(self):
        """
        Get signal quality status

        Returns:
        --------
        bool : True if signal quality is good, False otherwise
        """
        with self.lock:
            return self.signal_quality_good

    def get_debug_plot_data(self):
        """
        Get debug plot data from queue (non-blocking)

        Returns:
        --------
        debug_data : dict or None
            Dictionary containing debug plot data, or None if queue is empty
        """
        try:
            return self.debug_plot_queue.get_nowait()
        except queue.Empty:
            return None

    def set_debug_mode(self, enabled):
        """
        Enable or disable debug mode

        Parameters:
        -----------
        enabled : bool
            True to enable debug plots, False to disable
        """
        with self.lock:
            self.debug_mode = enabled

    def set_reference_frequency(self, ref_freq):
        """
        Set the reference frequency and invalidate cached filter

        Parameters:
        -----------
        ref_freq : float
            Reference frequency in Hz
        """
        with self.lock:
            self.reference_freq = ref_freq
            # Invalidate cached filter so it gets recomputed
            self.fir_coeff = None
            self.filter_lowcut = None
            self.filter_highcut = None
            self.filter_center_freq = None

    def get_filter_params(self):
        """
        Get cached filter parameters

        Returns:
        --------
        tuple : (lowcut, highcut, center_freq) or (None, None, None)
            Filter cutoff frequencies and center frequency
        """
        with self.lock:
            if self.filter_center_freq is not None:
                return self.filter_lowcut, self.filter_highcut, self.filter_center_freq
            else:
                return None, None, None

    def _compute_bandpass_filter(self, signal_length):
        """
        Compute and cache FIR bandpass filter coefficients

        Parameters:
        -----------
        signal_length : int
            Length of the signal to be filtered (for determining numtaps)

        Returns:
        --------
        fir_coeff : numpy array
            FIR filter coefficients
        lowcut : float
            Low cutoff frequency
        highcut : float
            High cutoff frequency
        center_freq : float
            Center frequency
        """
        with self.lock:
            center_freq = self.reference_freq
            bandwidth_fraction = 0.003
            lowcut = center_freq * (1 - bandwidth_fraction / 2)
            highcut = center_freq * (1 + bandwidth_fraction / 2)

            # Check if we need to recompute filter
            if (self.fir_coeff is None or
                self.filter_center_freq != center_freq):

                # Design FIR bandpass filter (linear phase, symmetric transients)
                # Filter length: longer = sharper transition, but more edge effects
                # For 1s at 48kHz, use ~10% of length for filter
                numtaps = np.min([signal_length // 10, 4800]) # min(int(signal_length * 0.4), 4*1001)
                if numtaps % 2 == 0:  # Make odd for Type I filter (better for bandpass)
                    numtaps += 1

                # Design FIR filter using windowed method
                fir_coeff = signal.firwin(numtaps, [lowcut, highcut],
                                          pass_zero=False,
                                          fs=self.sample_rate,
                                          window='hamming')

                # Cache the filter
                self.fir_coeff = fir_coeff
                self.filter_lowcut = lowcut
                self.filter_highcut = highcut
                self.filter_center_freq = center_freq
            else:
                fir_coeff = self.fir_coeff
                lowcut = self.filter_lowcut
                highcut = self.filter_highcut

        return fir_coeff, lowcut, highcut, center_freq
