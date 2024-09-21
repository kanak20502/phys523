import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# This analysis file looks at upstream and downstream data from the AS7341
# It plots 2 wavelengths from both streams in an overlayed plot
# It plots a fourier analysis of 2 wavelenths from both streams in an overlayed plot
# It plots a the ratio between two wavelengths at either end of the stream in an overlayed plot


# Load and clean the .TXT file (no headers)
filename = r'c:\Users\Richard\Downloads\DATA.TXT'
with open(filename, 'r') as file:
    raw_data = file.readlines()

# Remove 'Sensor 1:' and 'Sensor 2:' prefixes and split by commas
clean_data = []
for line in raw_data:
    # Remove unwanted prefixes and leading/trailing whitespace
    clean_line = line.replace('Sensor 1: ', '').replace('Sensor 2: ', '').strip()
    # Split the line by commas and convert each element to a float
    clean_data.append([float(value) for value in clean_line.split(',')])

# Convert clean_data to a DataFrame
data = pd.DataFrame(clean_data)

# Time step (in seconds)
time_step = 0.125  # Adjust based on your sampling period

# Split the data into upstream and downstream
downstream_data = data.iloc[::2].reset_index(drop=True)  # Every second row starting from the first (0-based index)
upstream_data = data.iloc[1::2].reset_index(drop=True)  # Every second row starting from the second (0-based index)

# Ensure both datasets have the same length
min_length = min(len(downstream_data), len(upstream_data))
downstream_data = downstream_data[:min_length]
upstream_data = upstream_data[:min_length]

# Create time vector based on the length of data
time = np.arange(0, min_length * time_step, time_step)

# Initialize DataFrames to store the detrended series
detrended_downstream = pd.DataFrame()
detrended_upstream = pd.DataFrame()

# Isolate and detrend the data for each sensor
for index, series in enumerate([downstream_data, upstream_data]):
    label = 'Downstream' if index == 0 else 'Upstream'
    for wavelength in ['680 nm', 'IR']:
        data_series = series[0] if wavelength == 'IR' else series[1]
        if len(data_series) != len(time):
            print(f"Error: Time and series length mismatch for {label} {wavelength}.")
            # Adjust time vector length to match data series
            time = time[:len(data_series)]
        # Linear fit coefficients (y = mx + b)
        m, b = np.polyfit(time, data_series, 1)
        linear_fit = m * time + b
        detrended_series = data_series - linear_fit  # Subtract the linear fit
        if index == 0:
            detrended_downstream[f'{label} {wavelength}'] = detrended_series
        else:
            detrended_upstream[f'{label} {wavelength}'] = detrended_series

# Remove any NaN values from the detrended data (check lengths after this step)
detrended_downstream = detrended_downstream.dropna()
detrended_upstream = detrended_upstream.dropna()

# Trim time vector to match the detrended data after dropping NaNs
if len(time) > len(detrended_downstream):
    time = time[:len(detrended_downstream)]

# 1. Overlayed Plot
plt.figure(figsize=(14, 8))
plt.plot(time, detrended_downstream['Downstream IR'], label='Downstream IR', color='blue')
plt.plot(time, detrended_upstream['Upstream IR'], label='Upstream IR', color='cyan')
plt.plot(time, detrended_downstream['Downstream 680 nm'], label='Downstream 680 nm', color='green')
plt.plot(time, detrended_upstream['Upstream 680 nm'], label='Upstream 680 nm', color='red')
plt.title('Detrended Sensor Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Detrended Intensity')
plt.legend()
plt.grid(True)
plt.savefig('figure_1.png')  # Save the figure as a file

# 2. Overlayed Fourier Analysis Plot
plt.figure(figsize=(14, 8))
for label, color in zip(['Downstream', 'Upstream'], ['blue', 'cyan']):
    for wavelength, color_offset in zip(['IR', '680 nm'], ['aqua', 'orange']):
        series = detrended_downstream[f'{label} {wavelength}'] if label == 'Downstream' else detrended_upstream[f'{label} {wavelength}']
        n = len(series)  # Number of samples
        frequencies = fftfreq(n, time_step)  # Frequency values
        fourier_transform = fft(series.values)  # Perform Fourier transform

        # Limit Fourier results to the desired frequency range (0.5 Hz to 2.5 Hz)
        min_freq = 0.5
        max_freq = 2.5
        valid_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
        limited_frequencies = frequencies[valid_indices]
        limited_fourier_transform = np.abs(fourier_transform[valid_indices])  # Get the magnitudes

        # Plot the Fourier analysis in the specified frequency range
        plt.plot(limited_frequencies, limited_fourier_transform, label=f'{label} {wavelength} Fourier', color=color_offset)
plt.title('Fourier Analysis of Each Sensor (0.5 Hz to 2.5 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig('figure_2.png')  # Save the figure as a file

# 3. Overlayed Ratio Plot
# Calculate ratios
downstream_ratio = detrended_downstream['Downstream 680 nm'] / detrended_downstream['Downstream IR']
upstream_ratio = detrended_upstream['Upstream 680 nm'] / detrended_upstream['Upstream IR']

plt.figure(figsize=(14, 8))
plt.plot(time, downstream_ratio, label='Downstream Ratio (680 nm / IR)', color='green')
plt.plot(time, upstream_ratio, label='Upstream Ratio (680 nm / IR)', color='red')
plt.title('Ratio of 680 nm to IR for Each Sensor')
plt.xlabel('Time (seconds)')
plt.ylabel('Ratio')
plt.legend()
plt.grid(True)
plt.savefig('figure_3.png')  # Save the figure as a file

# Show all figures
plt.show()
