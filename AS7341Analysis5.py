import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Load the .TXT file (no headers)
filename = '/Users/richardkanak/Downloads/L2-1min-Leron.TXT'
data = pd.read_csv(filename, header=None)

# Time step (in seconds)
time_step = 0.125  # Adjust based on your sampling period
num_samples = len(data)
time = np.arange(0, num_samples * time_step, time_step)  # Time vector

# Ensure the time vector length matches the data
if len(time) != num_samples:
    print(f"Warning: Time vector length ({len(time)}) and data length ({num_samples}) do not match.")
    time = time[:num_samples]  # Trim time to match data

# Check if data was loaded correctly
print("Data loaded successfully!")
print(data.head())

# Indices for Near IR and 680 nm wavelengths (Python is zero-indexed, so 680 nm is at index 7 and Near IR is at index 9)
wavelength_indices = {
    '680 nm': 7,
    'Near IR': 9
}

# Initialize a DataFrame to store the detrended series for each wavelength
detrended_data = pd.DataFrame()

# Isolate and detrend the specified wavelengths
for label, index in wavelength_indices.items():
    series = data.iloc[:, index]
    if len(series) != len(time):
        print(f"Error: Time and series length mismatch for {label}.")
    # Linear fit coefficients (y = mx + b)
    m, b = np.polyfit(time, series, 1)
    linear_fit = m * time + b
    detrended_series = series - linear_fit  # Subtract the linear fit
    detrended_data[label] = detrended_series

# Remove any NaN values from the detrended data (check lengths after this step)
detrended_data = detrended_data.dropna()

# Trim time vector to match the detrended data after dropping NaNs
if len(time) > len(detrended_data):
    time = time[:len(detrended_data)]

# 1. Original Comparison Plot
plt.figure(figsize=(12, 6))
plt.plot(time, detrended_data['680 nm'], label='680 nm', color='green')
plt.plot(time, detrended_data['Near IR'], label='Near IR', color='red')
plt.title('Detrended Wavelengths (680 nm and Near IR)')
plt.xlabel('Time (seconds)')
plt.ylabel('Detrended Intensity')
plt.legend()
plt.grid(True)
plt.savefig('figure_1.png')  # Save the figure as a file

# 2. Fourier Analysis Plot
plt.figure(figsize=(12, 6))
for label, color in zip(['680 nm', 'Near IR'], ['green', 'red']):
    series = detrended_data[label]
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
    plt.plot(limited_frequencies, limited_fourier_transform, label=f'{label} Fourier', color=color)
plt.title('Fourier Analysis of Each Wavelength (0.5 Hz to 2.5 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig('figure_2.png')  # Save the figure as a file

# 3. Raw Difference Plot
# Calculate the difference between the two wavelengths
difference_series = detrended_data['680 nm'] - detrended_data['Near IR']

# Plot the raw difference
plt.figure(figsize=(12, 6))
plt.plot(time, difference_series, label='Difference (680 nm - Near IR)', color='purple')
plt.title('Raw Difference Between 680 nm and Near IR')
plt.xlabel('Time (seconds)')
plt.ylabel('Difference in Detrended Intensity')
plt.legend()
plt.grid(True)
plt.savefig('figure_3.png')  # Save the figure as a file

# Show all figures
plt.show()
