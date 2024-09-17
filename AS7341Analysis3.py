import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Load the .TXT file (no headers)
filename = '/Users/richardkanak/Downloads/9-12 Ricky Calibration Data/R-9-12_ T1 70bpm.TXT'
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

# Initialize a DataFrame to store the linearly detrended series
detrended_data = pd.DataFrame()

# Subtract the linear line of best fit for each wavelength and store it in detrended_data
for i in range(data.shape[1]):
    series = data.iloc[:, i]  # Extract the i-th wavelength
    if len(series) != len(time):
        print(f"Error: Time and series length mismatch for Wavelength {i+1}.")
    # Linear fit coefficients (y = mx + b)
    m, b = np.polyfit(time, series, 1)
    linear_fit = m * time + b
    detrended_series = series - linear_fit  # Subtract the linear fit
    detrended_data[f'Wavelength {i+1}'] = detrended_series

# Check the structure of detrended data
print("Detrended data columns:", detrended_data.columns)
print(detrended_data.head())

# Remove any NaN values from the detrended data (check lengths after this step)
detrended_data = detrended_data.dropna()

# Trim time vector to match the detrended data after dropping NaNs
if len(time) > len(detrended_data):
    time = time[:len(detrended_data)]

# Define the correct wavelength labels
wavelength_labels = [
    "415 nm", "445 nm", "480 nm", "515 nm", "555 nm", 
    "590 nm", "630 nm", "680 nm", "Clear", "Near IR"
]

# Plot all wavelengths in a single overlay plot
plt.figure()
for i in range(detrended_data.shape[1]):
    plt.plot(time, detrended_data[f'Wavelength {i+1}'], label=wavelength_labels[i])

plt.title('All Detrended Wavelengths')
plt.xlabel('Time (seconds)')
plt.ylabel('Detrended Intensity')
plt.legend()
plt.grid(True)

# Calculate the average of all detrended wavelengths
average_detrended_series = detrended_data.mean(axis=1)
if average_detrended_series.isna().any():
    print("NaN values detected in average_detrended_series. Investigating further...")

print("Average detrended series:")
print(average_detrended_series.head())  # Check for potential issues here

# Ensure time and average_detrended_series have the same length
if len(average_detrended_series) != len(time):
    print(f"Warning: Length mismatch between time ({len(time)}) and average_detrended_series ({len(average_detrended_series)}).")
    average_detrended_series = average_detrended_series[:len(time)]  # Trim if needed

# Plot the average of all detrended wavelengths
plt.figure()
plt.plot(time, average_detrended_series, color='black', linewidth=2)
plt.title('Average of All Detrended Wavelengths')
plt.xlabel('Time (seconds)')
plt.ylabel('Average Detrended Intensity')
plt.grid(True)

# Fourier analysis of the averaged detrended series
n = len(average_detrended_series)  # Number of samples
frequencies = fftfreq(n, time_step)  # Frequency values
fourier_transform = fft(average_detrended_series.values)  # Perform Fourier transform

# Check the Fourier transform results
print("Fourier transform length:", len(fourier_transform))
print("Frequencies length:", len(frequencies))

# Limit Fourier results to the desired frequency range (0.5 Hz to 2.5 Hz)
min_freq = 0.5
max_freq = 2.5
valid_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
limited_frequencies = frequencies[valid_indices]
limited_fourier_transform = np.abs(fourier_transform[valid_indices])  # Get the magnitudes

# Find the peak frequency (frequency with the highest amplitude)
peak_index = np.argmax(limited_fourier_transform)  # Index of the peak in the limited range
peak_frequency = limited_frequencies[peak_index]  # Peak frequency in Hz
peak_amplitude = limited_fourier_transform[peak_index]  # Peak amplitude

# Convert peak frequency from Hz to BPM
peak_bpm = peak_frequency * 60  # Convert Hz to BPM

# Print the peak frequency and corresponding BPM
print(f"Peak Frequency: {peak_frequency:.2f} Hz")
print(f"Peak Frequency in BPM: {peak_bpm:.2f} BPM")

# Plot the Fourier analysis in the specified frequency range
plt.figure()
plt.plot(limited_frequencies, limited_fourier_transform, color='blue')
plt.title('Fourier Analysis (0.5 Hz to 2.5 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

# Show all plots
plt.show()