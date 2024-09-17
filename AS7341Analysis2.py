import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Load the .TXT file (no headers)
filename = 'Enter File Name'
data = pd.read_csv(filename, header=None)

#chad test


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

# Plot all wavelengths in a single overlay plot
plt.figure()
for i in range(detrended_data.shape[1]):
    plt.plot(time, detrended_data[f'Wavelength {i+1}'], label=f'Wavelength {i+1}')

plt.title('All Detrended Wavelengths')
plt.xlabel('Time (seconds)')
plt.ylabel('Detrended Intensity')
plt.legend()
plt.grid(True)

# Plot the average of all detrended wavelengths
average_detrended_series = detrended_data.mean(axis=1)
print("Average detrended series:")
print(average_detrended_series.head())  # Check for potential issues here

plt.figure()
plt.plot(time, average_detrended_series, color='black', linewidth=2)
plt.title('Average of All Detrended Wavelengths')
plt.xlabel('Time (seconds)')
plt.ylabel('Average Detrended Intensity')
plt.grid(True)


#show all plots
plt.show()
