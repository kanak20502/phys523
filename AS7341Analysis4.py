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

# Define wavelength groups
groups = {
    'Group 1': [0, 1, 2],   # First three wavelengths
    'Group 2': [8, 9],      # Last two wavelengths
    'Group 3': [3, 4, 5, 6, 7]  # Remaining wavelengths
}

# Initialize figure for all groups
fig, axs = plt.subplots(len(groups), 3, figsize=(18, 15), sharex=True)

for idx, (group_name, indices) in enumerate(groups.items()):
    # Extract data for the group
    group_data = data.iloc[:, indices]
    
    # Initialize DataFrame for detrended data
    detrended_data = pd.DataFrame()
    
    # Detrend each wavelength in the group
    for i in range(group_data.shape[1]):
        series = group_data.iloc[:, i]
        if len(series) != len(time):
            print(f"Error: Time and series length mismatch for {group_name} Wavelength {indices[i]+1}.")
        # Linear fit coefficients (y = mx + b)
        m, b = np.polyfit(time, series, 1)
        linear_fit = m * time + b
        detrended_series = series - linear_fit
        detrended_data[f'Wavelength {indices[i]+1}'] = detrended_series

    # Remove any NaN values from the detrended data
    detrended_data = detrended_data.dropna()

    # Trim time vector to match the detrended data after dropping NaNs
    if len(time) > len(detrended_data):
        time = time[:len(detrended_data)]

    # Plot all detrended wavelengths in the group
    for i in range(detrended_data.shape[1]):
        axs[idx, 0].plot(time, detrended_data[f'Wavelength {indices[i]+1}'], label=f'Wavelength {indices[i]+1}')
    axs[idx, 0].set_title(f'Detrended Wavelengths - {group_name}')
    axs[idx, 0].set_xlabel('Time (seconds)')
    axs[idx, 0].set_ylabel('Detrended Intensity')
    axs[idx, 0].legend()
    axs[idx, 0].grid(True)

    # Calculate the average of all detrended wavelengths for the group
    average_detrended_series = detrended_data.mean(axis=1)
    
    # Ensure time and average_detrended_series have the same length
    if len(average_detrended_series) != len(time):
        print(f"Warning: Length mismatch between time and average_detrended_series for {group_name}.")
        average_detrended_series = average_detrended_series[:len(time)]

    # Plot the average of all detrended wavelengths
    axs[idx, 1].plot(time, average_detrended_series, color='black', linewidth=2)
    axs[idx, 1].set_title(f'Average of All Detrended Wavelengths - {group_name}')
    axs[idx, 1].set_xlabel('Time (seconds)')
    axs[idx, 1].set_ylabel('Average Detrended Intensity')
    axs[idx, 1].grid(True)

    # Fourier analysis of the averaged detrended series
    n = len(average_detrended_series)  # Number of samples
    frequencies = fftfreq(n, time_step)  # Frequency values
    fourier_transform = fft(average_detrended_series.values)  # Perform Fourier transform

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
    print(f"{group_name} - Peak Frequency: {peak_frequency:.2f} Hz")
    print(f"{group_name} - Peak Frequency in BPM: {peak_bpm:.2f} BPM")

    # Plot the Fourier analysis in the specified frequency range
    axs[idx, 2].plot(limited_frequencies, limited_fourier_transform, color='blue')
    axs[idx, 2].set_title(f'Fourier Analysis - {group_name} (0.5 Hz to 2.5 Hz)')
    axs[idx, 2].set_xlabel('Frequency (Hz)')
    axs[idx, 2].set_ylabel('Amplitude')
    axs[idx, 2].grid(True)

# Adjust layout and show all plots
plt.tight_layout()
plt.show()
