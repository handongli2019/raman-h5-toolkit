import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

# Set folder path
folder_location = Path(r'C:\Users\lhd13\OneDrive\桌面\UCL\baml concentration collerbation curve\BAML\BAML\8 samples')

# Collect .h5 file paths
path_list = []
sample_name_list = []
for file in os.listdir(folder_location):
    if file.endswith('.h5'):
        full_path = folder_location / file
        path_list.append(full_path)
        sample_name = '_'.join(file.split('_')[:-1])
        sample_name_list.append(sample_name)

# Recursive collection for datasets with ndim > 2
def collect_h5_info_greater_than_2(name, obj, h5_dict):
    if isinstance(obj, h5py.Dataset) and obj.ndim > 2:
        h5_dict[name] = {
            'type': 'Dataset',
            'data': np.array(obj),
            'shape': obj.shape,
            'dtype': str(obj.dtype)
        }

def create_h5_dict_greater_than_2(h5file):
    h5_dict = {}
    h5file.visititems(lambda name, obj: collect_h5_info_greater_than_2(name, obj, h5_dict))
    return h5_dict

arrays_and_sizes_list = []
data_path_list = []

for full_path in path_list:
    with h5py.File(full_path, 'r') as h5file:
        h5_contents_dict = create_h5_dict_greater_than_2(h5file)
        arrays_and_sizes_with_keys = [
            {'key': key, 'array': details['data'], 'size': details['data'].size}
            for key, details in h5_contents_dict.items()
        ]
        paths = [item['key'] for item in arrays_and_sizes_with_keys]
        data_path_list.append(paths)
        arrays_and_sizes_list.append(arrays_and_sizes_with_keys)

# Extract datasets

data_list = []
for i, h5_file_path in enumerate(path_list):
    with h5py.File(h5_file_path, 'r') as h5file:
        dataset_path = data_path_list[i][0]
        if dataset_path in h5file:
            data = h5file[dataset_path][()]
            data_list.append(data)

# Extract axis values (assumes same index structure)
h5_file_path = path_list[0]
with h5py.File(h5_file_path, 'r') as h5file:
    time_values = h5file[data_path_list[0][1]][()]
    raman_shift_values = h5file[data_path_list[0][2]][()]

# Preprocess dimensions
two_dimensional_array_list = []
for data in data_list:
    squeezed_array = np.squeeze(data)
    reshaped_array = squeezed_array.reshape(squeezed_array.shape[-2], squeezed_array.shape[-1])
    two_dimensional_array_list.append(reshaped_array)

# Integrate between time gates
start_time = 7.2
end_time = 7.6
time_array = np.squeeze(time_values)
raman_shift_array = np.squeeze(raman_shift_values)
start_idx = np.where(time_array >= start_time)[0][0]
end_idx = np.where(time_array <= end_time)[0][-1]

integrated_data_list = []
for data in two_dimensional_array_list:
    integrated = np.trapz(data[start_idx:end_idx+1, :], axis=0)
    integrated_data_list.append(integrated)

# Normalization function
def min_max_normalisation(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.T).T

normalized_integrated = min_max_normalisation(pd.DataFrame(integrated_data_list))

# Plot 2D spectra
traces = []
for i, spectrum in enumerate(normalized_integrated):
    traces.append(go.Scatter(x=raman_shift_array, y=spectrum, mode='lines', name=sample_name_list[i]))

layout = go.Layout(
    title=f'2D Normalised Integrated Spectra ({start_time}-{end_time} ns)',
    xaxis=dict(title='Raman shift (1/cm)'),
    yaxis=dict(title='Normalized Intensity'),
    width=800,
    height=500,
    template='plotly_white'
)

fig = go.Figure(data=traces, layout=layout)
fig.show()

# 3D surface plot for one sample (e.g. index 2)
i = 7
data = two_dimensional_array_list[i]
x, y = np.meshgrid(raman_shift_array, time_array)

fig1 = go.Figure(data=[go.Surface(z=data[:-1], x=x, y=y, colorscale='Viridis')])
fig1.update_layout(
    title=f'3D Spectrum: {sample_name_list[i]}',
    scene=dict(
        xaxis_title='Raman shift (1/cm)',
        yaxis_title='Time (ns)',
        zaxis_title='Signal'
    ),
    height=600
)
fig1.show()


# Save all integrated spectra to Excel
all_df = pd.DataFrame(normalized_integrated, columns=raman_shift_array)
all_df.index = sample_name_list
all_df.to_excel("all_integrated_spectra.xlsx")
print("✅ All normalized integrated spectra saved to 'all_integrated_spectra.xlsx'")