import pandas as pd
import data
import matplotlib.pyplot as plt
import numpy as np


# read in waveform dataframe
waveforms = pd.read_parquet('msci_waveforms_df.parquet')
#waveforms.info()
#waveforms = waveforms.drop('Unnamed: 0', axis = 1)
# convert string entries to numpy arrays again (weird issue when saving)
# waveforms['samples'] = waveforms['samples'].apply(np.fromstring)
# waveforms['times'] = waveforms['times'].apply(np.fromstring)
#waveforms.info()
# read in tritium and background datasets and combine
trit_data = pd.read_csv('../Decision Tree Analysis/data/tritium_ML_data.csv')
bkg_data = pd.read_csv('../Decision Tree Analysis/data/bg_sr1_vetoes_gassplit.csv')
orig_train = pd.concat([trit_data, bkg_data], ignore_index=True)
orig_train = orig_train.drop('Unnamed: 0', axis = 1)

print(type(waveforms.at[194, 'times'])) # parquet file keeps it as a numpy array!!!
# test out the functions
final_dataset = data.waveforms_of_truth(waveforms, orig_train)

#print(final_dataset.head())
# final_dataset.info()

# print(final_dataset.runID)

# in the new dataset, runID=6940 eventID=2797 is gas                => check :)
# in the new dataset, runID=6940 eventID=3177 is cathode            => check :)
# in the new dataset, runID=6940 eventID=3705 is gate               => check :)

#final_dataset.to_csv('test_final_dataset.csv')

# check the padding works out
padded_data = data.pad_waveforms(final_dataset)
padded_data.info()

# padded_data.info()
rand_event_num = np.random.randint(0, padded_data.shape[0])
rand_event_samps = padded_data.at[rand_event_num, 'padded_samples']#.to_numpy()
print(type(rand_event_samps))
dummy_times = np.linspace(0, len(rand_event_samps)-1, len(rand_event_samps))
plt.figure()
plt.plot(dummy_times, rand_event_samps)
plt.xlabel('Time steps')
plt.ylabel('Signal Intensity')
plt.savefig('padding_test.png')

print('\nWaveform Min. Length - ' + str(padded_data['length'].min()))
print('Waveform Max. Length - ' + str(padded_data['length'].max()))
print('Waveform Mean Length - ' + str(np.around(padded_data['length'].mean(), 3)))
print('Waveform Length std. - ' + str(np.around(padded_data['length'].std(), 3)))