import pandas as pd
import data
import matplotlib.pyplot as plt
import numpy as np
import hist
from hist import Hist


# read in waveform dataframe
waveforms = pd.read_parquet('../../../Data/all_msci_waveforms_df.parquet')
print('Waveforms Set')
waveforms.info()
print()

#waveforms.info()
#waveforms = waveforms.drop('Unnamed: 0', axis = 1)
# convert string entries to numpy arrays again (weird issue when saving)
# waveforms['samples'] = waveforms['samples'].apply(np.fromstring)
# waveforms['times'] = waveforms['times'].apply(np.fromstring)
#waveforms.info()
# read in tritium and background datasets and combine
trit_data = pd.read_csv('../../../Data/tritium_ML_data.csv')
bkg_data = pd.read_csv('../../../Data/bg_sr1_vetoes_gassplit.csv')
orig_train = pd.concat([trit_data, bkg_data], ignore_index=True)
orig_train = orig_train.drop('Unnamed: 0', axis = 1)
print('Original Dataset')
orig_train.info()
print()

# orig_train = orig_train.set_index(['runID', 'eventID'])
# orig_train = orig_train[~orig_train.index.duplicated()]
# orig_train = orig_train.reset_index()
# print('Unduplicated Original Dataset')
# orig_train.info()
# print()

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
df_cut = final_dataset[final_dataset["type"] != 'gas']
df_cut.info()
padded_data = data.pad_sigma(df_cut,2,'remove')
print('Waveform Dataset')
padded_data.info()
print()
#padded_data.head()

"""
# padded_data.info()
rand_event_num = np.random.randint(0, padded_data.shape[0])
rand_event_samps = padded_data.at[rand_event_num, 'padded_samples']#.to_numpy()
print(type(rand_event_samps))
print()
dummy_times = np.linspace(0, len(rand_event_samps)-1, len(rand_event_samps))
plt.figure()
plt.plot(dummy_times, rand_event_samps)
plt.xlabel('Time steps')
plt.ylabel('Signal Intensity')
plt.savefig('padding_test.png')

print('\nWith Gas Events...')
print('Waveform Min. Length - ' + str(padded_data['length'].min()))
print('Waveform Max. Length - ' + str(padded_data['length'].max()))
print('Waveform Mean Length - ' + str(np.around(padded_data['length'].mean(), 3)))
print('Waveform Length std. - ' + str(np.around(padded_data['length'].std(), 3)))

# making a histogram of waveform lengths

length_axis = hist.axis.Regular(100,100,9000,name="length",
                                label="Number of timesteps", transform=hist.axis.transform.log)

gate_hist = Hist(length_axis)
gate_cut = padded_data.label == 1
gate_hist.fill(padded_data.length[gate_cut])

gas_hist = Hist(length_axis)
gas_cut = padded_data.label == 3
gas_hist.fill(padded_data.length[gas_cut])

cath_hist = Hist(length_axis)
cath_cut = padded_data.label == 0
cath_hist.fill(padded_data.length[cath_cut])

trit_hist = Hist(length_axis)
trit_cut = padded_data.label == 2
trit_hist.fill(padded_data.length[trit_cut])

fig, ax = plt.subplots(figsize=(10, 8))
gate_hist.plot(color='mediumpurple',ls='-',lw=2,yerr=False,label='Gate',ax=ax)
cath_hist.plot(color='red',ls='-',lw=2,yerr=False,label='Cathode',ax=ax)
trit_hist.plot(color='black',ls='-',lw=2,yerr=False,label='Tritium',ax=ax)
gas_hist.plot(color='orange',ls='-',lw=2,yerr=False,label='Gas',ax=ax)

ax.set(xlabel='Timesteps in Waveform',ylabel='Counts',xscale='log', yscale='log')#,ylim=[8,1000])
ax.legend(bbox_to_anchor=(1,1),loc='upper right',frameon=False, fontsize = 18)

plt.savefig('waveform_length_hist')

# removing gas events

gas_cut = padded_data["type"] != 'gas'
padded_data = padded_data[gas_cut]

print('\nWithout Gas Events...')
print('Waveform Min. Length - ' + str(padded_data['length'].min()))
print('Waveform Max. Length - ' + str(padded_data['length'].max()))
print('Waveform Mean Length - ' + str(np.around(padded_data['length'].mean(), 3)))
print('Waveform Length std. - ' + str(np.around(padded_data['length'].std(), 3)))

fig, ax = plt.subplots(figsize=(10, 8))
gate_hist.plot(color='mediumpurple',ls='-',lw=2,yerr=False,label='Gate',ax=ax)
cath_hist.plot(color='red',ls='-',lw=2,yerr=False,label='Cathode',ax=ax)
trit_hist.plot(color='black',ls='-',lw=2,yerr=False,label='Tritium',ax=ax)

ax.set(xlabel='Timesteps in Waveform',ylabel='Counts',xscale='log', yscale='log')#,ylim=[8,1000])
ax.legend(bbox_to_anchor=(1,1),loc='upper right',frameon=False, fontsize = 18)

plt.savefig('waveform_length_hist_nogas')

# padded_data_nogas = data.pad_waveforms(final_dataset[gas_cut])

# rand_event_num = np.random.randint(0, padded_data_nogas.shape[0])
# rand_event_samps = padded_data_nogas.at[rand_event_num, 'padded_samples']#.to_numpy()
# print(type(rand_event_samps))
# print()
# dummy_times = np.linspace(0, len(rand_event_samps)-1, len(rand_event_samps))
# plt.figure()
# plt.plot(dummy_times, rand_event_samps)
# plt.xlabel('Time steps')
# plt.ylabel('Signal Intensity')
# plt.savefig('padding_test_nogas.png')
"""