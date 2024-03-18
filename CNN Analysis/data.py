import pandas as pd
import numpy as np


def waveforms_of_truth(waveforms_df, orig_train_df):
    '''
    Filter original data set to only include events with runID and eventID present in waveform
    dataset

    Then reorder the original data set such that the runIDs and the eventIDs line up

    Then save the truth values (i.e. cath-0, gate-1, trit-2) to the waveform dataframe
    ''' #df1=waveforms_df, df2=orig_train_df
    # first going to remove any rows in the generated waveform dataset with zero length waveforms
    try:
        waveforms_df = waveforms_df.loc[waveforms_df['length'] != 0]
        print('Rows with length = 0 have been removed from the waveform data set')
    except KeyError:
        print("Error: The specified 'length' column does not exist in the DataFrame.")
    except Exception as e:
        # Catch other potential exceptions and print the error message
        print(f"An error occurred: {e}")
    # waveforms_df.info()
    
    # need to delete duplicated rows of the original dataset for this to work
    orig_train_df = orig_train_df.set_index(['runID', 'eventID'])
    orig_train_df = orig_train_df[~orig_train_df.index.duplicated()]
    orig_train_df = orig_train_df.reset_index()

    # Step 1: Filter orig_train_df to only include rows that have matching values in both 'runID' and 'eventID' in waveforms_df.
    orig_train_df_filtered = orig_train_df.merge(waveforms_df[['runID', 'eventID']], on=['runID', 'eventID'], how='inner')

    # Step 2: Reorder orig_train_df to match the order of waveforms_df.
    # First, ensure that 'columnA' and 'columnB' are set as indices for both dataframes.
    waveforms_df = waveforms_df.set_index(['runID', 'eventID'])
    orig_train_df_filtered = orig_train_df_filtered.set_index(['runID', 'eventID'])

    # Now, reorder df2_filtered to match df1's order. This is done by reindexing df2_filtered based on df1's index.
    orig_train_df_reordered = orig_train_df_filtered.reindex(waveforms_df.index).reset_index()

    # print(orig_train_df_reordered[['label', 'type']])

    # waveforms_df['label'] = orig_train_df_reordered['label'].values
    # waveforms_df['type'] = orig_train_df_reordered['type'].values
    # waveforms_df['ext_elec'] = orig_train_df_reordered['ext_elec'].values
    columns_and_types = {'area': 'float64',
                         'max_pulse_height': 'float64',
                         'ext_elec': 'float64',
                         'x':'float64',
                         'y':'float64',
                         'r':'float64',
                         'S2_width':'float64',
                         'label':'int64',
                         'type':'object'
                         }

    # waveforms_df[['area', 'max_pulse_height', 'ext_elec', 'x', 
    #               'y', 'r', 'S2_width', 'label', 'type']] = orig_train_df_reordered[['area', 'max_pulse_height', 'ext_elec', 
    #                                                                                  'x', 'y', 'r', 'S2_width', 'label', 'type']].values
    for column, dtype in columns_and_types.items():
        waveforms_df[column] = orig_train_df_reordered[column].values.astype(dtype)

    waveforms_df = waveforms_df.reset_index()

    # print(waveforms_df[['label', 'type']])

    return waveforms_df

def pad_waveforms(waveforms_df):
    '''
    Expectation is that the waveforms dataframe is put through this function after it has been assigned
    truth values by the waveforms_of_truth function
    '''
    def centre_padding(waveform, max_len):
        total_padding = max_len - len(waveform)
        padding_left = total_padding // 2
        padding_right = total_padding - padding_left
        return np.pad(waveform, (padding_left, padding_right), mode='constant', constant_values=0)
    
    # unsure if I can make the assumption that the times and samples columns are the same length
    # so adding try/except (could be overkill but oh well)
    max_length_time = waveforms_df['times'].apply(len).max()
    max_length_intensity = waveforms_df['samples'].apply(len).max()
    try:
        # Try to assert that the maximum lengths are the same
        assert max_length_time == max_length_intensity
        # If the assertion passes, set max_length to one of them (since they are equal)
        max_length = max_length_time
    except AssertionError:
        # If the assertion fails, calculate max_length as the max of both
        max_length = max(max_length_time, max_length_intensity)

    #max_len = waveforms_df['times'].apply(len).max()
    #waveforms_df['times'] = waveforms_df['times'].apply(lambda x: centre_padding(x, max_length))
    waveforms_df['padded_samples'] = waveforms_df['samples'].apply(lambda x: centre_padding(x, max_length))#.values#()
    return waveforms_df


def pad_sigma(waveforms_df, n_sigma, oversize='remove'):

    """
    Inputs: 
        waveform_df: Waveform dataframe with assigned truth values 
        n_sigma: distance from mean length to limit to (number of std deviations)
        oversize: "remove" or "crop" - remove cuts all waveforms about specified length, crop shortens them, removing equal number of time steps from either end
    Output: 
        Adds column to waveform dataset where all waveforms are all within sigma of the mean length

    """
    def equalise_length(waveform, max_len):
        total_padding = max_len - len(waveform)
        if total_padding >= 0: #i.e waveform length is within n_sigma
            padding_left = int(total_padding // 2)
            padding_right = int(total_padding - padding_left)
            return np.pad(waveform, (padding_left, padding_right), mode='constant', constant_values=0)
        elif total_padding < 0: #waveform needs to be removed/cropped
            if oversize == 'remove':
                return np.nan
            elif oversize == 'crop':
                crop_left = int(np.abs(total_padding) // 2)
                crop_right = int(np.abs(total_padding) - crop_left)
                return np.array(waveform[crop_left:-crop_right])
    
    def find_len(mean, sigma):
        return np.round(mean + n_sigma*sigma)
    
    # statistics 
    mean_time=np.mean(waveforms_df['times'].apply(len))
    sigma_time=np.std(waveforms_df['times'].apply(len))
    mean_intensity=np.mean(waveforms_df['samples'].apply(len))
    sigma_intensity=np.std(waveforms_df['samples'].apply(len))
    #going to leave the try/except in as it can't hurt :))
    try:
        assert mean_time == mean_intensity
        assert sigma_time == sigma_intensity
        mean=mean_time
        sigma=sigma_time
    except AssertionError:
        # If the assertion fails, take the average of the values - this is somewhat arbitrary 
        mean = np.mean([mean_time, mean_intensity])
        std = np.mean([sigma_time, sigma_intensity])
    
    #calculate new max_length
    max_length=find_len(mean,sigma)
    print(max_length)

    waveforms_df['padded_samples'] = waveforms_df['samples'].apply(lambda x: equalise_length(x, max_length))#.values#()
    return waveforms_df.dropna(axis=0)
