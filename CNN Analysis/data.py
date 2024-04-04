import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt


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


def pad_sigma(waveforms_df, n_sigma, oversize='remove'):

    """
    Inputs: 
        waveforms_df: Waveform dataframe with assigned truth values 
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

    waveforms_df['padded_sigma_{}'.format(n_sigma)] = waveforms_df['samples'].apply(lambda x: equalise_length(x, max_length))#.values#()
    return waveforms_df.dropna(axis=0)

# I am going to rewrite the original pad_waveforms to add capacity to pad with different numbers - this will be easier to play around with later on than loads of slightly different functions 


def pad_waveforms(waveforms_df, padding_name, padding_length=0, padding_type='constant',mean=0,std=1):
    '''
    Inputs: 
        waveforms_df: Waveform dataframe with assigned truth values 
        padding_type: type of padding to add to edges - the options are 'constant' (default), 'edge', 'reflect' (from np.pad) and 'gaussian' (which adds gaussian noise to the padding, default is mean=0, std=1 but can be changed)
        padding_length: number of time steps to over length of longest pulse - the number will be split evenly on both sides, i.e. a padding_length of 10 adds 5 time steps to either side
    Returns: 
        Added column to dataset including the padded arrays. All waveforms are padded to be the same length (equal to the length of longest waveform)
    '''

    def centre_padding(waveform, max_len):
        total_padding = max_len - len(waveform) 
        padding_left = total_padding // 2
        padding_right = total_padding - padding_left 
        return np.pad(waveform, (padding_left, padding_right), mode='constant', constant_values=0)
    
    def gaussian_padding(waveform, max_len):
        total_padding = max_len - len(waveform)
        left = total_padding // 2
        right = total_padding - left
        start_padding=np.random.normal(loc=mean,scale=std,size=left)
        end_padding=np.random.normal(loc=mean,scale=std,size=right)
        return np.concatenate((start_padding,waveform,end_padding))
    
    # unsure if I can make the assumption that the times and samples columns are the same length
    # so adding try/except (could be overkill but oh well)
    max_length_time = waveforms_df['times'].apply(len).max()
    max_length_intensity = waveforms_df['samples'].apply(len).max()
    try:
        # Try to assert that the maximum lengths are the same
        assert max_length_time == max_length_intensity
        # If the assertion passes, set max_length to one of them (since they are equal)
        max_length = max_length_time + padding_length 
    except AssertionError:
        # If the assertion fails, calculate max_length as the max of both
        max_length = max(max_length_time, max_length_intensity) + padding_length

    #max_len = waveforms_df['times'].apply(len).max()
    #waveforms_df['times'] = waveforms_df['times'].apply(lambda x: centre_padding(x, max_length))
    if padding_type == 'gaussian':
        padding_function = gaussian_padding
    else:
        padding_function = centre_padding
    waveforms_df[padding_name] = waveforms_df['samples'].apply(lambda x: padding_function(x, max_length))#.values#()
    return waveforms_df


def crop_waveforms(waveforms_df, crop_value, crop_set, location):

    """
    Removes crop_value time steps from ether front or back of waveform depending on location value
    Inputs: 
        waveforms_df: Waveform dataframe with assigned truth values 
        crop_value: number of time steps to remove
        location: 'front' or 'back' - determines which end of the waveform to crop from
    """

    padding_name = '{}_{}'.format(crop_value,location)
    def crop(waveform, crop_value, location):
        if location == 'front':
                return waveform[crop_value:] 
        if location == 'back':
            return waveform[:-crop_value] 

    waveforms_df[padding_name] = waveforms_df[crop_set].apply(lambda x: crop(x,crop_value,location))        
    return waveforms_df
        

def make_ML_data(waveforms_df, data_name):
    data_array=waveforms_df[data_name].to_numpy()
    x_data=np.stack(data_array,axis=0)

    max_phd=x_data.max()
    x_data=x_data/max_phd 
    #x_data[x_data < 0] = 0 -not sure ahout this, can implement if needed
    y_data = waveforms_df['label'].to_numpy().reshape((-1,1))
    input_length=x_data.shape[-1]

    runID = waveforms_df['runID']
    eventID = waveforms_df['eventID']
    weights_array = waveforms_df['weights_no_gas'].to_numpy()

    ML_data=train_test_split(x_data, y_data, weights_array, runID, eventID, random_state=0)

    return {'data':ML_data, 'input_length':input_length}

def plot_history(model,model_name):
    fig,axs=plt.subplots(3,1,figsize=(10,15))

    #plotting Accuracy
    axs[0].plot(model.history['accuracy'], label='Training Accuracy')
    axs[0].plot(model.history['val_accuracy'],label='Validation Accuracy')
    axs[0].legend()
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylim(top=1)

    #plotting Weighted Accuracy
    axs[1].plot(model.history['weighted_accuracy'], label='Weighted Training Accuracy')
    axs[1].plot(model.history['val_weighted_accuracy'],label='Validation Accuracy')
    axs[1].legend()
    axs[1].set_ylabel('Weighted Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylim(top=1)

    #plotting Loss
    axs[2].plot(model.history['loss'], label='Training Loss')
    axs[2].plot(model.history['val_loss'],label='Validation Loss')
    axs[2].legend()
    axs[2].set_ylabel('Loss')
    axs[2].set_xlabel('Epochs')

    plt.savefig('{}{}'.format(model_name,'.png'))
    
    plt.show()
   

def calc_aft(waveform, aft):
    """
    Calculates the desired aft value by summing the total area of the pulse
    Inputs: waveform - array of waveform samples
            aft - the aft proportion required eg for aft75 aft=0.75
    Outputs: index at which aft occurs in the pulse
    """
    i=0
    x=0
    while x<=aft*np.sum(waveform):
        x+=waveform[i]
        i+=1

    return i 

