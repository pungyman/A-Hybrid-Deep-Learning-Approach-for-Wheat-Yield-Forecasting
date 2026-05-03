import ast
import torch
import pandas as pd
from torch.utils.data import Dataset

class YieldPredictionDataset(Dataset):
    '''
    An object of this class can be instantiated by providing a path to a csv file with sequence data and the lagged yield feature ('past_yield' column).
    Optionally supports soil features if soil_features_available=True.
    Can be filtered for a specific state for finetuning if ft=True and a state is provided.
    '''
    def __init__(self, csv_file, inference=False, soil_features_available=False, ft=False, state=None, disable_ndvi_evi=False):
        """
        Args:
            csv_file (str): Path to the csv file with sequence data.
            inference (bool): If True, __getitem__ will also return district_id and sowing_year.
            soil_features_available (bool): If True, expects 'soil_features' column in csv and includes soil features in __getitem__.
            ft (bool): If True, the dataset is filtered for a specific state for finetuning.
            state (str): The state to filter the dataset by. Required if ft is True.
            disable_ndvi_evi (bool): If True, NDVI and EVI features are dropped from the temporal sequences.
        """
        # Load the dataframe
        data_frame = pd.read_csv(csv_file)

        # Filter for state if finetuning is enabled
        if ft:
            if state is None:
                raise ValueError("State must be provided for finetuning dataset.")
            data_frame = data_frame[data_frame['DISTRICT_ID'].str.startswith(state)].copy()

        # Process and convert feature sequences to a tensor.
        sequences_list = data_frame['feature_sequence'].apply(ast.literal_eval).tolist()
        self.sequences = torch.tensor(sequences_list, dtype=torch.float32)
        # shape of self.sequences: number of data points, sequence length, number of features

        # If disable_ndvi_evi is True, drop the last two features (NDVI and EVI)
        if disable_ndvi_evi:
            self.sequences = self.sequences[:, :, :-2]

        # Process and convert yields to a tensor
        yields_list = data_frame['Yield'].tolist()
        self.yields = torch.tensor(yields_list, dtype=torch.float32)
        
        # Process lagged yield (past_yield) data
        past_yields_list = data_frame['past_yield'].tolist()
        self.past_yields = torch.tensor(past_yields_list, dtype=torch.float32)

        # Process soil features if available
        self.soil_features_available = soil_features_available
        if soil_features_available:
            soil_features_list = data_frame['soil_features'].apply(ast.literal_eval).tolist()
            self.soil_features = torch.tensor(soil_features_list, dtype=torch.float32)
            # shape of self.soil_features: number of data points, features, depth

        # Store inference flag and additional data if needed
        self.inference = inference
        if inference is True:
            self.district_ids = data_frame['DISTRICT_ID'].tolist()
            self.sowing_years = data_frame['sowing_year'].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        '''
        for inference=True and soil_features_available=False:
            returns district id, sowing year, sequence of temporal features, lagged yield, yield (target)
        for inference=False and soil_features_available=False:
            returns sequence of temporal features, lagged yield, yield (target)
        for inference=True and soil_features_available=True:
            returns district id, sowing year, sequence of temporal features, lagged yield, soil features, yield (target)
        for inference=False and soil_features_available=True:
            returns sequence of temporal features, lagged yield, soil features, yield (target)
        '''
        if self.inference is True:
            if self.soil_features_available:
                return self.district_ids[idx], self.sowing_years[idx], self.sequences[idx], self.past_yields[idx], self.soil_features[idx], self.yields[idx]
            else:
                return self.district_ids[idx], self.sowing_years[idx], self.sequences[idx], self.past_yields[idx], self.yields[idx]
        else:
            if self.soil_features_available:
                return self.sequences[idx], self.past_yields[idx], self.soil_features[idx], self.yields[idx]
            else:
                return self.sequences[idx], self.past_yields[idx], self.yields[idx]