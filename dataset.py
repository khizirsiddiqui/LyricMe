import torch
import pandas as pd
import torch.utils.data as data
from utils import *

class LyricsDataset(data.Dataset):
    def __init__(self, csv_path, min_song_count=None, artists=None):
        self.lyrics_dataframe = pd.read_csv(csv_path)
        if artists:
            self.lyrics_dataframe = self.lyrics_dataframe[self.lyrics_dataframe.artists.isin(artists)]
            self.lyrics_dataframe = self.lyrics_dataframe.reset_index()
        if min_song_count:
            self.lyrics_dataframe = self.lyrics_dataframe.group_by('artist').filter(lambda x: len(x) > min_song_count)
            self.lyrics_dataframe = self.lyrics_dataframe.reset_index()
        self.max_text_len = self.lyrics_dataframe.text.str.len().max()
        self.indices = range(len(self.lyrics_dataframe))
        self.artists_list = list(self.lyrics_dataframe.artist.unique())
        self.num_of_artists = len(self.artists_list)
        print("===x----Lyrics Dataset----x===")
        print("CSV File      :", csv_path)
        print("Dataset Length:", len(self.indices))
        print("Total Artists :", self.num_of_artists)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        index = self.indices[index]
        seq_raw = self.lyrics_dataframe.loc[index].text
        seq_raw_labels = string_to_label(seq_raw)
        seq_len = len(list(seq_raw_labels)) - 1
        input_str_labels = seq_raw_labels[:1]
        output_str_labels = seq_raw_labels[1:]
        input_str_padded = pad_sequence(input_str_labels, max_length=self.max_text_len)
        output_str_padded = pad_sequence(output_str_labels, max_length=self.max_text_len, pad_label=-100)
        return (torch.LongTensor(input_str_padded),
               torch.LongTensor(output_str_padded),
               torch.LongTensor([seq_len]))