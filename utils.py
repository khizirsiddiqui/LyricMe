import string

all_chars = string.printable
num_chars = len(all_chars)

def character_to_label(char):
    return all_chars.find(char)

def string_to_label(char_string):
    return list(map(lambda char: character_to_label(char), char_string))

def pad_sequence(seq, max_length, pad_label=100):
    seq += [pad_label for i in range(max_length - len(seq))]
    return seq
