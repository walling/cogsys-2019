import numpy as np
import pandas as pd

np.random.seed(483627689)

def round_to_sum(arr, expected_sum):
    values  = np.array(arr)
    rounded = np.round(values).astype(int)
    while True:
        actual_sum = np.sum(rounded)
        if actual_sum == expected_sum: return rounded
        error_index = np.argmax(np.abs(rounded - values))
        rounded[error_index] += np.sign(expected_sum - actual_sum)

all_words = pd.read_csv('words.csv', na_filter=False)
short_words = all_words.query('RemovalChoice == "-"')

pos_distribution = (short_words
    .groupby('POS')
    .agg({'WordSense': 'count'})
    .rename(columns={'WordSense': 'WordSenseCount'})
    .sort_values('WordSenseCount', ascending=False)
    .reset_index()
    .set_index('POS'))

pos_distribution['WordSensePercent'] = round_to_sum(
    pos_distribution.WordSenseCount / pos_distribution.WordSenseCount.sum() * 100, 100)

print()
print('POS distribution for short words (<= 2 syllables):')
print()
print(pos_distribution)
print()
print()

word_senses = (short_words
    .groupby('WordSense')
    .agg({'Word':'count','POS':'first'})
    .rename(columns={'Word':'WordCount'})
    .reset_index()
    .set_index('WordSense'))

selected_word_senses = word_senses.query('WordCount > 1')
selected_words = short_words[short_words.WordSense.isin(selected_word_senses.index)]

remaining_word_senses = word_senses.query('WordCount == 1')
remaining_words = short_words[short_words.WordSense.isin(remaining_word_senses.index)]

print()
print()
print(remaining_word_senses)
print()
print()

remaining_pos_distribution = pos_distribution.WordSensePercent.to_dict()
for pos in selected_word_senses.POS: remaining_pos_distribution[pos] -= 1

selected_words_segments = [selected_words]
for pos, count in remaining_pos_distribution.items():
    if count == 0: continue
    word_senses_for_pos = remaining_word_senses[remaining_word_senses.POS == pos].index.to_list()
    selected_pos_word_senses = np.random.permutation(word_senses_for_pos)[:count]
    selected_pos_words = remaining_words[remaining_words.WordSense.isin(selected_pos_word_senses)]
    print(pos, count, len(selected_pos_word_senses))
    selected_words_segments.append(selected_pos_words)

word_list = pd.concat(selected_words_segments).set_index('WordSense').sort_index(0)
word_list.to_csv('word_list.csv')

print()
print('POS distribution for final word list:')
print()
print(word_list
    .groupby('WordSense')
    .agg({'POS': 'first'})
    .groupby('POS')
    .agg({'POS': 'count'})
    .rename(columns={'POS': 'Count'})
    .sort_values('Count', ascending=False)
    .reset_index()
    .set_index('POS'))
print()
