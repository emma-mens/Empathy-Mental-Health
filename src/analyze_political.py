import os
import os.path as path
import re
import numpy as np
import pandas as pd

inputPath = os.path.abspath('/local1/baughan/dataset/political_output.csv')
inputPath2 = os.path.abspath('/local1/baughan/dataset/political_tweets.csv')

raw = pd.read_csv(inputPath)
raw2 = pd.read_csv(inputPath2)

df = raw.join(raw2['conv_type'], on="id")
df.to_csv('/local1/baughan/dataset/dataset/political_outputc.csv', index=True)
print(df.head())

# annotate_key = df.sample(50)
# annotate_blind = annotate_key[['id', 'seeker_post', 'response_post']]
# print(annotate_key)
# print(annotate_blind)

# annotate_key.to_csv('/Users/baughancse/repos/Empathy-Mental-Health/dataset/annotate_key.csv', index=True)
# annotate_blind.to_csv('/Users/baughancse/repos/Empathy-Mental-Health/dataset/annotate_blind.csv', index=True)