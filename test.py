import pandas as pd

print("serial - deserialize")
df = pd.read_pickle('dataframe.pkl')

print(df['sentence'][0])
# print(type(df['sentence'][0]))
# print the deserialized DataFrame
print(df)
english_df = df[df['lang'] == 'en']
dutch_df = df[df['lang'] == 'nl']

print("english stuff ...")
print(english_df)

print("dutch stuff ... ")
print(dutch_df)

# write to a file examples training
with open("sentences.txt", "w", encoding='utf-8') as f:
    for index, row in df.iterrows():
        f.write(f"{row['lang']}|{row['sentence']}\n")


