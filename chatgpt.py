import os
import openai
import pandas as pd
openai.api_key = "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"


testset_high = pd.read_pickle('prompts/testset_high.pkl')

def answer_gen(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        frequency_penalty=1
    )
    text = response.choices[0].text
    return text.strip()


already = []
with open('logs/davinci002_high.txt', 'r') as f:
    for line in f.readlines():
        already.append(line.split()[0])
with open('logs/davinci002_high_2.txt', 'r') as f:
    for line in f.readlines():
        already.append(line.split()[0])

# i = 0
len_total = len(testset_high)
for x in testset_high:
    # print(i+1, "/", len_total)
    # i+=1
    if x in already:
        # print("Skip")
        continue
    else:
        ans = answer_gen(testset_high[x]['prompt'])
        print(f"{x}\t{ans.strip()}\t{testset_high[x]['answer']}")