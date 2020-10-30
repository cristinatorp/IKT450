#!/usr/bin/env python
# coding: utf-8

# # Chatbot
# * Make the chatbot so that you classify a category (i.e., tag) of input text
#     * Return a dialog from the correct class
#     * Note that one question could have multiple tags and you may need to simplify
# * Alternatively, make a sequence to sequence network that automatically learns what to respond
#     * It can be character based or word based
# * Hint:  Start with a subset of the dataset
# ---

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
from nltk.stem import *

import re
import random
np.random.seed(7)


# ## Load into dataframes

# In[2]:


questions_df = pd.read_csv("./archive/Questions.csv")
answers_df = pd.read_csv("./archive/Answers.csv")
tags_df = pd.read_csv("./archive/Tags.csv")


# In[3]:


questions_df.head(15)
#questions_df.describe()


# In[4]:


answers_df.head(15)


# In[5]:


tags_df.dropna(inplace=True)
tags_df.head(15)
#tags_df.nunique()
#tags_df.count()


# In[6]:


print("Questions:\t", len(questions_df))
print("Answers:\t", len(answers_df))
print("Tags:\t\t", len(tags_df))
print("\nThese must be the same size!")


# ---
# ## Assumptions and decisions
# * Drop questions without answers
# * Only keep the top answer (the one with the most votes)
# * As all questions are related to python, `python` is not a relevant tag
#     * Drop all tags with `Tag='python'`
#     * Only keep the first tag per `Id` for easier classification
# * **The question, answer, and tag lists must have the same size**
#     * All questions must have an answer and a tag
#     * One answer per question
#     * One tag per question

# In[7]:


# Drop questions without answers
questions_with_answer = questions_df[questions_df['Id'].isin(answers_df['ParentId'])]
print("Questions with answers:", len(questions_with_answer))


# In[8]:


# Drop all python tags
no_python_tags = tags_df[tags_df['Tag'] != 'python']

# Only keep the first tag per question (if several)
no_dup_tags = no_python_tags.drop_duplicates(subset='Id')

# As we have dropped questions, we also need to drop the corresponding tags
tags_with_questions = no_dup_tags[no_dup_tags['Id'].isin(questions_with_answer['Id'])]
print("Tags with questions:", len(tags_with_questions))


# In[9]:


# Drop the questions that no longer have tags
questions_with_tag = questions_with_answer[questions_with_answer['Id'].isin(tags_with_questions['Id'])]
print("Questions with answers and tags:", len(questions_with_tag))


# In[10]:


# Drop answers that no longer have questions
answers_with_questions = answers_df[answers_df['ParentId'].isin(questions_with_tag['Id'])]

# Keep the answer with the highest score, sort them by question ID
grouped_answers = answers_with_questions.sort_values(['Score'], ascending=False).groupby('ParentId').head(1)
sorted_answers = grouped_answers.sort_values(['ParentId'])
print("Top answers: ", len(sorted_answers))


# In[11]:


print("Questions:\t", len(questions_with_tag))
print("Answers:\t", len(sorted_answers))
print("Tags:\t\t", len(tags_with_questions))

if len(questions_with_tag) == len(sorted_answers) and len(questions_with_tag) == len(tags_with_questions):
    print("\nAll are now the same size!")
else:
    print("\nRecalculate!")


# In[12]:


df_questions = questions_with_tag.reset_index(drop=True)
df_answers = sorted_answers.reset_index(drop=True)
df_tags = tags_with_questions.reset_index(drop=True)
num_questions = len(df_questions)


# In[13]:


def detect_indexing_errors():
    detected_errors = False
    for i in trange(num_questions):
        q_id = df_questions.loc[i]['Id']
        a_id = df_answers.loc[i]['ParentId']
        t_id = df_tags.loc[i]['Id']
        is_same = q_id == a_id and q_id == t_id
        if not is_same:
            detected_errors = True
            print(f"Q: {q_id}, A: {a_id}, T: {t_id}")
            break
    return detected_errors


# **NB! This takes a while, do not run unless necessary**

# In[14]:


detected_errors = detect_indexing_errors()
can_proceed = "Yes" if not detected_errors else "NO"
print("Can proceed:", can_proceed)


# ## Preprocessing text
# * Questions is x (ID is index)
#     * Only use the title of the question, as I assume this is the most similar to what will be asked in a chatbot
# * Tags is y (ID is index)
# * Answers are given as response in the chatbot
# * **Testing with a dataset of 1000 entries first**

# In[16]:


train_size = 1000


# In[17]:


x_train_temp = df_questions['Title'].to_numpy()[:train_size]
y_train_temp = df_tags['Tag'].to_numpy()[:train_size].tolist()
html_answers = df_answers['Body'].to_numpy()[:train_size].tolist()


# In[19]:


print(x_train_temp[:20])


# In[20]:


print(y_train_temp[:100])


# In[21]:


def clean_html(raw_html):
    cleanr = re.compile("<.*?>")
    clean_text = re.sub(cleanr, "", raw_html)
    return clean_text


# In[27]:


# Remove all html tags from answers
answers = [clean_html(ans) for ans in html_answers]

print("EXAMPLE:")
print(f"\"{html_answers[0]}\"")
print("\nTURNS INTO:")
print(f"\"{answers[0]}\"")


# ### Getting categories

# In[28]:


categories = list(set(y_train_temp))
print(len(categories))


# In[29]:


# Replace each label with its category index (both x and y here are now numpy arrays)
y_train_org = np.array([categories.index(i) for i in y_train_temp])
x_train_org = x_train_temp[:]

num_classes = len(categories)
y_train = []


# In[30]:


for n in [categories.index(i) for i in y_train_temp]:
    y_train.append([0 for i in range(num_classes)]) # Create an array of 324 zeros (category set length)
    y_train[-1][n] = 1 # Set the index to 1 for the label in mention


# ### Embedding and stemming

# In[31]:


all_words = " ".join(x_train_temp).lower().split(" ")
unique_words = list(set(all_words))

stemmer = PorterStemmer()
# TODO: Actually do some stemming

x_train = []
x_array_size = 10


# **Make text into numbers:**
# * Takes each line and returns an array of ten numbers
# * If ten or more words, append the appropriate `unique_words` index
# * If less than ten words, simply add zeros so the array is always of length 10
# 
# Example:  
# "*Continuous Integration System for a Python Codebase*" → `[1271, 24, 1360, 375, 588, 294, 1789, 0, 0, 0]`

# In[32]:


def make_text_into_numbers(text):
    iwords = text.lower().split(" ")
    numbers = []
    
    for n in iwords:
        try:
            numbers.append(unique_words.index(n))
        except ValueError:
            numbers.append(0)
    
    zeros_array = np.zeros(x_array_size - 1).tolist()
    numbers = numbers + zeros_array # add zeros in case of less than required array size
    return numbers[:x_array_size]


# In[33]:


for i in x_train_temp:
    t = make_text_into_numbers(i)
    x_train.append(t)


# In[34]:


# Assure that each array in x_train is of intended array size
print(set([len(i) for i in x_train]))


# In[35]:


x_train = torch.LongTensor(x_train)
y_train = torch.Tensor(y_train)


# In[36]:


print(x_train)
print(y_train)


# ---
# ## Model

# In[37]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(len(unique_words), 128)
        
        self.lstm = nn.LSTM(input_size = 128,
                           hidden_size = 128,
                           num_layers = 1,
                           batch_first = True,
                           bidirectional = False)
        
        self.fc1 = nn.Linear(128, 512)
        # TODO: A middle layer here?
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, inp):
        e = self.embedding(inp)
        output, hidden = self.lstm(e)
        
        x = self.fc1(output[:, -1, :])
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


# In[40]:


net = Net()
print("Model", net)
print("Parameters", [param.nelement() for param in net.parameters()])


# In[41]:


max_words = 10
batch_size = 1
epochs = 5
learning_rate = 0.001


# In[42]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


# In[43]:


t_loss = []
v_loss = []

t_acc = []
v_acc = []


# In[44]:


def avg(l):
    return sum(l) / len(l)


# ## Train model

# In[45]:


n_steps = 1000

for i in trange(n_steps):
    y_pred_train = net(x_train)
    loss_train = loss_fn(y_pred_train, y_train)
    
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    loss = loss_train.detach().numpy()
    t_loss.append(loss)
    
    # Print loss each 100th round
    if i%100 == 0:
        print(loss)


# ### Plot loss

# In[46]:


x_values = [i for i in range(n_steps)]

plt.plot(x_values, t_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# ## Test model

# In[47]:


def classify(user_input):
    indices = make_text_into_numbers(user_input)
    question_tensor = torch.LongTensor([indices])
    output = net(question_tensor).detach().numpy()
    tag_index = np.argmax(output)
    return tag_index


# In[48]:


print("Chatbot: What can I help you with today?")

user_input = input("You: ")
while user_input != "Bye":
    tag_index = classify(user_input)
    answer = answers[tag_index]
    print(f"Chatbot: {answer}")
    user_input = input("Question: ")

print("Chatbot: Have a nice day! Beep boop")


# In[ ]:




