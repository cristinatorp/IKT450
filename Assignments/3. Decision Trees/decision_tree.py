import pandas as pd
import math

df = pd.read_csv("./data/breast-cancer.data", header=None)
df = df.rename(columns={
    0: "Label", 
    1: "Age", 
    2: "Menopause",
    3: "Tumor Size",
    4: "Inv Nodes",
    5: "Node Caps",
    6: "Malignance Degree",
    7: "Breast",
    8: "Breast Quadrant",
    9: "Irradiated"
})


"""
ASSUMPTIONS:

Age:
  - Young if between 20-39
  - Aging if between 40-59
  - Old if 60 or more
Tumor size:
  - Small if between 0-19
  - Medium if between 20-39
  - Large if 40 or more
"""


def get_age(age_range):
    age = int(age_range.split("-")[1])
    try:
        if age < 40:
            return "Young"
        if age < 60:
            return "Aging"
    except ValueError:
        pass
    return "Old"


def get_tumor_size(size_range):
    size = int(size_range.split("-")[1])
    try:
        if size < 20:
            return "Small"
        if size < 40:
            return "Medium"
    except ValueError:
        pass
    return "Large"


def get_breast_quad(quad):
    try:
        if quad == "left_low":
            return "Left Low"
        if quad == "left_up":
            return "Left Up"
        if quad == "right_up":
            return "Right Up"
        if quad == "right_low":
            return "Right Low"
    except ValueError:
        pass
    return "Central"


def get_degree(degree):
    try:
        if degree == 1:
            return "Degree 1"
        if degree == 2:
            return "Degree 2"
    except ValueError:
        pass
    return "Degree 3"


def get_label(label):
    if label == "no-recurrence-events":
        return 1
    return 0


# Refactor the relevant columns
df["Age"] = df["Age"].apply(lambda x: get_age(x))
df["Tumor Size"] = df["Tumor Size"].apply(lambda x: get_tumor_size(x))
df["Breast Quadrant"] = df["Breast Quadrant"].apply(lambda x: get_breast_quad(x))
df["Malignance Degree"] = df["Malignance Degree"].apply(lambda x: get_degree(x))
df["Label"] = df["Label"].apply(lambda x: get_label(x))


# Extract the desired columns to a new dataframe
filtered_df = df.filter(["Label", "Age", "Tumor Size", "Malignance Degree"], axis=1)
filtered_df.head(15)


# TRAIN / TEST SPLIT
seed = 200
training_size = 0.5

df_training_data = filtered_df.sample(frac=training_size, random_state=seed)
df_test_data = filtered_df.drop(df_training_data.index).sample(frac=1.0, random_state=seed)
training_data, test_data = df_training_data.to_numpy().tolist(), df_test_data.to_numpy().tolist()

X_train = [i[1:] for i in training_data]
Y_train = [i[0] for i in training_data]

X_test = [i[1:] for i in test_data]
Y_test = [i[0] for i in test_data]


"""
DECISION TREE FUNCTIONS

Entropy
* pos = number of first label
* neg = number of second label
* p_pos = pos / (pos + neg)
* p_neg = neg / (pos + neg)
* If either pos or neg is 0, return 0
* Else, return the entropy

Get highest gain
1. Find out how many classes there are in the given node, add the range as a list
2. Use the gain function for each individual class
3. Return the index of the minimum entropy value (this is the highest gain value)
    * This index will be used to further split the child nodes

Gain
1. Loop through all values of the split dictionary and measure the entropy for each attribute in the class (i.e. "Old")
2. Each subset in the main set consists of two values: its entropy and its length
3. Returns the gain of the class that was passed to it:

Split
Splits the node into its class values.
* All unique attributes in the argument `node` will be stored in a set `all_attributes`
    * i.e. `{"Young", "Aging", "Old"}`
* Loops through each element in the node and adds them to their own class list
* Returns a dictionary containing all the class lists

Build tree
This function is recursive, and returns when the node is either empty or pure.
1. Takes a node as a parameter
2. If the node is empty or pure:
    1. Find the most common label
    2. Add it to the classifier as a return value
    3. Return from the recursive function
3. Else:
    1. Find the class that has the highest gain
    2. Split on said class
    3. Add the necessary `if` sentence to the classifier
    4. Call `build_tree` again on each child of the class (will propagate down each branch)
"""


def entropy(node):
    pos = len([i for i in node if i[0] == 0])
    neg = len([i for i in node if i[0] == 1])
    total = pos + neg
    if min(pos, neg) == 0:
        return 0
    p_pos, p_neg = (pos / total), (neg / total)
    entropy = - p_pos * math.log(p_pos, 2) - p_neg * math.log(p_neg, 2)
    return entropy


def get_highest_gain(node):
    before = entropy(node)
    # print("First node", node[0])
    classes = [i for i in range(1, len(node[0]))]
    # print("Classes", classes)
    entropies = [gain(node, c) for c in classes]
    # print("Entropies", entropies)
    return entropies.index(min(entropies)) + 1


def gain(node, attribute):
    main_set = [(entropy(i), len(i)) for i in split(node, attribute).values()]
    # print("Entropy, length of split values", main_set)
    n_all = sum([subset[1] for subset in main_set])
    gain = sum([(subset[0] * subset[1]) / n_all for subset in main_set])
    return gain


def split(node, attribute_index, remove=False):
    retvals = {}
    all_attributes = set([n[attribute_index] for n in node])
    # print("All attributes", all_attributes)
    for n in node:
        c = n[attribute_index]
        a_list = retvals.get(c, [])
        if remove:
            n.pop(node)
        a_list.append(n)
        retvals[c] = a_list
    return retvals


# Convenience functions
def is_pure(node):
    classes = [i for i in range(1, len(node[0]))]
    for c in classes:
        if len(set([i[c] for i in node])) > 1:
            return False
    return True


def is_empty(node):
    return len(node[0]) <= 1


def most_common(node):
    label_list = [i[0] for i in node]
    return max(set(label_list), key=label_list.count)


def confidence(node):
    most_common_value = most_common(node)
    return len([i[0] for i in node if i[0] == most_common_value]) / len(node)


# Initiate the actual classifier
actual_classifier = "def classify(data):"


def build_tree(node, spaces="    "):
    global actual_classifier
    if is_empty(node) or is_pure(node):
        # print(f"Empty: {is_empty(node)}, Pure: {is_pure(node)}")
        most_common_value = most_common(node)
        print(f"{spaces}then {most_common_value}")
        print(f"{spaces}# confidence {confidence(node):.2f}")
        actual_classifier += f"\n{spaces}return {most_common_value}" 
        return
    
    highest = get_highest_gain(node)
    d = split(node, highest)
    for key, value in d.items():
        print(f"{spaces}if {key}:")
        actual_classifier += f"\n{spaces}if data[{highest}] == \"{key}\":"
        build_tree(value, spaces + "  ")


print("Printing pseudocode:")
build_tree(training_data)
print("\n\n\n")
print(actual_classifier)
print("\n\n\n")


# TEST CLASSIFIER
exec(actual_classifier)
correct, wrong = 0, 0

for data in test_data:
    if data[0] == classify(data):
        correct += 1
    else:
        wrong += 1

    
accuracy = round((correct / (correct + wrong)) * 100)
print(f"Correct classifications {correct}")
print(f"Wrong classifications {wrong}")
print(f"Accuracy {accuracy}%")
