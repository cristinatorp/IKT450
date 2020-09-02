import csv
import math
import pdb

data = csv.reader(open("./titanic.csv", "r"))
all_data = []


def getOldYoung(age):
    try:
        if int(age) < 18:
            return "Young"
    except ValueError:
        pass
    return "Old"


for d in data:
    # print (d)
    # training_data.append([d[0],d[1],d[3],d[5]]) #Survived,class,sex
    all_data.append([d[0], d[1], d[3], getOldYoung(d[5])])
    # Survived,class,sex,age
all_data = all_data[1:]

print(all_data)

training_data = all_data[int(len(all_data) / 2):]
verification_data = all_data[:int(len(all_data) / 2)]


def entropy(oneclass):
    pos = len([i for i in oneclass if i[0] == "0"])
    neg = len([i for i in oneclass if i[0] == "1"])
    total = pos + neg
    if min((pos, neg)) == 0:
        return 0
    entropy = - (pos / total) * math.log(pos / total, 2) - (neg / total) * math.log(neg / total, 2)
    return entropy


def split(data, attribute, remove=False):
    # import pdb;pdb.set_trace()
    retvals = {}
    all_attributes = set([i[attribute] for i in data])
    for d in data:
        c = d[attribute]
        aList = retvals.get(c, [])
        if remove:
            d.pop(attribute)
        aList.append(d)
        retvals[c] = aList
    # pdb.set_trace()
    return retvals


def getHighestGain(oneclass):
    before = entropy(oneclass)
    classes = [i for i in range(1, len(oneclass[0]))]
    entropies = [gain(oneclass, c) for c in classes]
    return entropies.index(min(entropies)) + 1


def isPure(oneclass):
    classes = [i for i in range(1, len(oneclass[0]))]
    for c in classes:
        if len(set([i[c] for i in oneclass])) > 1:
            return False
    return True


def isEmpty(oneclass):
    return len(oneclass[0]) <= 1


def mostCommon(oneclass):
    lst = [i[0] for i in oneclass]
    return max(set(lst), key=lst.count)


def confidence(oneclass):
    most_common = mostCommon(oneclass)
    return len([i[0] for i in oneclass if i[0] == most_common]) / len(oneclass)


def gain(oneclass, attribute):
    d = [(entropy(i), len(i)) for i in split(oneclass, attribute, False).values()]
    nAll = sum([i[1] for i in d])
    gain = sum([(i[0] * i[1]) / nAll for i in d])
    return gain


actualClassifier = "def classify(data):"


def buildTree(oneclass, spaces="    "):
    global actualClassifier
    if isEmpty(oneclass) or isPure(oneclass):
        print(spaces, "then ", mostCommon(oneclass))
        print(spaces, "#confidence", confidence(oneclass))
        actualClassifier += "\n" + spaces + "return (" + mostCommon(oneclass) + ")"
        return
    highest = getHighestGain(oneclass)
    d = split(oneclass, highest)
    for key, value in d.items():
        print(spaces, "if ", key)  # ,"(item", highest,")")
        actualClassifier += "\n" + spaces + "if(data[" + str(highest) + "]==\"" + str(key) + "\"):"
        buildTree(value, spaces + "   ")


buildTree(training_data)
print(actualClassifier)


def gain(oneclass, attribute):
    d = [(entropy(i), len(i)) for i in split(oneclass, attribute, False).values()]
    nAll = sum([i[1] for i in d])
    gain = sum([(i[0] * i[1]) / nAll for i in d])
    return gain


exec(actualClassifier)
correct, wrong = 0, 0
for data in verification_data:
    if int(data[0]) == int(classify(data)):
        correct += 1
    else:
        wrong += 1
print("Correct classifications", correct)
print("Wrong classifications", wrong)
print("Accuracy", correct / (correct + wrong))
