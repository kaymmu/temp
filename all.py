# ============================================
# DATA MINING LAB: FEATURE SELECTION
# 1. Information Gain (Entropy)
# 2. Pearson Correlation
# 3. ANOVA F-Test
# ============================================

import pandas as pd
import math

# Load dataset
df = pd.read_csv("test.csv")

print("Dataset:")
print(df.head())

# Separate features and target
X = df.drop(columns=["Record_ID", "Class"], errors="ignore")
Y = df["Class"]

features = list(X.columns)
n = len(Y)


# ============================================
# PART 1: INFORMATION GAIN (ENTROPY)
# ============================================

# Calculate entropy manually

def entropy(data):
    classes = list(set(data))
    total = len(data)
    ent = 0
    for c in classes:
        count = 0
        for value in data:
            if value == c:
                count += 1
        p = count / total

        if p > 0:
            ent = ent - p * math.log2(p)

    return ent


# Discretize continuous features into 5 equal-width bins

def discretize(data):

    minimum = min(data)
    maximum = max(data)
    bins = 5
    width = (maximum - minimum) / bins
    result = []

    for value in data:
        if width == 0:
            result.append(0)
        else:
            b = int((value - minimum) / width)

            if b >= bins:
                b = bins - 1

            result.append(b)

    return result


# Calculate Information Gain

parent_entropy = entropy(Y)
ig_scores = {}

for feature in features:
    values = discretize(X[feature])
    unique_values = list(set(values))
    weighted_entropy = 0
    for v in unique_values:
        subset = []
        for i in range(n):
            if values[i] == v:
                subset.append(Y.iloc[i])

        weight = len(subset) / n
        weighted_entropy += weight * entropy(subset)

    ig = parent_entropy - weighted_entropy
    ig_scores[feature] = ig


# Rank Information Gain

ig_rank = sorted(ig_scores.items(),
                 key=lambda x: x[1],
                 reverse=True)

print("\n================================")
print("INFORMATION GAIN RESULTS")
print("================================")

for feature, score in ig_rank:
    print(feature, ":", round(score, 5))

top_ig = [feature for feature, score in ig_rank[:10]]

print("\nTop 10 Features (Information Gain):")
print(top_ig)


# ============================================
# PART 2: PEARSON CORRELATION
# ============================================

def pearson(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = 0
    sum_x = 0
    sum_y = 0

    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        numerator += dx * dy
        sum_x += dx ** 2
        sum_y += dy ** 2

    denominator = math.sqrt(sum_x * sum_y)

    if denominator == 0:
        return 0

    return numerator / denominator


pearson_scores = {}

for feature in features:
    x = list(X[feature])
    y = list(Y)
    r = pearson(x, y)
    pearson_scores[feature] = abs(r)


# Rank Pearson Correlation

pearson_rank = sorted(pearson_scores.items(),
                      key=lambda x: x[1],
                      reverse=True)

print("\n================================")
print("PEARSON CORRELATION RESULTS")
print("================================")

for feature, score in pearson_rank:
    print(feature, ":", round(score, 5))

top_pearson = [feature for feature, score in pearson_rank[:10]]

print("\nTop 10 Features (Pearson):")
print(top_pearson)


# ============================================
# PART 3: ANOVA F-TEST
# ============================================

def anova_f(x, y):
    classes = list(set(y))
    total = len(x)
    grand_mean = sum(x) / total
    between = 0
    within = 0

    for c in classes:
        group = []
        for i in range(total):
            if y[i] == c:
                group.append(x[i])

        group_mean = sum(group) / len(group)

        # Between-group sum of squares
        between += len(group) * (group_mean - grand_mean) ** 2

        # Within-group sum of squares
        for value in group:
            within += (value - group_mean) ** 2

    k = len(classes)
    df_between = k - 1
    df_within = total - k
    ms_between = between / df_between
    ms_within = within / df_within

    if ms_within == 0:
        return 0

    return ms_between / ms_within


anova_scores = {}

for feature in features:
    x = list(X[feature])
    y = list(Y)
    f = anova_f(x, y)
    anova_scores[feature] = f

# Rank ANOVA

anova_rank = sorted(anova_scores.items(),
                    key=lambda x: x[1],
                    reverse=True)

print("\n================================")
print("ANOVA F-TEST RESULTS")
print("================================")

for feature, score in anova_rank:
    print(feature, ":", round(score, 5))

top_anova = [feature for feature, score in anova_rank[:10]]

print("\nTop 10 Features (ANOVA):")
print(top_anova)


# ============================================
# PART 4: COMPARISON OF METHODS
# ============================================

print("\n================================")
print("COMPARISON OF FEATURE SELECTION")
print("================================")

# Common features between Information Gain and Pearson

common = []

for feature in top_ig:
    if feature in top_pearson:
        common.append(feature)

print("\nCommon: Information Gain & Pearson")
print(common)


# Features selected only by Information Gain

only_ig = []

for feature in top_ig:
    if feature not in top_pearson:
        only_ig.append(feature)

print("\nOnly Information Gain:")
print(only_ig)


# Features selected only by Pearson

only_pearson = []

for feature in top_pearson:
    if feature not in top_ig:
        only_pearson.append(feature)

print("\nOnly Pearson:")
print(only_pearson)


# Compare all three methods

common_three = []

for feature in top_ig:
    if feature in top_pearson and feature in top_anova:
        common_three.append(feature)

print("\nCommon in ALL THREE methods:")
print(common_three)
print("\n================================")
print("FINAL SUMMARY")
print("================================")
print("Information Gain Top 10:", top_ig)
print("Pearson Top 10:", top_pearson)
print("ANOVA Top 10:", top_anova)
print("\nAnalysis completed successfully!")



### entropy long code

import pandas as pd
import math

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("test.csv")

# Separate features and target
X = df.drop(columns=["Record_ID", "Class"], errors="ignore")
Y = df["Class"]

# Convert continuous values into integers
for feature in X.columns:
    X[feature] = X[feature].astype(int)


# ==========================================
# 2. ENTROPY FUNCTION
# ==========================================

def entropy(data):

    total = len(data)
    counts = {}

    for value in data:
        if value not in counts:
            counts[value] = 0
        counts[value] += 1

    result = 0

    for value in counts:
        p = counts[value] / total
        result -= p * math.log2(p)

    return result


# ==========================================
# 3. CALCULATE PARENT ENTROPY
# ==========================================

parent_entropy = entropy(Y)

print("Parent Entropy:", parent_entropy)


# ==========================================
# 4. CALCULATE INFORMATION GAIN
# ==========================================

IG = {}

for feature in X.columns:

    weighted_entropy = 0

    values = X[feature].unique()

    for value in values:

        subset = Y[X[feature] == value]

        weight = len(subset) / len(Y)

        weighted_entropy += weight * entropy(subset)

    information_gain = parent_entropy - weighted_entropy

    IG[feature] = information_gain

    print(feature, "Information Gain:", information_gain)


# ==========================================
# 5. RANK FEATURES
# ==========================================

ranked_features = sorted(
    IG.items(),
    key=lambda item: item[1],
    reverse=True
)

print("\n========== FEATURE RANKING ==========")

for feature, score in ranked_features:
    print(feature, ":", score)


# ==========================================
# 6. SELECT TOP 10 FEATURES
# ==========================================

top_10 = ranked_features[:10]

print("\n========== TOP 10 FEATURES ==========")

for feature, score in top_10:
    print(feature, ":", score)

top_10_features = [feature for feature, score in top_10]

print("\nSelected Features:", top_10_features)




# entropy shortcut
import pandas as pd
import math

df = pd.read_csv("test.csv")

X = df.drop(columns=["Record_ID", "Class"], errors="ignore")
Y = df["Class"]

# Convert float values to integers
X = X.astype(int)

# Entropy function
def entropy(y):
    counts = y.value_counts()
    total = len(y)
    return -sum((c/total) * math.log2(c/total) for c in counts)

# Parent entropy
H = entropy(Y)

# Information Gain
IG = {}

for col in X.columns:
    weighted = 0

    for val in X[col].unique():
        subset = Y[X[col] == val]
        weighted += (len(subset)/len(Y)) * entropy(subset)

    IG[col] = H - weighted

# Print top 10
top10 = sorted(IG, key=IG.get, reverse=True)[:10]

print("Parent Entropy:", H)
print("\nTop 10 Features:")

for f in top10:
    print(f, IG[f])



# counting the maximum only

from collections import Counter

# 1. Combine all lists into one big list
all_features = top_anova + top_ig + top_pearson

# 2. Count the occurrences of each feature
feature_counts = Counter(all_features)

# 3. Get the features sorted from best (most votes) to worst (least votes)
best_to_worst = [feature for feature, count in feature_counts.most_common()]

print(best_to_worst)
# Output: ['f3', 'f1', 'f11', 'f20', 'f14']


# weighted sorting best to worst

from collections import defaultdict

# Your feature lists with the exact same features but different orders
# A = ["f3", "f1", "f11", "f14"]
# B = ["f14", "f3", "f1", "f11"]
# C = ["f1", "f3", "f11", "f14"]

feature_scores = defaultdict(int)

# Loop through each list and give points based on position
for lst in [top_anova, top_ig, top_pearson]:
  for index, feature in enumerate(lst):
    # 1st place (index 0) gets 4 points, 2nd gets 3 points, etc.
    points = len(lst) - index
    feature_scores[feature] += points

# Sort features from highest total points to lowest
best_to_worst = sorted(
    feature_scores.keys(), key=lambda x: feature_scores[x], reverse=True
)

print(best_to_worst)
# Output: ['f3', 'f1', 'f14', 'f11']

