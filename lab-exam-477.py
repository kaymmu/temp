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
