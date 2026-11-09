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
