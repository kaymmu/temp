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

