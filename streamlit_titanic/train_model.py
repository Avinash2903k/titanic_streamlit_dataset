import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# 1) Load dataset
df = pd.read_csv("titanic.csv")

print("\nAvailable columns in titanic.csv:")
print(list(df.columns))

# 2) Normalize column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

# 3) Map possible names -> standard names
column_map = {
    "pclass":   ["pclass", "passengerclass", "class", "p_class"],
    "sex":      ["sex", "gender", "male/female", "gender_of_passenger"],
    "age":      ["age", "ages", "age_years"],
    "sibsp":    ["sibsp", "siblings/spouses", "siblings", "sib_sp"],
    "parch":    ["parch", "parents/children", "parents_children", "par_ch"],
    "fare":     ["fare", "ticket_fare", "fareamount", "ticketfare"],
    "survived": ["survived", "survival", "target", "label"]
}

final_cols = {}   # standard_name -> actual_name

for key, options in column_map.items():
    for col in options:
        if col in df.columns:
            final_cols[key] = col
            break

# 4) Check missing
missing = [k for k in column_map.keys() if k not in final_cols]
if missing:
    print("\n❌ Missing important columns in titanic.csv:", missing)
    print("👉 Please rename your columns to one of these options:")
    for k in missing:
        print(f"   {k} -> {column_map[k]}")
    raise SystemExit("Cannot train model without required columns.")

print("\n✅ Detected column mapping:")
for k, v in final_cols.items():
    print(f"   {k}  <--  {v}")

# 5) Rename to standard names
df = df.rename(columns=final_cols)

# 6) Encode gender (if not already numeric)
if df["sex"].dtype == object:
    df["sex"] = df["sex"].str.strip().str.lower()
    df["sex"] = df["sex"].map({"female": 1, "male": 0})

# 7) Keep necessary columns ONLY
df = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "survived"]]

# 8) Drop missing rows
df = df.dropna()

# 9) Split X, y
X = df[["pclass", "sex", "age", "sibsp", "parch", "fare"]]
y = df["survived"].astype(int)

# 10) Train model
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# 11) Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n🎉 Model trained successfully and saved as model.pkl")
