import os
import dotenv
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
dotenv.load_dotenv()

dataset_path = os.getenv('DATASET_PATH')
df = pd.read_csv(os.path.join(dataset_path, "loan_dataset.csv"))

df = df.drop(columns=['Loan_ID'])  # ACTUAL DROP

X = df.drop('Loan_Status', axis=1)
# print(X)
y = df.Loan_Status.map({'Y':1,'N':0})  # string → binary
# print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# print(y_test)

# # 6.1 Identify column lists
num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']
cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
# # Credit_History is actually 0/1 → treat as numeric
#
# # 6.2 Two separate transformers
numeric_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median'))  # median robust to outliers
])
#
categoric_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_pipe, num_cols),
        ('cat', categoric_pipe, cat_cols)
    ])

tree_clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=25,
    class_weight='balanced',  # mitigates the Y/N imbalance
    random_state=42)

model = Pipeline(steps=[('prep', preprocess),
                        ('clf', tree_clf)])


def evaluate(pipeline, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds_te = pipeline.predict(X_te)
    print('Hold-out accuracy: {:.3f}'.format(accuracy_score(y_te, preds_te)))
    print('\nClassification report:\n', classification_report(y_te, preds_te))
    print('Confusion matrix:\n', confusion_matrix(y_te, preds_te))
    # 5-fold cross-val on training data
    cv = cross_val_score(pipeline, X_tr, y_tr, cv=5, scoring='accuracy')
    print('5-fold CV accuracy: {:.3f} ± {:.3f}'.format(cv.mean(), cv.std()))

evaluate(model, X_train, y_train, X_test, y_test)


plt.figure(figsize=(35,8))
# pull the trained tree out of the pipeline
tr = model.named_steps['clf']
# get feature names after preprocessing
ohe = model.named_steps['prep'].named_transformers_['cat'].named_steps['onehot']
cat_features = ohe.get_feature_names_out(cat_cols)
all_features = np.r_[num_cols, cat_features]
plot_tree(tr, feature_names=all_features, class_names=['Reject','Approve'],
          filled=True, rounded=True, fontsize=9)
plt.show()

imp = pd.Series(tr.feature_importances_, index=all_features).sort_values(ascending=False)
print(imp.head(10))
sns.barplot(x=imp.head(10), y=imp.head(10).index)


new_customer = {
    'Gender': 'Female',
    'Married': 'No',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 6500,
    'CoapplicantIncome': 0,
    'LoanAmount': 120,
    'Loan_Amount_Term': 360,
    'Credit_History': 0,   # 1 = good, 0 = bad
    'Property_Area': 'Urban'
}

# turn the dict into a one-row DataFrame
X_new = pd.DataFrame([new_customer])

# 2. PREDICT ------------------------------------------------------------------
pred = model.predict(X_new)[0]          # 0 or 1
prob = model.predict_proba(X_new)[0]    # [P(reject), P(approve)]

print('Prediction :', 'APPROVE' if pred else 'REJECT')
print('Probabilities : Reject={:.2%}  Approve={:.2%}'.format(*prob))

# 3. SHOW THE DECISION PATH (which nodes were visited) ------------------------
#    retrieve the fitted tree from the pipeline

tree_model = model.named_steps['clf']
feature_names = (model.named_steps['prep']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(cat_cols))
feature_names = np.r_[num_cols, feature_names]

# ✅ Transform X_new first
X_new_prepared = model.named_steps['prep'].transform(X_new)

# ✅ Then use the numeric version
node_indicator = tree_model.decision_path(X_new_prepared)
leaf_id = tree_model.apply(X_new_prepared)

print('\nDecision path:')
for node_id in node_indicator.indices:
    if tree_model.tree_.feature[node_id] != _tree.TREE_UNDEFINED:
        feat = feature_names[tree_model.tree_.feature[node_id]]
        threshold = tree_model.tree_.threshold[node_id]
        sign = "<=" if X_new_prepared[0, tree_model.tree_.feature[node_id]] <= threshold else ">"
        print(f"  node {node_id}:  {feat} {sign} {threshold:.3f}")
print(f"=> reached leaf {leaf_id[0]} → class {pred}")