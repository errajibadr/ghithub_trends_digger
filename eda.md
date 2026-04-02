THE 5-STEP EDA  FLOW
══════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: LOOK AT THE DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Before anything, I want to understand what I'm working with."

  df.shape                      → how many rows and columns
  df.dtypes                     → spot wrong types (salary as string?)
  df.describe(include='all')    → stats for everything
  df.head(10)                   → eyeball the actual values

SAY: "I have 2000 rows, 15 columns. I notice monthly_income
      is stored as 'object' — that should be numeric. I'll fix that."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: WHAT'S MISSING AND WHY?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Now I check for missing values — and more importantly, WHY
 they're missing, because that changes how I fix them."

  FIND what's missing:
  df.isnull().sum().sort_values(ascending=False)

  CHECK why — is one group missing more than others?
  df.groupby('department')['salary'].apply(lambda x: x.isnull().mean())

  Then decide:
  ┌──────────────────────────────────────────────────────────┐
  │ All groups ~same miss rate  → MCAR → fill with median    │
  │ One group way more missing  → MAR  → fill PER GROUP      │
  │ Can't explain from data     → MNAR → flag + fill         │
  └──────────────────────────────────────────────────────────┘

  FIX (pick one):
  df['col'].fillna(df['col'].median())                        # MCAR
  df.groupby('dept')['col'].transform(lambda x: x.fillna(x.median()))  # MAR
  df['col_missing'] = df['col'].isnull().astype(int)          # MNAR flag

SAY: "training_hours is 25% missing for Sales but ~0% for
      other departments. That's MAR — missingness depends on
      department. I'll impute with the Sales-specific median,
      not the global one."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: WHAT DOES EACH COLUMN LOOK LIKE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"I want to check for skew, weird values, and whether
 my target is balanced."

  CHECK skew (one line):
  df.select_dtypes(include='number').skew().sort_values()
  → |skew| > 1 = problem for linear models

  CHECK target balance:
  df['target'].value_counts(normalize=True)
  → if 84/16 or worse = imbalanced = accuracy is a lie

  CHECK for garbage values:
  df.describe()    → look at min/max. Negative age? $0 salary?

  FIX skew:
  df['log_income'] = np.log1p(df['income'])

  FIX garbage:
  df.loc[df['age'] < 0, 'age'] = np.nan       # then impute

SAY: "Income has skew of 2.3 — I'll log-transform for linear
      models. Target is 84/16 — I'll use class_weight='balanced'
      and evaluate with F1 and AUC, not accuracy."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4: HOW DO COLUMNS RELATE TO EACH OTHER?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Now I look at relationships — which features predict the
 target, and which features are redundant."

  WHICH FEATURES MATTER (correlation with target):
  df.corr()['target'].sort_values(key=abs, ascending=False)

  WHICH FEATURES ARE REDUNDANT (correlation with each other):
  df.corr()   → look for pairs > 0.7 = multicollinearity

  GROUP COMPARISONS (categorical vs target):
  df.groupby('overtime')['attrition'].mean()

  VISUALIZE:
  sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0)

SAY: "Satisfaction (-0.35) and overtime (0.28) are the
      strongest predictors. years_at_company and
      years_since_promotion are correlated at 0.85 — I'll
      drop one to avoid multicollinearity."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5: MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Based on what EDA told me, here's my modeling approach."

  ALWAYS DO:
  train_test_split(X, y, stratify=y, test_size=0.2)   # keep target ratio
  Pipeline([('prep', preprocessor), ('clf', model)])   # no data leakage
  cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')  # honest score

  PICK MODEL:
  ┌──────────────────────────────────────────────────────────┐
  │ Always start  → LogisticRegression(class_weight='balanced')
  │ Then try      → RandomForestClassifier(class_weight='balanced')
  │ If time       → GradientBoostingClassifier or XGBoost     │
  └──────────────────────────────────────────────────────────┘

  EVALUATE:
  classification_report(y_test, y_pred)     # precision, recall, F1
  roc_auc_score(y_test, y_prob)             # AUC

SAY: "Because I found non-linear patterns and outliers in EDA,
      Random Forest should outperform Logistic Regression.
      I'm using stratified split to keep the 84/16 ratio,
      and evaluating with AUC because accuracy would be
      misleading at 84% just by guessing the majority class."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE ONE SENTENCE THAT WORKS EVERYWHERE:

  "Because I found _____, I'm doing _____ because _____."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━