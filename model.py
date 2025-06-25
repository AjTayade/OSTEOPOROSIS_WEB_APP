

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.proportion import proportions_ztest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix




data = pd.read_csv("C:\\Users\\ajayt\\Downloads\\Telegram Desktop\\osteoporosis.csv")
data.head()





data.describe()




data.info()





data.isna().sum() 





# Handle None data

# Handle None data for the Alcohol Consumption column
data['Alcohol Consumption'] = data['Alcohol Consumption'].fillna('None')
# Handle None data for the Medical Conditions column
data['Medical Conditions'] = data['Medical Conditions'].fillna('None')

# Handle None data for the Medications column
data['Medications'] = data['Medications'].fillna('None')





# Data visualization 
# Histogram of the Age column
plt.hist(data['Age'], bins=20, edgecolor='black')  # bins is the number of bins, edgecolor creates a border for the bars
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()




sns.boxplot(x=data['Age'])
plt.title('Boxplot of the Age column')
plt.xlabel('Age')
plt.show()


# In[ ]:


# Statistics of age
age = data['Age']
Q1 = age.quantile(0.25)
Q3 = age.quantile(0.75)
IQR = Q3 - Q1

lowerbound = Q1 - 1.5 * IQR
upperbound = Q3 + 1.5 * IQR
print(f"Mean: {age.mean()}")
print(f"Median: {age.median()}")
print(f"Mode: {age.mode()[0]}")
print(f"Variance: {age.var()}")
print(f"Standard deviation: {age.std()}")
print(f"Range: {age.max() - age.min()}")
print(f"Min: {age.min()}")
print(f"Max: {age.max()}")
print(f"Q1: {age.quantile(0.25)}")
print(f"Q3: {age.quantile(0.75)}")
print(f"IQR: {age.quantile(0.75) - age.quantile(0.25)}")
print(f"Lowerbound: {lowerbound}")
print(f"Upperbound: {upperbound}")


# In[ ]:


# Manual Calculation
age = data['Age'].tolist()  # Convert the Age column to a list

n = len(age)
age_sorted = sorted(age)

# Mean
mean = sum(age) / n

# Median
if n % 2 == 0:
    median = (age_sorted[n // 2 - 1] + age_sorted[n // 2]) / 2
else:
    median = age_sorted[n // 2]

# Mode
counts = {}
for x in age:
    counts[x] = counts.get(x, 0) + 1
max_count = max(counts.values())
mode = [k for k, v in counts.items() if v == max_count][0]

# Variance
variance = sum((x - mean) ** 2 for x in age) / (n - 1)

# Standard Deviation
std_dev = variance ** 0.5

# Range
range_val = max(age) - min(age)

# Q1 and Q3
q1 = age_sorted[n // 4]
q3 = age_sorted[3 * n // 4]

# IQR
iqr = q3 - q1

# Lowerbound and Upperbound
lowerbound = q1 - 1.5 * iqr
upperbound = q3 + 1.5 * iqr

# Print results
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode} (appears most frequently)")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")
print(f"Range: {range_val}")
print(f"Q1: {q1}")
print(f"Q3: {q3}")
print(f"IQR: {iqr}")
print(f"Lowerbound: {lowerbound}")
print(f"Upperbound: {upperbound}")


# In[ ]:


# Draw Pie Chart for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns  # Get categorical columns
sns.set_palette('pastel') 
for col in categorical_cols:
    plt.figure(figsize=(6, 6))
    data[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Pie Chart of {col}')
    plt.ylabel('')  # Remove the y-axis label to make the chart cleaner
    plt.show()


# In[ ]:


plt.figure(figsize=(6, 6))
data['Osteoporosis'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('Pie Chart of Osteoporosis')
plt.ylabel('')  
plt.show()


# In[ ]:


# Distribution of columns by Osteoporosis

categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    # Create preprocessed DataFrame
    grouped_data = data.groupby([col, 'Osteoporosis']).size().unstack(fill_value=0)

    # Draw a double bar chart
    plt.figure(figsize=(10, 6))
    grouped_data.plot(kind='bar')

    plt.title(f'Distribution of {col} by Osteoporosis')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Osteoporosis', labels=['None', 'Osteoporosis'])

    plt.tight_layout()
    plt.show()


# In[ ]:


# Draw histogram by age for people with and without the disease
age_osteoporosis = data[data['Osteoporosis'] == 1]['Age']
age_none = data[data['Osteoporosis'] == 0]['Age']

# Draw histogram for Osteoporosis
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(age_osteoporosis, bins=20, edgecolor='black')
plt.title('Age Distribution (Osteoporosis)')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Draw histogram for None
plt.subplot(1, 2, 2)
plt.hist(age_none, bins=20, edgecolor='black')
plt.title('Age Distribution (None)')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[ ]:


# Prepare data
# Drop the Id column because this column does not have valuable information for data evaluation
data = data.drop(columns=["Id"], errors='ignore')
data.head()


# In[ ]:


# Encode categorical variables using LabelEncoder

label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

for col, le in label_encoders.items():
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f'Mapping for {col}: {mapping}\n')


# In[ ]:


data.head()


# In[ ]:


# Select input variables and target variable
X = data.drop(columns=["Osteoporosis"])
y = data["Osteoporosis"]
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Normalize data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[ ]:


'''
165 samples (TN): Correctly predicted as no osteoporosis.
150 samples (TP): Correctly predicted as osteoporosis.
49 samples (FP): Incorrectly predicted, should have been no osteoporosis but the model predicted osteoporosis.
28 samples (FN): Incorrectly predicted, should have been osteoporosis but the model predicted no osteoporosis.
'''


# In[ ]:


feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
print(feature_importance)


# In[ ]:


print(data.groupby("Calcium Intake")["Osteoporosis"].mean())


# In[ ]:


plt.figure(figsize=(10, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:


# Train models and evaluate using important metrics
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
results_df = pd.DataFrame(results).T
print(results_df)


# In[ ]:


# Retrain Decision Tree with max_depth=5
dt_selected = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_selected.fit(X_train, y_train)
y_pred_selected = dt_selected.predict(X_test)

# Reevaluate Decision Tree after feature selection
print("🔹 Decision Tree (max_depth=5) 🔹")
print(f"Accuracy: {accuracy_score(y_test, y_pred_selected):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_selected):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_selected):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_selected):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_selected):.4f}")


# In[ ]:


import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

# Train Gradient Boosting model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Decision Tree Model (unchanged)
decision_tree_model = dt_selected

# Get Feature Importance values
gb_importances = gb_model.feature_importances_
dt_importances = decision_tree_model.feature_importances_

# Get feature names
feature_names = X.columns  

# Create a DataFrame to compare Feature Importance
df_importance = pd.DataFrame({
    'Feature': feature_names,
    'Decision Tree Importance': dt_importances,
    'Gradient Boosting Importance': gb_importances
})

# Plot comparison chart
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Decision Tree Feature Importance
ax[0].barh(df_importance['Feature'], df_importance['Decision Tree Importance'], color='skyblue')
ax[0].set_title('Decision Tree - Feature Importance')
ax[0].set_xlabel('Importance')

# Gradient Boosting Feature Importance
ax[1].barh(df_importance['Feature'], df_importance['Gradient Boosting Importance'], color='salmon')
ax[1].set_title('Gradient Boosting - Feature Importance')
ax[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()


# In[ ]:


# Create DataFrame to compare Feature Importance
df_importance = pd.DataFrame({
    'Feature': feature_names,
    'Decision Tree': dt_importances,
    'Gradient Boosting': gb_importances  # Replace AdaBoost with Gradient Boosting
})

# Sort by Gradient Boosting importance in descending order
df_importance = df_importance.sort_values(by='Gradient Boosting', ascending=False)

# Display the result table
print(df_importance)


# In[ ]:


import scipy.stats as stats
import numpy as np
# T-test for two independent samples
# 1. Set hypothesis
'''
H₀: There is no difference in the mean age between the two groups.
H₁: There is a difference in the mean age between the two groups.
'''

# 🔹 2. Determine the critical region
alpha = 0.05  # Significance level
df = len(age_none) + len(age_osteoporosis) - 2  # Degrees of freedom
t_critical = stats.t.ppf(1 - alpha/2, df)  # Critical t-value for two-tailed test

# 3. Calculate the test statistic
t_stat, p_value = stats.ttest_ind(age_none, age_osteoporosis)

# 4. Compare test statistic with the critical region
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Critical t-value: ±{t_critical:.4f}")

if abs(t_stat) > t_critical:
    print("Reject H₀: There is a significant difference in the mean age between the two groups.")
else:
    print("Not enough evidence to reject H₀: There is no significant difference in the mean age between the two groups.")


# In[ ]:


# One-sample T-test
'''
H₀: The population mean age is equal to 55 years.
H₁: The population mean age is not equal to 55 years.
'''
mu_0 = 55  # Hypothesis value for the mean age
alpha = 0.05  # Significance level

# Calculate the test statistic value
t_stat, p_value = stats.ttest_1samp(age_osteoporosis, mu_0)

# Find the critical t-value for a two-tailed t-distribution
df = len(age_osteoporosis) - 1  # Degrees of freedom (n-1)
t_critical = stats.t.ppf(1 - alpha/2, df)  # Critical value for two-tailed test

# Print results
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Critical t-value: ±{t_critical:.4f}")

# Compare with the significance level alpha = 0.05
if abs(t_stat) > t_critical:
    print("Reject H₀: The mean age is significantly different from 55 years.")
else:
    print("Not enough evidence to reject H₀: The mean age is not significantly different from 55 years.")


# In[ ]:


# 
'''
H₀: The osteoporosis prevalence in the population is equal to 30%.
H₁: The osteoporosis prevalence in the population is not equal to 30%.
'''
n_osteoporosis = data['Osteoporosis'].sum()
# Total number of patients in the data
n_total = len(data)

# Point estimation for osteoporosis prevalence
p_hat = n_osteoporosis / n_total
print(f"Point estimation for osteoporosis prevalence: {p_hat:.4f}")

# Hypothesis value for osteoporosis prevalence
p_0 = 0.3  # Assuming the expected prevalence is 30%

# Z-test for one sample proportion
z_stat, p_value = proportions_ztest(count=n_osteoporosis, nobs=n_total, value=p_0, alternative='two-sided')

# Critical z-value with alpha = 0.05 (two-tailed)
alpha = 0.05
z_critical = stats.norm.ppf(1 - alpha/2)

# Print results
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Critical z-value: ±{z_critical:.4f}")

# Compare with the significance level alpha = 0.05
if abs(z_stat) > z_critical:
    print("Reject H₀: The osteoporosis prevalence is significantly different from 30%.")
else:
    print("Not enough evidence to reject H₀: The osteoporosis prevalence is not significantly different from 30%.")


# In[ ]:


# Build Confidence Interval (CI) for the mean age of the osteoporosis group
# Calculate the mean and standard deviation
mean_age = np.mean(age_osteoporosis)
std_age = np.std(age_osteoporosis, ddof=1)
n_samples = len(age_osteoporosis)

# Calculate the t-score for a 95% confidence level
confidence_level = 0.95
t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n_samples-1)

# Confidence Interval
margin_of_error = t_critical * (std_age / np.sqrt(n_samples))
lower_bound = mean_age - margin_of_error
upper_bound = mean_age + margin_of_error

# Results
print(f"95% Confidence Interval for the mean age of the osteoporosis group: ({lower_bound:.2f}, {upper_bound:.2f})")


# In[ ]:


# Build Confidence Interval (CI) for the mean age of the non-osteoporosis group
# Calculate the mean and standard deviation
mean_age = np.mean(age_none)
std_age = np.std(age_none, ddof=1)
n_samples = len(age_none)

# Calculate the t-score for a 95% confidence level
confidence_level = 0.95
t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n_samples-1)

# Confidence Interval
margin_of_error = t_critical * (std_age / np.sqrt(n_samples))
lower_bound = mean_age - margin_of_error
upper_bound = mean_age + margin_of_error

# Results
print(f"95% Confidence Interval for the mean age of the non-osteoporosis group: ({lower_bound:.2f}, {upper_bound:.2f})")


# In[ ]:


# Point estimation
mean_age_healthy = np.mean(age_none)
mean_age_osteoporosis = np.mean(age_osteoporosis)

print(f"Point estimation for the mean age of the non-osteoporosis group: {mean_age_healthy:.2f}")
print(f"Point estimation for the mean age of the osteoporosis group: {mean_age_osteoporosis:.2f}")


# In[ ]:


'''
 (Null Hypothesis): The osteoporosis rate in the calcium intake group is not lower than in the no calcium intake group (p1 = p2).
 (Alternative Hypothesis): The calcium intake group has a lower osteoporosis rate than the no calcium intake group.
'''

# Split the data into two groups: Calcium Intake (0) and No Calcium Intake (1)
group_calcium = data[data['Calcium Intake'] == 0]
group_no_calcium = data[data['Calcium Intake'] == 1]

# Count the number of osteoporosis cases in each group
x1 = group_calcium['Osteoporosis'].sum()  # Number of osteoporosis cases in the calcium group
n1 = len(group_calcium)  # Total number of people in the calcium group

x2 = group_no_calcium['Osteoporosis'].sum()  # Number of osteoporosis cases in the no calcium group
n2 = len(group_no_calcium)  # Total number of people in the no calcium group

# One-sided Z-test (calcium group has a lower osteoporosis rate)
z_stat, p_value = proportions_ztest(count=[x1, x2], nobs=[n1, n2], alternative='smaller')

# Critical z-value with alpha = 0.05 (one-sided)
alpha = 0.05
z_critical = stats.norm.ppf(1 - alpha)  # One-sided critical value

# Print results
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Critical z-value: {z_critical:.4f}")

# Conclusion
if z_critical < z_stat:  # If z-stat is too small, reject H0
    print("Reject : The calcium intake group has a significantly lower osteoporosis rate than the no calcium intake group.")
else:
    print("Not enough evidence to reject H₀: Cannot conclude that the calcium intake group has a lower osteoporosis rate.")


# In[ ]:


from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as stats


'''
(Null Hypothesis): There is no difference in the osteoporosis rate between the two groups (p1 = p2).
(Alternative Hypothesis): There is a significant difference in the osteoporosis rate between the two groups (p1 ≠ p2).
'''

# Group data by calcium intake (0 for calcium, 1 for no calcium)
group_calcium = data[data['Calcium Intake'] == 0]
group_no_calcium = data[data['Calcium Intake'] == 1]

# Count the number of osteoporosis cases in each group
x1 = group_calcium['Osteoporosis'].sum()  # Number of osteoporosis cases in the calcium group
n1 = len(group_calcium)  # Total number of people in the calcium group

x2 = group_no_calcium['Osteoporosis'].sum()  # Number of osteoporosis cases in the no calcium group
n2 = len(group_no_calcium)  # Total number of people in the no calcium group

# Perform Z-test for two proportions
z_stat, p_value = proportions_ztest(count=[x1, x2], nobs=[n1, n2], alternative='two-sided')

# Critical z-value with alpha = 0.05 (two-sided)
alpha = 0.05
z_critical = stats.norm.ppf(1 - alpha/2)

# Print the results
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Critical z-value: ±{z_critical:.4f}")

# Conclusion
if abs(z_stat) > z_critical:
    print("Reject H₀: There is a significant difference in the osteoporosis rate between the calcium and no calcium groups.")
else:
    print("Not enough evidence to reject H₀: There is no significant difference in the osteoporosis rate between the two groups.")


# In[ ]:


print("Please enter the following information about the individual:")
print("Use only the values exactly as shown in the parentheses.")

# Dictionary to take input from user
input_data = {}

# Ask user for values (with hint labels)
input_data['Age'] = int(input("Enter Age (18 - 90): "))

input_data['Gender'] = input("Enter Gender (Male / Female): ").strip()

# Automatically handle Hormonal Changes for males
if input_data['Gender'].lower() == "male":
    input_data['Hormonal Changes'] = "Normal"
    print("Hormonal Changes set to 'Normal' automatically for Male.")
else:
    input_data['Hormonal Changes'] = input("Enter Hormonal Changes (Normal / Postmenopausal): ").strip()

input_data['Family History'] = input("Family History (Yes / No): ").strip()
input_data['Race/Ethnicity'] = input("Enter Race (Asian / Caucasian / African American): ").strip()
input_data['Body Weight'] = input("Enter Body Weight (Normal / Underweight): ").strip()
input_data['Calcium Intake'] = input("Enter Calcium Intake (Adequate / Low): ").strip()
input_data['Vitamin D Intake'] = input("Enter Vitamin D Intake (Sufficient / Insufficient): ").strip()
input_data['Physical Activity'] = input("Physical Activity (Active / Sedentary): ").strip()
input_data['Smoking'] = input("Smoking? (Yes / No): ").strip()
input_data['Alcohol Consumption'] = input("Alcohol Consumption (Moderate / None): ").strip()
input_data['Medical Conditions'] = input("Medical Conditions (None / Hyperthyroidism / Rheumatoid Arthritis): ").strip()
input_data['Medications'] = input("Medications (None / Corticosteroids): ").strip()
input_data['Prior Fractures'] = input("Prior Fractures? (Yes / No): ").strip()


# In[ ]:


# Convert to DataFrame
user_df = pd.DataFrame([input_data])

# Apply Label Encoders used during training
# Apply Label Encoders only to categorical columns
for col in user_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        try:
            user_df[col] = le.transform(user_df[col])
        except ValueError as e:
            print(f"Invalid value for '{col}': {user_df[col].values[0]}")
            print(f"Allowed values: {list(le.classes_)}")
            raise e



# In[ ]:


# Scale using the same scaler
user_scaled = scaler.transform(user_df)

# Predict
pred = model.predict(user_scaled)[0]
prob = model.predict_proba(user_scaled)[0][1]

# Output
if pred == 1:
    print(" Prediction: The person is likely to have **osteoporosis**.")
else:
    print(" Prediction: The person is **not likely** to have osteoporosis.")

print(f" Confidence (probability of osteoporosis): {prob:.2f}")


# In[ ]:




