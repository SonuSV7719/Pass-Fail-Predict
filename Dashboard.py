import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


label_encoders = {}
# Function to preprocess input data
def preprocess_input(data, label_encoders):
    data_encoded = data.copy()
    for feature, encoder in label_encoders.items():
        data_encoded[feature] = encoder.transform(data_encoded[feature])
    return data_encoded

# Data preprocessing
def label_encode(df):
    le = LabelEncoder()
    df_encoded = df.copy()
    encoding = {}
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            df_encoded[column] = le.fit_transform(df_encoded[column])
            encoding[column] = {'encoder': le, 'classes': le.classes_}
    return df_encoded, encoding

def labelEncode(df):
    le = LabelEncoder()
    dfEncoded = df.copy()
    encoding = {}
    global label_encoders
    try:
        for column in dfEncoded.columns:
            if dfEncoded[column].dtype == 'object':
                dfEncoded[column] = le.fit_transform(dfEncoded[column])
                encoding[column] = {'encoder' : le, 'classes' : le.classes_}
                label_encoders[column] = le
        return dfEncoded, encoding
    except AttributeError:
        dfEncoded["Result"] = le.fit_transform(y)
        encoding["Result"] = {'encoder' : le, 'classes' : le.classes_}
        return dfEncoded, encoding

# Model training
def train_models(x, y):
    models = {
        "LogisticRegression": LogisticRegression(),
        'GaussianNB': GaussianNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'BaggingClassifier': BaggingClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier()
    }

    train_scores = {modelName: [] for modelName in models}
    test_scores = {modelName: [] for modelName in models}

    for modelName, model in models.items():
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
            model.fit(X_train, y_train)
            train_score = accuracy_score(y_train, model.predict(X_train))
            test_score = accuracy_score(y_test, model.predict(X_test))
            train_scores[modelName].append(train_score)
            test_scores[modelName].append(test_score)

    return models, test_scores

# Load data
df = load_data("student123.csv")

# Preprocess data
x = df.drop(columns=["FIRSTNAME", "MIDDLENAME", "LASTNAME", "RESULT", "No", "ENROLLMENTNO", "STUDENTNAME", "ACTUAL_CATEGORY", "COURSE_YEAR"], axis=1)
y = df["RESULT"]
x, _ = label_encode(x)

# Input form
st.sidebar.header("Enter Student Details")
course_name = st.sidebar.selectbox("Course Name", ["Engineering", "Medical", "Arts"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
annual_income = st.sidebar.number_input("Annual Income")
ssc_marks = st.sidebar.number_input("SSC Marks")
hsc_marks = st.sidebar.number_input("HSC Marks")
cet_marks = st.sidebar.number_input("CET Marks")
jee_marks = st.sidebar.number_input("JEE Marks")


# Train models
models, test_scores = train_models(x, y)

# Sidebar
st.sidebar.title('Model Prediction')
st.sidebar.write('Enter student details to predict the result:')


# Main content
st.title('Student Data Analysis and Model Prediction')

# Data Summary
st.write("## Data Summary")
st.write(df.head())


st.write("## Statistical Description")
st.write(df.describe())


#===============================================

# Sidebar - Interactive Filtering
st.sidebar.title('Filter Data')

# Filter by Course
selected_course = st.sidebar.multiselect('Select Course', df['COURSENAME'].unique())

# Filter by Gender
selected_gender = st.sidebar.selectbox('Select Gender', ['Male', 'Female'])

# Filter by Result
selected_result = st.sidebar.multiselect('Select Result', df['RESULT'].unique())

# Apply Filters
filtered_df = df.copy()
if selected_course:
    filtered_df = filtered_df[filtered_df['COURSENAME'].isin(selected_course)]
if selected_gender:
    filtered_df = filtered_df[filtered_df['GENDER'] == selected_gender]
if selected_result:
    filtered_df = filtered_df[filtered_df['RESULT'].isin(selected_result)]

# Main content
st.title('Filtered Data')
st.write(filtered_df)


# Data Visualization
st.write("## Data Visualization")
st.write("### Count Plot of COURSENAME")
figure = plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='COURSENAME', palette='viridis')
plt.xticks(rotation=45)
st.pyplot(figure)


st.write("## Gender wise pass and fail")
figure = plt.figure(figsize=(8, 6))
sns.countplot(df, x = 'RESULT', hue = 'GENDER')
st.pyplot(figure)

st.write("### Count Plot of RESULT")
figure = plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='RESULT', palette='pastel')
st.pyplot(figure)

# Additional Visualizations
st.write("### Distribution of SSC Marks by Course")
figure = plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='SSC_MARKS', hue='COURSENAME', kde=True, palette='Set2', multiple='stack')
plt.title("Distribution of SSC Marks by Course")
plt.xlabel("SSC Marks")
plt.ylabel("Density")
plt.legend(title='Course')
st.pyplot(figure)

st.write("### Box Plot of HSC Marks by Course")
figure = plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='HSC_MARK', y='COURSENAME', palette='Set3')
plt.title("Box Plot of HSC Marks by Course")
plt.xlabel("HSC Marks")
plt.ylabel("Course")
st.pyplot(figure)

# Additional Important Visualizations
st.write("### Distribution of CET Marks by Course")
figure = plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='CET_MARKS', hue='COURSENAME', kde=True, palette='Set2', multiple='stack')
plt.title("Distribution of CET Marks by Course")
plt.xlabel("CET Marks")
plt.ylabel("Density")
plt.legend(title='Course')
st.pyplot(figure)

st.write("### Relationship Between SSC Marks and HSC Marks")
figure = plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='SSC_MARKS', y='HSC_MARK', hue='RESULT', palette='coolwarm')
plt.title("Relationship Between SSC Marks and HSC Marks")
plt.xlabel("SSC Marks")
plt.ylabel("HSC Marks")
st.pyplot(figure)


heatMapData, _ = labelEncode(df.iloc[:, 2:])

st.write("### Correlation Matrix")

figure = plt.figure(figsize=(8, 6))
sns.heatmap(heatMapData.corr())
st.pyplot(figure)

# Select columns for the 3D scatter plot
x_data = df['SSC_MARKS']
y_data = df['HSC_MARK']
z_data = df['CET_MARKS']

# Create a Streamlit app
st.write('### SSC, HSC and CET Marks Relationship')

x_data = 'SSC_MARKS'
y_data = 'HSC_MARK'
z_data = 'CET_MARKS'

# Create the 3D scatter plot
fig = px.scatter_3d(df, x=x_data, y=y_data, z=z_data, color='RESULT', hover_data=['COURSENAME'])
st.plotly_chart(fig)


# Model Performance
st.write("## Model Performance")
figure = plt.figure(figsize=(10, 6))
for model_name in models:
    plt.plot(np.arange(1, 11), test_scores[model_name], label=model_name)
plt.title("Model Performance")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
st.pyplot(figure)

# ---------------------------------------------------------------------------------
# Sidebar - Model Selection Dropdown
selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))
predictButton = st.sidebar.button("Predict")

# Train the selected model
selected_model_instance = models[selected_model]


if predictButton:
    # Preprocess input data for prediction
    input_data = pd.DataFrame({
        "COURSENAME": [course_name],
        "GENDER": [gender],
        "ANNUAL_INCOME": [annual_income],
        "SSC_MARKS": [ssc_marks],
        "HSC_MARK": [hsc_marks],
        "CET_MARKS": [cet_marks],
        "JEE_MARKS": [jee_marks]
    })

    # Label encode selected columns
    input_data_encoded = input_data.copy()
    for column in ['COURSENAME', 'GENDER']:
        input_data_encoded[column] = label_encoders[column].transform(input_data[column])

    # Make prediction
    prediction = selected_model_instance.predict(input_data_encoded)

    # Display prediction
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("The student is likely to get admission.")
    else:
        st.write("The student is unlikely to get admission.")





# print(x)
newX, encoding = labelEncode(x)
le = LabelEncoder()
newY = le.fit_transform(y)

# Train the selected model with data
X_train, X_test, y_train, y_test = train_test_split(newX, newY, test_size=0.2, random_state=42)
selected_model_instance.fit(X_train, y_train)

# Get predictions
# print(X_test[0][0])
y_pred = selected_model_instance.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display model evaluation metrics
st.write(f"## Evaluation Metrics for {selected_model}")
st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1-score: {f1:.2f}")
st.write(f"## Confusion matrix")

st.write(confusion_matrix(y_test, y_pred))
