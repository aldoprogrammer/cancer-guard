import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset from sklearn
data = load_breast_cancer()

# Select only 5 features for training (radius, texture, perimeter, area, and smoothness)
X = data.data[:, [0, 1, 2, 3, 4]]  # First 5 features (mean radius, texture, perimeter, area, smoothness)
y = data.target  # Labels (malignant or benign)

# Train the classifier using Decision Tree with Entropy criterion (as a proxy for Gain Ratio)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scaling features

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Classifier
dt = DecisionTreeClassifier(criterion='entropy')  # Entropy as a proxy for Gain Ratio
dt.fit(X_train, y_train)

# Evaluate the model's accuracy
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title('Breast Cancer Diagnosis and Treatment Suggestions')

# Sidebar for navigation
page = st.sidebar.radio("Choose a page", ["General User", "Doctor's Page"])

if page == "General User":
    # General User Page - Symptom-based self-assessment
    st.header("Breast Cancer Risk Assessment Based on Symptoms")

    # Input Fields for General User (Symptom-based)
    has_lump = st.radio('Do you feel a lump in your breast?', ('Yes', 'No'))
    pain = st.radio('Do you feel any pain in your breast or underarm?', ('Yes', 'No'))
    skin_changes = st.radio('Have you noticed any changes in the skin of your breast (e.g., redness, dimpling)?', ('Yes', 'No'))
    nipple_discharge = st.radio('Is there any discharge from your nipple?', ('Yes', 'No'))

    # Add a Scan button
    if st.button('Scan'):
        # Logic for General User based on their responses
        if has_lump == 'Yes' or pain == 'Yes' or skin_changes == 'Yes' or nipple_discharge == 'Yes':
            st.write("### Based on your symptoms, it is recommended to consult a healthcare professional.")
            st.write("A **mammogram** or other diagnostic tests may be necessary to determine the cause of your symptoms.")
            st.write("Early detection is key for effective treatment. Please visit your doctor for further evaluation.")
        else:
            st.write("### Your symptoms do not indicate an immediate concern.")
            st.write("However, it is still important to have regular check-ups and be mindful of any changes in your breast health.")
            st.write("If you notice new symptoms in the future, please consult your doctor.")

elif page == "Doctor's Page":
    # Doctor's Page - Diagnosis based on input features
    st.header("Breast Cancer Classification Using Gain Ratio Algorithm")

    # Input Fields for Doctor Data (medical feature-based)
    radius = st.number_input('Enter Mean Radius (in mm)', min_value=0.0, max_value=100.0, step=0.1, value=14.0)
    texture = st.number_input('Enter Mean Texture (value)', min_value=0.0, max_value=50.0, step=0.1, value=19.5)
    perimeter = st.number_input('Enter Mean Perimeter (in mm)', min_value=0.0, max_value=200.0, step=0.1, value=93.0)
    area = st.number_input('Enter Mean Area (in mm²)', min_value=0.0, max_value=2000.0, step=1.0, value=600.0)
    smoothness = st.number_input('Enter Mean Smoothness (value)', min_value=0.0, max_value=1.0, step=0.01, value=0.1)

    # Add a Scan button
    if st.button('Scan'):
        # Create input data array for prediction
        input_data = np.array([[radius, texture, perimeter, area, smoothness]])
        input_scaled = scaler.transform(input_data)  # Scale input the same way as the training data

        # Make prediction using the trained model
        prediction = dt.predict(input_scaled)

        # Construct the result text
        if prediction == 1:
            result_text = f"""**Diagnosis**: The tumor is **malignant**.
Immediate consultation for further testing and possible treatment, including biopsy, surgery, or chemotherapy, is recommended.

### Suggested Lifestyle and Treatment Plan:
- **Urgent Action Needed**: Immediate consultation with a doctor for biopsy, surgery, and possibly chemotherapy.
- **Diet Recommendations**: Anti-inflammatory foods like berries, leafy greens.
- **Physical Activity**: Avoid strenuous physical activity during chemotherapy or radiation.
"""
        else:
            result_text = f"""**Diagnosis**: The tumor is **benign**.
No immediate action needed. Regular check-ups and monitoring are recommended.

### Suggested Lifestyle and Treatment Plan:
- **Monitor and Observe**: Regular check-ups and mammograms to track any changes.
- **Diet Recommendations**: Maintain a balanced diet with healthy fats, lean proteins, and plenty of fruits and vegetables.
- **Physical Activity**: Stay active with low-impact activities like walking, swimming, or yoga.
"""

        result_text += f"\n### Model Accuracy (on test data):\nAccuracy: {accuracy * 100:.2f}%"
        result_text += "\nThis model helps assess the likelihood of breast cancer based on selected features.\nThe results provided by this tool are based on statistical modeling and are intended to assist you in decision-making. However, your clinical expertise and comprehensive evaluation of the patient are essential for making the final diagnosis and treatment plan."

        # Display the result as normal text (not in an input field)
        st.markdown(f"### Diagnosis and Treatment Plan:\n\n{result_text}")

        # Button to print the page (using browser's print functionality)
        st.markdown("""
            <a href="javascript:window.print()" target="_self">
                <button>Click to Print</button>
            </a>
        """, unsafe_allow_html=True)
