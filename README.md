# **CancerGuard - Breast Cancer Diagnosis and Treatment Suggestions**

## **Inspiration**

The inspiration for **CancerGuard** came from the growing need for early detection and diagnosis in the fight against breast cancer, one of the leading causes of death worldwide. Many people, especially those in rural or underserved areas, face challenges in accessing timely diagnostic tools like mammograms and biopsies. We sought to create a simple, accessible tool that could help both doctors and patients assess the risk of breast cancer based on easily accessible medical data and symptoms. Our aim was to empower healthcare providers with AI-driven tools to support decision-making while also providing patients with valuable information for proactive health management.

---

## **What it does**

**CancerGuard** is an AI-powered tool designed to help with breast cancer diagnosis by analyzing medical data and symptoms. The tool provides two primary functionalities:

1. **Symptom-based Self-Assessment**: For general users, it offers a simple survey to assess the likelihood of breast cancer based on common symptoms (e.g., lump, pain, skin changes, or nipple discharge). This can help users make informed decisions about whether they should consult a healthcare professional.
   
2. **Breast Cancer Classification**: For healthcare professionals, **CancerGuard** takes in key medical data (such as mean radius, texture, perimeter, area, and smoothness of the tumor) and uses a decision tree classifier model to predict whether a tumor is benign or malignant. It then offers recommendations for further steps and treatment options based on the results.

The goal of **CancerGuard** is to help doctors with initial screening and diagnosis, as well as provide patients with basic guidance about their health.

---

## **How we built it**

The tool was built using the following technologies:

- **Streamlit**: We used **Streamlit** to create a simple and interactive web application that allows users to input data and receive instant predictions. It provides a clean, user-friendly interface for both doctors and general users.
  
- **Python Libraries**: 
  - **scikit-learn** for implementing machine learning algorithms, specifically the **Decision Tree Classifier** for predicting whether the tumor is benign or malignant.
  - **Numpy** for handling and manipulating the numerical data.
  - **Pandas** for processing and cleaning any input or output data.
  
- **Data**: We used the **Breast Cancer Wisconsin (Diagnostic)** dataset from **scikit-learn**, which contains key features about the tumor's size, shape, and texture. The model is trained using this dataset to predict tumor malignancy based on user-provided data.

- **StandardScaler**: To ensure the accuracy of the predictions, we used **StandardScaler** to normalize the input data before passing it into the model.

---

## **Challenges we ran into**

- **Data Preprocessing**: We had to clean and preprocess the data effectively, ensuring that the features were appropriately scaled. This was especially challenging because we had to make sure that user inputs matched the format of the dataset.
  
- **Model Accuracy**: While we used a decision tree classifier, ensuring that the model provided accurate predictions for a wide variety of input data was a challenge. We had to tweak and test different parameters to improve accuracy.

- **UI/UX Design**: Making sure that the interface was user-friendly for both doctors and general users was important, but challenging. We wanted to balance complexity and simplicity, ensuring the tool was intuitive while still powerful.

- **Integrating User Feedback**: For general users, predicting breast cancer based on symptoms alone required careful consideration of how to guide them in a responsible and non-alarming way.

---

## **Accomplishments that we're proud of**

- We built a fully functional AI-powered tool that can predict breast cancer risk based on both symptoms and medical features.
  
- The tool is accessible to both general users and healthcare professionals, which broadens its potential use case. Doctors can use it for preliminary assessments, and patients can use it to understand their symptoms and decide whether they need to see a doctor.
  
- We achieved over 90% accuracy on the breast cancer prediction model using the decision tree classifier, which is impressive for a simple machine learning model on such a critical task.
  
- We integrated user inputs and real-time predictions with a simple yet functional interface using **Streamlit**, which allows the tool to be used without requiring complex setup or technical knowledge.

---

## **What we learned**

- We gained experience in integrating AI and machine learning with real-world healthcare applications, focusing on ensuring both accuracy and user-friendliness.
  
- We learned how to properly scale and normalize data, and how to adjust machine learning models to ensure better predictions in sensitive areas like healthcare.

- The importance of balancing technological innovation with real-world applicability became clear. While AI can offer significant assistance, we had to ensure that the tool was not overstepping its bounds and that it acted as an aid to medical professionals rather than a replacement.
  
- Streamlining the user experience, especially for doctors who are busy and need fast results, was another key learning point.

---

## **What's next for CancerGuard**

Moving forward, **CancerGuard** will focus on expanding and refining the following areas:

1. **Model Improvement**: We plan to explore other machine learning models (e.g., Random Forest, Neural Networks) to improve accuracy and performance further.

2. **Expanded Dataset**: Incorporating more diverse and larger datasets from multiple medical sources will help to make the predictions even more reliable.

3. **Broader Use Cases**: We aim to extend the tool to other cancers, such as lung or prostate cancer, by integrating more features into the system.

4. **Real-time Data Integration**: We would like to explore the potential for integrating real-time patient data from devices like smartwatches, wearables, and other diagnostic tools to further personalize the risk assessments.

5. **Collaboration with Healthcare Providers**: Partnering with healthcare institutions to validate and deploy the tool in real-world settings will be a key focus to ensure that it can be used in clinical practice.

By improving the system and incorporating feedback from healthcare professionals, **CancerGuard** has the potential to become a vital tool in early cancer detection, making healthcare more efficient and accessible.
