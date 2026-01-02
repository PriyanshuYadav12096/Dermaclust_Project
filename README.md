# 🧠 DermaClust: AI-Powered Skincare Recommender

An intelligent skincare product recommendation system that analyzes both user skin type via image classification and product ingredients via NLP to provide personalized suggestions.

---

## 📋 Table of Contents
- [About The Project](#about-the-project)
- [✨ Key Features](#-key-features)
- [📸 Screenshots](#-screenshots)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## About The Project

DermaClust is a dual-model recommendation engine designed to demystify the process of choosing the right skincare products. It leverages two distinct machine learning models:

1.  **CNN for Skin Analysis:** A Convolutional Neural Network (CNN) trained on thousands of skin images to classify the user's skin concern into one of five categories: `acne`, `dry`, `oily`, `wrinkles`, or `normal`.
2.  **Transformer for Ingredient Analysis:** A BERT-style Transformer model that processes a product's ingredient list to understand its suitability for different skin types, identifying key active ingredients and potential irritants.

The Streamlit-based web application provides a simple interface for users to upload a photo of their skin and receive tailored product recommendations from a comprehensive database.

---

## ✨ Key Features
- **AI-Powered Skin Classification:** Upload an image and get an instant analysis of your primary skin concern.
- **Intelligent Ingredient Matching:** The system understands which ingredients work best for each skin type.
- **Personalized Recommendations:** Get product suggestions that are scientifically matched to your skin's needs.
- **Interactive UI:** A clean and user-friendly interface built with Streamlit.

---

## 📸 Screenshots


<table>
  <tr>
    <td><img src="https://github.com/Priyanshu12yadav/Dermaclust-A-Skincare-Recommendation-System/blob/main/asset/Screenshot%202025-09-20%20000839.png" alt="Main Interface" width="300"></td>
    <td><img src="https://github.com/Priyanshu12yadav/Dermaclust-A-Skincare-Recommendation-System/blob/main/asset/Screenshot%202025-09-19%20233531.png" alt="Results Page" width="300"></td>
  </tr>
</table>


---

## 🛠️ Tech Stack
- **Backend & ML:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data Science:** Pandas, Scikit-learn, NumPy
- **Web Framework:** Streamlit
- **Computer Vision:** OpenCV

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Python 3.8+
- Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Generate and install dependencies:**
    *(First, run this command to create the requirements file)*
    ```sh
    pip freeze > requirements.txt
    ```
    *(Then, install from the newly created file)*
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```
The application should now be running in your web browser!

---

## 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
