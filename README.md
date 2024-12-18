# VeriFi-FakeNewsDetection

## Overview
The exponential growth of online news platforms and social media has made the dissemination of information faster than ever before. While this has enabled users to access news instantly, it has also created fertile ground for the spread of misinformation, or "fake news." Fake news has significant societal impacts, influencing public opinion, political decisions, and even causing unrest. Detecting fake news is thus a critical challenge in today's digital era.

This project focuses on leveraging deep learning techniques, particularly a **Bidirectional Long Short-Term Memory (Bi-LSTM)** model, to detect fake news. Bi-LSTM, a variant of Recurrent Neural Networks (RNNs), is well-suited for text data analysis due to its ability to capture long-term dependencies and context in sequential data.

By preprocessing textual data, generating embeddings, and training an efficient classification model, this project aims to achieve high accuracy in distinguishing between fake and real news articles.

---

## Dependencies
The project uses the following libraries and frameworks:

- **TensorFlow/Keras**: For building and training the LSTM neural network.
- **NLTK**: For natural language processing tasks such as tokenization, stemming, and removing stopwords.
- **Pandas & NumPy**: For data manipulation and handling numerical computations.
- **Matplotlib & Seaborn**: For data visualization and result analysis.

---

## Dataset

### Source
- Publicly available datasets, such as those on **Kaggle**, containing labeled news articles (e.g., `true.csv` and `fake.csv`).

### Structure
- The dataset consists of textual data with attributes such as:
  - **Article title**
  - **Article body**
  - **Label**: "true" or "fake"

### Size
- Approximately **45,000 articles**, ensuring sufficient data for training, validation, and testing.

---

## Methodology
1. **Data Preprocessing**
   - Tokenization, stemming, and stopword removal using **NLTK**.
   - Preparing clean text data for embedding.

2. **Text Embedding**
   - Generating word embeddings using pre-trained models or custom embeddings.

3. **Model Training**
   - Implementing a **Bi-LSTM** neural network for classification.
   - Training the model on labeled news articles.

4. **Evaluation**
   - Analyzing the model's performance using metrics such as accuracy, precision, recall, and F1-score.

5. **Visualization**
   - Using **Matplotlib** and **Seaborn** to visualize results and insights.

---

## Objective
- Build an accurate and robust fake news detection system using deep learning.
- Understand and apply natural language processing techniques to preprocess news data.
- Leverage Bi-LSTM to efficiently classify news articles as "fake" or "real."

---

## How to Run
1. Install the required dependencies:
   ```bash
   pip install tensorflow keras nltk pandas numpy matplotlib seaborn
   ```
2. Download the dataset and place the `true.csv` and `fake.csv` files in the project directory.
3. Run the Jupyter Notebook or Python script to preprocess the data, train the model, and evaluate the results.

---

## Results
- High classification accuracy on distinguishing between fake and real news articles.
- Visualizations to highlight the dataset distribution and model performance.

---

## Future Work
- Incorporate additional datasets for better generalization.
- Explore transformer-based models like **BERT** for improved results.
- Deploy the model as a web application for real-time fake news detection.

