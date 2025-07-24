# Spam Classification Project

A machine learning project that implements email spam classification using Logistic Regression. This project demonstrates both the use of scikit-learn's built-in LogisticRegression and a custom implementation from scratch.

## 📊 Project Overview

This project classifies emails as either "Spam" or "Not Spam" using machine learning techniques. The model achieves an impressive *97.2% accuracy* on the test dataset, making it highly effective for real-world spam detection applications.

## 🎯 Key Features

- *High Accuracy*: 97.2% accuracy with excellent precision and recall scores
- *Dual Implementation*: Both scikit-learn and custom from-scratch Logistic Regression
- *Real-time Prediction*: Function to classify new email content
- *Comprehensive Evaluation*: Detailed classification reports and metrics
- *Feature Engineering*: TF-IDF vectorization for text processing

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| *Accuracy* | 97.2% |
| *Precision (Spam)* | 94% |
| *Recall (Spam)* | 96% |
| *F1-Score (Spam)* | 95% |
| *Precision (Not Spam)* | 98% |
| *Recall (Not Spam)* | 98% |
| *F1-Score (Not Spam)* | 98% |

## 🛠 Technical Implementation

### Data Processing
- Dataset: emails.csv containing email features and labels
- Feature extraction using word frequency analysis
- Train-test split (80-20) with random state for reproducibility

### Model Architecture

#### 1. Scikit-learn Implementation
python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)


#### 2. Custom Implementation
The project includes a custom LogisticRegressionScratch class that implements:
- Sigmoid activation function with overflow protection
- Gradient descent optimization
- Bias term integration
- Customizable learning rate and iterations

### Feature Engineering
- TF-IDF vectorization for text feature extraction
- Word frequency counting using Counter
- Regular expression-based word extraction

## 🚀 Usage

### Prerequisites
bash
pip install pandas numpy scikit-learn


### Running the Project
1. Ensure you have the emails.csv dataset in your project directory
2. Open spam_classification.ipynb in Jupyter Notebook
3. Run all cells sequentially

### Making Predictions
python
# Example usage
email_text = "Your email content here"
result = predict_email_spam(email_text)
print(result)  # Output: 'Spam' or 'Not Spam'


## 📝 Example Predictions

### Legitimate Email
python
email = "Dear MHR Residents, This is to inform you that a new mess menu has been prepared for the upcoming week..."
result = predict_email_spam(email)
# Output: 'Not Spam'


### Spam Email
python
email = "I'm Margaret Wilson, a UX-UI Graphic Designer. I help businesses like yours improve their website design..."
result = predict_email_spam(email)
# Output: 'Spam'


## 🔧 Custom Implementation Details

The custom LogisticRegressionScratch class includes:

- *Sigmoid Function*: 1 / (1 + exp(-z)) with overflow protection
- *Gradient Descent*: Iterative weight updates using the gradient
- *Bias Term*: Automatic addition of intercept term
- *Weight Initialization*: Random normal distribution initialization

## 📊 Dataset Information

- *Format*: CSV file (emails.csv)
- *Features*: Word frequency columns (all columns except first and last)
- *Target*: Binary classification (0 = Not Spam, 1 = Spam)
- *Split*: 80% training, 20% testing

## 🎯 Model Performance Analysis

The model shows excellent performance across all metrics:
- *High Precision*: Low false positive rate
- *High Recall*: Low false negative rate
- *Balanced Performance*: Similar performance for both classes
- *Robust Generalization*: Good performance on unseen data

## 🔮 Future Enhancements

Potential improvements for the project:
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add feature importance analysis
- [ ] Implement other algorithms (SVM, Random Forest, Neural Networks)
- [ ] Create a web interface for real-time classification
- [ ] Add email preprocessing (removing HTML, handling attachments)
- [ ] Implement ensemble methods for improved accuracy

## 📚 Dependencies

- *pandas*: Data manipulation and analysis
- *numpy*: Numerical computations
- *scikit-learn*: Machine learning algorithms and utilities
- *re*: Regular expressions for text processing
- *collections.Counter*: Word frequency counting

## 🤝 Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

This project was developed as part of a machine learning study focusing on spam classification using logistic regression.

---

*Note*: This project demonstrates both practical machine learning implementation and educational value through the custom algorithm implementation.
