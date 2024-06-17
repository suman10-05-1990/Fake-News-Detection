# Fake-News-Detector

**OUTCOME
Conclusion:**
**This report has presented a comprehensive approach to developing a fake news detection system using natural language processing (NLP) techniques and machine learning. The project was structured into several key phases:**

1.	Data Preparation: We imported and merged datasets containing both fake and true news articles, ensuring each article was appropriately labeled for classification purposes.

2.	Text Preprocessing: Extensive preprocessing techniques were applied to clean and normalize the text data. This included removing HTML tags, non-alphabetic characters, stopwords, and lemmatizing tokens to their root forms. The cleaned text was then transformed into numerical features using TF-IDF vectorization.

3.	Model Development: A logistic regression model was chosen for its interpretability and effectiveness in binary classification tasks. The model was trained on the TF-IDF transformed data to distinguish between fake and true news articles.

4.	Model Evaluation: The performance of the trained model was evaluated using various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Visualizations like confusion matrices, ROC curves, and feature importance graphs provided insights into the model's effectiveness and areas for improvement.

5.	Visualization and Insights: To enhance interpretability, a wide range of visualizations were utilized. These included bar graphs for class distribution, top words analysis, heatmaps for correlation matrices, network diagrams for word relationships, and scatterplots for article length analysis. These visualizations not only helped in understanding the dataset but also in interpreting the model's predictions.

