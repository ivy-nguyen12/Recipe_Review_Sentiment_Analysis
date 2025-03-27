# Recipe Ratings Analysis

## Overview
This project explores which recipe attributes and user sentiments influence high ratings on a recipe-sharing platform. Using machine learning models on two Kaggle datasets, we identify key drivers of recipe success based on recipe metadata and review content. Our findings offer actionable insights for food bloggers, recipe platforms, and meal kit services to optimize recipe development, presentation, and engagement strategies.

Prepared by: Vy Nguyen, Kento Morita, Qui Nguyen, Aria Zhou, Sahil Chennadi, Yuheshwar Kamakkapalayam Subramani

Key business insights include:
- Taste, ease of preparation, and user experience drive high ratings more than nutritional complexity
- Highlighting simplicity (7–9 ingredients, 5–8 steps) and using positive language improves perceived quality
- Encouraging detailed reviews and nutritional moderation can enhance trust and engagement

## Dataset
- **Source:** Kaggle ([Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions))  
- **Scope:**  
  - `RAW_recipes.csv`: 231,637 recipes with metadata and ingredients  
  - `RAW_interactions.csv`: 1,158,039 user reviews and ratings  
- **Target Variables:**  
  - Recipe Attribute Modeling: Recipe Rating (0–5 stars)  
  - Review Sentiment Modeling: Binary Recommendation (High Rating ≥4 vs Low Rating <4)

### 📋 Data Dictionary

| Feature                    | Definition                                                                 | Data Type  |
|----------------------------|----------------------------------------------------------------------------|------------|
| minutes                    | Preparation time for the recipe                                            | Numerical  |
| n_steps                    | Number of steps required to make the recipe                                | Numerical  |
| n_ingredients              | Total number of ingredients used                                           | Numerical  |
| n_tags                     | Number of associated food.com tags (e.g., “low-carb”, “quick”)             | Numerical  |
| nutrition_sum              | Total sum of nutrition values (calories, fat, sugar, protein, etc.)        | Numerical  |
| total_year_from_submission| Number of years since the recipe was published (from 2019)                 | Numerical  |
| review_length              | Word count of the user’s review                                            | Numerical  |
| sentiment_score            | Polarity score of review text (positive/negative sentiment)                | Numerical  |

## Tools & Methodology Overview
**Languages and Libraries:** Python (pandas, numpy, sklearn, matplotlib, seaborn, nltk, imblearn)

### 1. Recipe Attribute Modeling:
- Joined recipe data with average user rating
- Engineered features: tag count, nutrition sum, years since published
- Target: Recipe Rating (0–5 stars)
- Models: Linear Regression, GLM, Neural Network, Random Forest
- Evaluation: 10-fold Cross-Validation (MSE)

### 2. Review Sentiment Modeling:
- Preprocessed reviews (tokenization, lemmatization, TF-IDF)
- Features: review length, sentiment score
- Target: Binary classification (High ≥4 vs Low <4)
- Model: Random Forest with SMOTE
- Evaluation: F1-score

### Evaluation Metrics:
- For Regression: Mean Squared Error (MSE)
- For Classification: Accuracy, Precision, Recall, F1-Score

## Highlighted Visualizations

**Word Cloud of Recipe Attributes**  
![Attribute Word Cloud](Notebooks/wordcloud_recipe_tags.png)

**Word Cloud of Review Texts**  
![Review Word Cloud](Notebooks/wordcloud_review_texts.png)

**Feature Importance (Random Forest)**  
![Feature Importance](Notebooks/feature_importance.png)

## Results & Key Insights

### Model Performance Comparison

| Model                     | Mean Squared Error (MSE) |
|--------------------------|---------------------------|
| Linear Regression         | 127.941                   |
| Generalized Linear Model  | 0.980                     |
| Neural Network            | 0.977                     |
| Random Forest         | 0.145                 |

- **Random Forest** significantly outperformed other models in both tasks
- Most predictive attributes: `nutrition_sum`, `n_tags`, and `n_steps`
- **Sentiment polarity** and **review length** were strong predictors of user satisfaction
- Highly rated reviews frequently include terms like *"delicious"*, *"easy"*, and *"flavorful"*

### Business Recommendations

- **Focus on Taste & Ease**: Use simple language and design recipes with 7–9 ingredients and 5–8 steps
- **Leverage Nutrition**: Emphasize moderate calorie/fat content while avoiding overuse of “low sodium/sugar”
- **Encourage Detailed Reviews**: Prompt users to leave thoughtful feedback and address common complaints
- **Optimize Recipe Tags**: Use effective, searchable tags like “30-minute meal” or “gluten-free”

## Key Deliverables
- Jupyter Notebooks:  
  - `Notebooks/Recipe_Attribute_Analysis.ipynb`  
  - `Notebooks/Recipe_Review_Analysis.ipynb`  
- Final Report (PDF)  
- Presentation Slides (PDF)

## What I Learned
- Combining attribute and sentiment modeling enhances prediction and interpretation
- Text data offers valuable signals when processed with NLP and TF-IDF
- SMOTE improves performance with imbalanced data
- Feature engineering plays a crucial role in modeling performance

## What I Plan to Improve
- Apply XGBoost and LightGBM for boosting comparisons
- Explore advanced NLP (topic modeling, NER) for review insights
- Extend to user-based recommendation systems based on preference clusters

## About Me
Hi, I’m Vy Nguyen and I’m currently pursuing my MS in Business Analytics at UC Irvine. I’m passionate about data analytics in Finance, Investment, and Consumer Behavior. Connect with me on [LinkedIn](https://www.linkedin.com/in/vy-ngoc-lan-nguyen).
