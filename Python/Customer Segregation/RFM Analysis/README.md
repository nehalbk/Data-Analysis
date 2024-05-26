# RFM Analysis Project

## Overview

This project performs an RFM (Recency, Frequency, Monetary) analysis on customer transaction data. The goal is to segment customers into different groups based on their purchasing behavior and to derive actionable insights for targeted marketing strategies.

## Project Structure

This project is part of the `Data-Analysis` repository, located under `Python/Customer Segregation/RFM Analysis`.

- **Data Preparation**: Cleaning and preprocessing the data.
- **RFM Calculation**: Computing Recency, Frequency, and Monetary values for each customer.
- **Outlier Removal**: Removing outliers to ensure robust analysis.
- **Normalization**: Normalizing the RFM values to bring them to a common scale.
- **Clustering**: Applying K-means clustering to segment customers into four groups.
- **Cluster Analysis**: Analyzing and interpreting the clusters.
- **Visualization**: Visualizing the clusters using 3D plots.

## Dataset, Notebooks and Scripts

- **Dataset** : Actual transactions from UK retailer. [Link to the dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **RFM_Analysis.ipynb**: Jupyter notebook containing the full analysis.
- **RFM_Analysis.py**: Python script converted from the Jupyter notebook.

## Data Preparation

1. **Loading Data**: Importing customer transaction data.
2. **Data Cleaning**: Handling missing values and correcting data types.
3. **Outlier Removal**:
    - Impact on Clustering: Outliers can skew clustering results, leading to less meaningful clusters.
    - Statistical Measures: Outliers affect the mean and standard deviation, making data less representative.
    - Normalization and Scaling: Outliers stretch the scale, reducing the granularity of normalized values.
    - Interpretability: Removing outliers makes results more interpretable and actionable for business decisions.
4. **Normalization**:
    - Equal Weighting: Ensures recency has the same scale as frequency and monetary value, giving each component equal importance.
    - Improved Model Performance: Helps clustering algorithms work more effectively.
    - Consistency: Brings all values into a common scale.
    - Stability: Reduces the impact of outliers and extreme values.

## RFM Calculation

- **Recency**: Calculated as the number of days since the last purchase.
- **Frequency**: Calculated as the total number of purchases.
- **Monetary**: Calculated as the total spend.

## Clustering

- **K-means Clustering**: Applied to the normalized RFM data to segment customers into four clusters:
    - **True Friends**: High Recency, High Frequency, High Monetary
    - **Butterflies**: High Recency, Low Frequency, High Monetary
    - **Barnacles**: High Recency, High Frequency, Low Monetary
    - **Strangers**: Low Recency, Low Frequency, Low Monetary

## Cluster Analysis

- **True Friends**: Implement loyalty programs, provide exclusive offers, and maintain personalized communication.
- **Butterflies**: Offer special promotions, use remarketing strategies, create urgency with limited-time offers, and suggest complementary products.
- **Barnacles**: Encourage higher spending through upselling, offer product bundles, educate on higher-value products, and provide spending incentives.
- **Strangers**: Increase brand awareness, engage with compelling content, offer attractive promotions, and analyze further to understand their needs.

## Visualization

- **3D PCA Plot**: Visualizing the clusters in a 3D space using Principal Component Analysis (PCA).

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/nehalbk/Data-Analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Data-Analysis/Python/Customer Segregation/RFM Analysis
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook or the Python script:
   ```bash
   jupyter notebook RFM_Analysis.ipynb
   ```
   or
   ```bash
   python RFM_Analysis.py
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Thanks to the open-source community for providing the tools and libraries used in this project.
- Inspired by various RFM analysis tutorials and customer segmentation strategies.
```