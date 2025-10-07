# West Midlands House Prices Analysis

## Overview
This project analyses residential property prices across the **West Midlands** region to uncover the key drivers of value and predict future prices using explainable machine-learning models.  
It combines open UK data sources — property transactions, energy efficiency, and socio-economic indicators — into a single analytical dataset and presents insights through an **interactive Tableau dashboard**.

This work forms the **Capstone Project** for the *Code Institute Data Analytics with Artificial Intelligence Bootcamp*.

---

## Business Requirements

### Problem Statement
Property valuation is complex and buyers, investors and estate agencies often lack transparency about how different factors affect price.  

This project aims to address the question:

*What are the most significant factors influencing house prices in the West Midlands, and can we build a transparent model to predict them?*

The clients are buyers and estate agencies. 

Business Requirement 1 - The clients need to understand the drivers of property prices in order to make more informed purchase decisions.

Business Requirement 2 - The clients is interested in using existing housing sold data and related property and location information to predict the selling price for a property.


### Business Goals & Success Criteria
This project aims to: 
1. **Identify** which features (property type, floor area, deprivation level, tenure, EPC band etc.) best explain price variation.  
2. **Develop** a machine-learning model to predict `price`.  
3. **Visualise** regional and feature-level patterns via a Tableau dashboard.  
4. **Communicate** actionable insights to the clients.

---

## Hypotheses

Hypothesis 1: The property type has an effect on the price per square metre: Detached houses have a higher price per square metre than semi-detached, which are higher than terraced, which are higher than flats. This will be validated using visualizations and statistical (ANOVA and Tukey) tests.

Hypothesis 2: New builds sell at a premium compared to older properties. This will be validated using visualizations and statistical tests t-test).

Hypothesis 3: A higher Index of Multiple Deprivation (IMD) decile is associated with a higher price per square metre. The expected relationship is a positive correlation. This will be validated using visualizations, statistical tests (Spearman’s rho) or regression analysis.

Hypothesis 4: Properties with EPC bands A to C have a price premium compared to bands D to G. The expected relationship is a positive effect. This will be validated using ANOVA or regression.

Hypothesis 5: Leasehold properties sell at a discount compared to freehold properties. The expected relationship is a negative effect. This will be validated using a t-test and regression analysis.

In summary:

| ID | Hypothesis | Expected Relationship | Validation |
|----|-------------|----------------------|-------------|
| H1 | Detached > Semi > Terraced > Flat in price per m² | Positive hierarchy | ANOVA + Tukey |
| H2 | New builds sell at a premium | Positive effect | t-test |
| H3 | Higher IMD decile → higher price per m² | Positive correlation | Spearman ρ / Regression |
| H4 | EPC A–C bands have premium vs D–G | Positive effect | ANOVA / Regression |
| H5 | Leasehold discount vs freehold | Negative effect | t-test + Regression |

---
