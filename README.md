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

Hypothesis 2: New builds sell at a premium compared to older properties. This will be validated using visualizations and statistical tests (t-test).

Hypothesis 3: A higher Index of Multiple Deprivation (IMD) decile is associated with a higher price per square metre. The expected relationship is a positive correlation. This will be validated using visualizations, statistical tests (Spearman’s rho) or regression analysis.

Hypothesis 4: Properties with EPC bands A to C have a price premium compared to bands D to G. The expected relationship is a positive effect. This will be validated using visualizations, ANOVA or regression.

Hypothesis 5: Leasehold properties sell at a discount compared to freehold properties. The expected relationship is a negative effect. This will be validated using visualizations, a t-test and regression analysis.

In summary:

| ID | Hypothesis | Expected Relationship | Validation |
|----|-------------|----------------------|-------------|
| H1 | Detached > Semi > Terraced > Flat in price per m² | Positive hierarchy | Viz, ANOVA, Tukey |
| H2 | New builds sell at a premium | Positive effect | Viz, t-test |
| H3 | Higher IMD decile → higher price per m² | Positive correlation | Viz, Spearman rho, Regression |
| H4 | EPC A–C bands have premium vs D–G | Positive effect | Viz, ANOVA, Regression |
| H5 | Leasehold discount vs freehold | Negative effect | Viz, t-test, Regression |

---
## Dataset Content

The analysis combines **four open datasets** joined on postcode, address and LSOA codes.

### 1 UK Land Registry — Price Paid Data (PPD)
- Residential sales across England & Wales (2015–2025).  
- Key fields: `price`, `transfer_date`, `postcode`, `property_type`, `new_build`, `tenure`.  
- Purpose: target variable and core property attributes.  

Price Paid Data is released under the [Open Government Licence (OGL)](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
  
📎 [Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)

### 2️ ONS Postcode Directory (ONSPD)
- Maps postcodes to LSOA/MSOA/Local Authority.  
- Fields: `postcode`, `lsoa11cd`, `msoa11nm`, `ladnm`.  
- Purpose: spatial join between PPD, IMD and EPC.  
- PCD_OA_LSOA_MSOA_LAD_MAY22_UK_LU.CSV is a UK government geospatial dataset. It stands for Postcode Directory (PCD), Output Area (OA), Lower Layer Super Output Area (LSOA), Middle Layer Super Output Area (MSOA), Local Authority District (LAD), May 2022, UK, Lookup (LU). It's published by the Office for National Statistics (ONS) and used for mapping postcodes to statistical and administrative areas. 
  
📎 [License details](https://www.ons.gov.uk/methodology/geography/licences)
  
📎 [ONS Postcode Directory](https://geoportal.statistics.gov.uk/)

### 3️ Indices of Multiple Deprivation (IMD 2019)
- Official UK deprivation scores at LSOA level.  
- Fields: `lsoa11cd`, `imd_score`, `imd_rank`, `imd_decile`.  
- Purpose: adds socio-economic context.  
  
📎 [IMD 2019](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)

### 4️ Energy Performance Certificates (EPC)
- Energy ratings and floor areas for individual properties.  
- Fields: `postcode`, `property_type`, `built_form`, `total_floor_area`, `current_energy_rating`.  
- Purpose: floor area (for `price_per_sqm`) and energy efficiency feature.  
  
📎 [EPC Open Data](https://epc.opendatacommunities.org/)
