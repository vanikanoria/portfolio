![Profile Picture](/docs/assets/p1.jpg)
# Vani Kanoria: Data Scientist

I'm a Data Scientist passionate about explainable machine learning and product strategy. I grew up being fascinating by Mathematics, which drew me to Data Analysis projects at Colgate University. From there, I transitioned into the tech world, analyzing data to shape business and product strategy at a fast-paced startup. Now, I’m further honing my expertise in machine learning as a Masters student at the University of Pennsylvania.

In addition to technical skills, I bring a unique perspective on high-growth tech businesses as an **<a href="https://eent.seas.upenn.edu/fellows/2024-cohort/#kanoria" target="_blank">Engineering Entrepreneurship Fellow</a>**, where I study the intersection of innovation, technology, and business strategy.




# Projects

### Machine Learning
- [Multi-Modal Models for Breast Cancer Classification](#multimodal-breast-cancer-classification)
- [Churn Prediction Model for  Sisense Customers](#churn-prediction-model-for-sisense)
- [Adapting Explainability Methods to Image Generation Models](#adapting-explainability-methods-to-image-generation-models)  
- [Use Case Clustering for No-Code Platform <a href="https://www.unqork.com/" target="_blank">Unqork</a>](#workflow-use-case-clustering)
- [Explaining Medical Insurance Price Prediction Models](#explaining-medical-insurance-price-prediction-models)  

### Probability Modeling
- [Modeling Credit Card Usage using Negative Binomial Distributions](#modeling-credit-card-usage-using-negative-binomial-distributions)  
- [Generative Probability Models Capturing Trends in Ticket Sales of the Movie ‘Wonka’](#generative-probability-models-for-wonka-ticket-sales)  

### Data Analysis  
- [Decomposing Recent Protests in the US](#decomposing-recent-protests-in-the-us)  

### Business Strategy in Tech  
- [End-to-End Business Plan for Biodegradable Soil Sensors](#business-plan-for-biodegradable-soil-sensors)  
- [Consulting Fortune 500 Insurance Company on Integrating AR/VR](#consulting-insurance-company-on-arvr-integration)  

<a id="multimodal-breast-cancer-classification"></a> 
## Multi-Modal Models for Breast Cancer Classification
	

**Objective:** 
- Develop a multimodal deep learning framework to enhance the accuracy of breast cancer diagnosis by integrating mammogram imaging data with clinical features.

**Methods:** 
- Utilized the CBIS-DDSM dataset, combining mammogram images with tabular clinical data for multimodal analysis.
- Explored multiple architectures:
	•	ResNet for image-only classification.
	•	Multimodal combinations of CLIP (image and text), BERT (text), ResNet (image), and Vision Transformers, or ViT (image).

**Results:**  
- Achieved 86.36% test accuracy with the BERT-ViT configuration, demonstrating a 25% improvement over the ResNet baseline.
- Demonstrated the potential of multimodal learning for breast cancer detection, combining complementary imaging and clinical features for better diagnostic support.
<img width="1073" alt="Architecture" src="https://github.com/user-attachments/assets/f7b1ecdc-02ed-4694-bfb2-c3070e3a4e6b" />

<a href="https://github.com/vanikanoria/MultiModalBreastCancerClassification" target="_blank">Github link</a>

<a id="churn-prediction-model-for-sisense"></a>  
## Churn Prediction Model for **<a href="https://www.sisense.com/" target="_blank">Sisense</a>**

In this internship project, I built two machine learning models to predict churn using features I developed from support and engineering tickets, customer call transcripts, product usage data and onboarding data. I achieved 94% test accuracy overall by combining the models.

**Objective:**  
- Develop machine learning models to predict customer churn for Sisense's client base.

**Methods:**  
- Collected and processed data from support/engineering tickets, Gong call transcripts, product usage metrics, and onboarding data.  
- Segmented customers by size and use case
- Engineered predictive features and built Random Forests model for each segment
- Add feature attribution using Shapley values, explaining to customer success which features are signalling high churn risk
- Designed user interface to optimize actionability

**Results:**  
- Achieved **94% test accuracy** by combining models.  
- Identified key churn drivers, enabling targeted customer success interventions.  

<a id="adapting-explainability-methods-to-image-generation-models"></a>
## Adapting Explainability Methods to Image Generation Models

**Objective:**  
- Identify key tokens in text-to-image generation models, focusing on Stable Diffusion.  

**Methods:**  
- Adapted Shapley values (SHAP) to identify key tokens in text prompts.  
- Applied the framework to understand how input tokens influence generated images.  

**Results:**  
- Achieved **90% accuracy** in identifying key tokens.  
- Enhanced model transparency and interpretability for end-users.  
    
![Trustworthy ML Project](/docs/assets/shap.jpg)

<a href="https://github.com/CalebG1/TrustworthyML/blob/main/Automating_SHAP_value_calculation.ipynb" target="_blank">Github link</a>

<a id="workflow-use-case-clustering"></a>  
## Use Case Clustering for **<a href="https://www.unqork.com/" target="_blank">Unqork</a>**'s No-Code Workflow Development Feature

As a Product Data Analyst at **<a href="https://www.unqork.com/" target="_blank">Unqork</a>**, I quantified characteristics of applications created using Unqork’s software and applied clustering algorithms to categorize use cases, leading to a pivot in product strategy and saving 70% of engineering resources allocated.

**Objective:**  
- Analyze and categorize applications built using the workflow feature **<a href="https://www.unqork.com/" target="_blank">Unqork</a>**'s no-code platform to inform future product roadmap.

**Methods:**  
- Extracted key characteristics from applications.  
- Applied clustering algorithms to identify distinct use case categories.  

**Results:**  
- Optimized product strategy by focusing on key use cases.  
- Saved **70% of engineering resources** by prioritizing high-impact clusters.

<a id="explaining-medical-insurance-price-prediction-models"></a>  
## Explaining Medical Insurance Price Prediction Models

In this project I apply 4 different machine learning models (Linear Regression, Random Forests, Decision Tree Regressor and XGBoost Regressor) to predict charges on medical insurance based on demographic characteristics of the holds of insurance. I then utilize feature importance functions in sklearn and SHAP analysis to attribute feature importance of each model.

**Objective:**   
- Predict medical insurance charges based on demographic data and identify most optimal model in terms of the accuracy-interpretability tradeoff.  

**Methods:**  
- Applied four models: Linear Regression, Random Forest, Decision Tree Regressor, and XGBoost.  
- Used feature importance (scikit-learn) and SHAP analysis to interpret model outputs.  

**Results:**  
- Identified key factors influencing insurance costs (e.g., age, BMI, smoking status).  
- Improved model transparency, aiding in fair and explainable predictions.  

![Insurance ML Project](/docs/assets/insurance.jpg)

<a href="https://github.com/vanikanoria/Explaining-Medical-Insurance-Price-Prediction-Models/blob/main/Medical%20Insurance%20Price%20Prediction.ipynb" target="_blank">Github link</a>
  
 

<a id="modeling-credit-card-usage-using-negative-binomial-distributions"></a>  
## Modeling Credit Card Usage using Negative Binomial Distributions

How many credit cards do people in the United States own and why? This breakdown of the number of credit cards regularly used by different age groups sheds light on the preferred number of cards for different demographic groups, and may even provide hints about financial habits exhibited by them. 

**Objective**  
- Analyzed patterns in credit card ownership to inform credit card strategy.  

**Methods:**  
- Applied Negative Binomial Distribution models to survey data.  
- Segmented data by age and demographic characteristics.  

**Results:**  
- Identified trends in credit card usage across demographics.  
- Provided insights into financial habits and credit preferences.

  ![Credit Card Usage Project](/docs/assets/creditCard.jpg)

<a href= "https://docs.google.com/document/d/19pGHoWhPxMCa6Oa3IsEaXWJZjmh3kMTXBLWdWFak0HE/edit?usp=sharing" target="_blank">Link to paper</a>


<a id="generative-probability-models-for-wonka-ticket-sales"></a>  
## Capturing Behavioral Trends in Ticket Sales of the Movie ‘Wonka’

This paper examines the ticket sales of the movie Wonka in the three months after its release by applying time-series probability distributions and including time-varying covariates such as weekend and holiday effects. The 2-segment Weibull Distribution calibrated to the data captures the presence of two behavioral segments in the population with differing propensities to watch Wonka.

**Objective**  
- Modeled ticket sale trends for the movie "Wonka" using probabilistic methods to learn about behavioral trends and factors influencing moviegoers.  

**Methods:**  
- Applied time-series probability distributions (e.g., Weibull Distribution).  
- Incorporated time-varying covariates (weekends, holidays) to capture sales patterns.  

**Results:**  
- Accurately captured dual behavioral segments in ticket sales.  
- Provided insights into movie-going trends and seasonal effects.
  
![Wonka Project](/docs/assets/wonka.jpg)

<a href= "https://docs.google.com/document/d/1-PKIqhM4UE7eD6I4BJn0yteaYdaPIJSlaMNlUWYxIjc/edit?usp=sharing" target="_blank">Link to paper</a>

<a id="decomposing-recent-protests-in-the-us"></a>  
## Decomposing Recent Protests in the US

In this analysis, I decompose 42533 unique protests or riots occurring across the United States from the beginning of 2020 to mid 2022 by location, nature, time frame, and associated actors, which are the organizations involved. 

**Objective**  
- Analyze and categorize 42,533 protests across the U.S. (2020–2022) to indentify key actors and trends driving the protests.

**Methods:**  
- Decomposed data by location, nature, timeframe, and associated organizations.  
- Conducted exploratory data analysis (EDA) to identify patterns and trends.  

**Results:**  
- Revealed geographic and temporal trends in protest activity.  
- Identified key actors and causes, informing sociopolitical analyses.  


  ![Protests in the US: Exploratory Data Analysis](/docs/assets/protests.jpg)

<a href="https://github.com/vanikanoria/ProtestsInTheUS/blob/main/Exploratory%20Data%20Analysis.ipynb" target="_blank">Github link</a>

<a id="business-plan-for-biodegradable-soil-sensors"></a>  
## End-to-end business plan for biodegradable soil sensors

As an Engineering Entrepreneurship Fellow, I worked with teammates Ashley Cousin, Caleb Gupta, and David Bakalov to develop a product and market strategy, and financial and operations plan for biodegradable soil sensors developed by Penn researchers.

**Objective**  
- Develope a comprehensive business plan to commercialize biodegradable soil sensors  

**Methods:**  
- Conducted market research and competitor analysis.  
- Created a product strategy, financial model, and operations plan.  

**Results:**  
- Presented a viable go-to-market strategy.  
- Highlighted the potential impact on sustainable agriculture.
 ![Soil Sensors Business Plan](docs/assets/SLIDES12_AgriVue.jpg)

<a href="https://docs.google.com/presentation/d/1LCqmTNJ7ml7fCio8PndM6OE5DTvxfYXaJ7hTeSNl6aE/edit?usp=drive_link" target="_blank" rel="noopener noreferrer">Link to slide deck</a>

<a id="consulting-insurance-company-on-arvr-integration"></a>  
## Consulting Fortune 500 Insurance Company on Integrating AR/VR  

I collaborated with Wharton MBAs Tanmay Sharma, Zachary Gross and Ziyi Zhou Collaborating advise Assurant Solutions, a Fortune 500 insurance provider, on leveraging Augmented and Virtual Reality technologies in their smart devices insurance division across their value chain. We presented a competitor analysis and Go-to-Market plan to implement the technology into Assurant’s existing business model

**Objective**   
- Advise Assurant Solutions on leveraging AR/VR technology to reduce costs in their value chain.  

**Methods:**  
- Conducted competitor analysis and identified use cases for smart device insurance.  
- Developed a go-to-market strategy for AR/VR integration.  

**Results:**  
- Proposed an implementation plan to improve efficiency.  
- Provided actionable insights to enhance product offerings. 

 ![AR/VR Consulting Project](docs/assets/ARVR.jpg)
 
 <a href="https://docs.google.com/presentation/d/1SJXlQ_9DXfKgh1gjMd89HgMOJMibB6_I/edit?usp=drive_link&ouid=113368363770653803719&rtpof=true&sd=true" target="_blank">Link to slide deck</a>


# Education

* Masters of Science in Engineering: **Data Science** at the **University of Pennsylvania**

* Bachelors in **Applied Mathematics** and **Economics** at **Colgate University**

# Work Experience

* Machine Learning Engineer (Part-Time) @ **<a href="https://www.sisense.com/" target="_blank">Sisense</a>**

* Machine Learning Engineer Intern @ **<a href="https://www.sisense.com/" target="_blank">Sisense</a>**

* Data Analyst @ **<a href="https://www.unqork.com/" target="_blank">Unqork</a>**

* Analytics Intern @ **<a href="https://www.unqork.com/" target="_blank">Unqork</a>**
