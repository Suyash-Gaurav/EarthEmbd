# EarthEM: Embedding-Based Remote Sensing Analysis of Land Use and Climate Dynamics

This repository contains code and documentation for an investigation into the utility of satellite image embeddings for geospatial machine learning tasks. The analysis focuses on a case study of the Krishna Raja Sagara (KRS) Reservoir region in Karnataka, India, using the AlphaEarth 64-dimensional embedding dataset provided by Google Earth Engine.


Authors: Suyash Gaurav & Aman Kamat

Department of Digitial Business and Innovation & Department of Business Economics 


 
Work Distribution: 

Code, Analysis, Repo Creation,  (RQ-2 - RQ3) and its Hypothesis : Suyash

Dataset Authentication, Data Api calling, visualizaion, (RQ1) and its Hyptohesis: Aman



Email: s23224522@al.tiu.ac.jp

---

## Dataset Description

* **Source:** [Google Earth Engine: Satellite Embedding V1 (Annual)](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
* **Data Type:** 64-dimensional spectral-temporal embeddings for each 10-meter pixel
* **Temporal Coverage:** 2017–2024 (annual composites)
* **Spatial Coverage:** Global, subset to KRS and Mysore region in this study
* **Format:** Latent feature vectors A00–A63 derived from multi-modal satellite data (e.g., optical, radar)

This dataset enables remote sensing analysis without direct reliance on raw spectral bands, allowing for data-efficient applications in classification, change detection, and trend analysis.

---

## Research Objectives and Methodology

This notebook addresses three research questions through a hypothesis-driven empirical framework. Each question is approached using minimal supervision to evaluate the performance and interpretability of the satellite embeddings.

### RQ1: Can dominant crop types be reliably mapped with minimal labeled samples using the embedding space?

* **Hypothesis:** With sparse training data, the AlphaEarth embeddings encode sufficient land cover-specific information to allow a classifier to exceed 80% accuracy in distinguishing crops such as rice, sugarcane, and other vegetation.

* **Approach:**

  * Manually curated ~130 labeled points (rice, sugarcane, other crops, water, urban) from 2024 imagery
  * Trained a Random Forest Classifier on the 64-dimensional features
  * Used 70/30 stratified train-test split
  * Evaluated performance using accuracy, confusion matrix, PCA, and feature importance metrics

* **Findings:**

  * Achieved ~82.1% accuracy on held-out test set
  * Perfect classification for water and urban classes
  * Moderate confusion among crop types
  * Principal Component Analysis revealed separability in low-dimensional space
  * Embedding dimensions A09, A22, A28, etc., contributed significantly to class discrimination

### RQ2: Do interannual variations in embedding similarity reflect climatic extremes such as droughts?

* **Hypothesis:** The temporal embedding signature of the KRS Reservoir will deviate significantly in known drought years due to spectral and areal changes in the water body, yielding reduced similarity scores relative to normal years.

* **Approach:**

  * Extracted annual mean embeddings over the KRS water body for 2017–2024
  * Computed pairwise cosine similarity between year-to-year embeddings
  * Grouped years by climatic category (drought, normal, good monsoon)
  * Applied ANOVA to test statistical differences among climate groups
  * Visualized temporal similarity shifts using heatmaps and PCA trajectory plots

* **Findings:**

  * Cosine similarity among drought years was significantly lower (~0.89) than among normal (~0.97)
  * ANOVA yielded p ≈ 0.012, confirming climate-linked divergence
  * PCA showed that dry years diverged in embedding space, suggesting altered water body characteristics
  * Embedding similarity serves as a proxy for reservoir hydrological conditions

### RQ3: Can embedding shifts detect urban encroachment into agricultural land?

* **Hypothesis:** As urbanization progresses, agricultural land proximal to city boundaries will acquire spectral-temporal signatures more similar to urban areas, resulting in decreased cluster separation in embedding space over time.

* **Approach:**

  * Sampled embedding vectors from urban and agricultural polygons near Mysore for 2017 and 2024
  * Conducted unsupervised clustering (KMeans) to identify latent space separability
  * Measured centroid distances and within-class dispersion (separation ratio)
  * Performed two-sample t-tests on distances to urban centroid
  * Visualized change using PCA scatter plots and histograms

* **Findings:**

  * Centroid separation between urban and agricultural classes decreased by 10.6%
  * 2024 agricultural embeddings were ~21% closer to urban centroid than in 2017
  * t-test confirmed significance (p < 0.0001), supporting hypothesis of land cover convergence
  * Spatial signature of agriculture in embedding space is becoming urban-like

---

## Technical Implementation

* **Platform:** Google Earth Engine (Python API)
* **ML Libraries:**

  * scikit-learn (Random Forest, KMeans, PCA, metrics)
  * SciPy (ANOVA, t-tests)
  * NumPy, Pandas
* **Visualization:**

  * Matplotlib
  * Seaborn
* **Preprocessing:**

  * Embedding sampling over defined polygons (urban, agri, water)
  * Standardization for clustering
  * PCA decomposition for visualization and interpretability

---

## Execution Instructions

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Authenticate with Earth Engine:

   ```python
   import ee
   ee.Authenticate()
   ee.Initialize()
   ```
3. Launch the notebook:

   ```bash
   jupyter notebook EarthEMbdRQ_1.ipynb
   ```

---

## Visual Outputs

* Confusion matrix for crop classification
* Classification map of land cover predictions
* Cosine similarity heatmap of reservoir embeddings
* PCA trajectory of water body over climate years
* PCA plots and cluster separation metrics for urbanization analysis

---

## Citation & Attribution

If used in academic work, please cite the Google Earth Engine Satellite Embedding dataset and acknowledge the repository authors.

---

## License

MIT License – open for academic and non-commercial research applications.
