# Unemployment Analysis with Python (COVID-19 Impact in India)

## 📌 Project Overview
This repository contains an end-to-end Exploratory Data Analysis (EDA) and time-series investigation into India's unemployment dynamics before, during, and after the acute COVID-19 pandemic lockdowns. 

Using Python, pandas, matplotlib, and seaborn, this project analyzes regional disparities, sectoral asymmetries (Urban vs. Rural), macro-zone patterns, and the dramatic labor market contraction following the March 2020 national lockdown.

---

## 🚀 Key Feature Checklist
- [x] **Dataset Sourcing & Ingestion**: Loaded and audited both standard Kaggle datasets (`Unemployment in India.csv` and `Unemployment_Rate_upto_11_2020.csv`).
- [x] **Data Cleaning & Preprocessing**: Handled whitespace trimming on headers and string values, dropped empty padding rows, parsed dates, and engineered temporal features.
- [x] **Regional & Temporal EDA**: Computed summary statistics across 28 states, 5 geographic zones, and Urban vs. Rural divisions.
- [x] **Time-Series Analysis**: Multi-line visualization tracking unemployment trajectories across major economic hubs (Maharashtra, Delhi, Tamil Nadu, Uttar Pradesh, Bihar, West Bengal).
- [x] **Top 10 High-Unemployment States**: Bar chart quantifying states with the highest sustained average unemployment rates.
- [x] **Correlation Heatmap**: Pearson correlation matrix analyzing interactions between Unemployment Rate, Employed workforce count, and Labour Participation Rate.
- [x] **Pre-COVID vs. Post-COVID Comparison**: Chronological partition (cutoff: April 2020) calculating absolute delta and percentage surges per state.
- [x] **Rich Analytical Commentary**: Markdown narratives between every chart explaining the socio-economic context and labor market implications.
- [x] **Production Jupyter Notebook**: Fully executed, clean, well-commented Jupyter Notebook (`unemployment_analysis.ipynb`) with pre-computed visualizations and tables.
- [x] **Standalone CLI Script**: Reproducible Python script (`analysis.py`) that executes the entire analytical pipeline headlessly and saves all figures.

---

## 📊 Summary of Key Findings

| Metric / Dimension | Pre-COVID (May 2019 - Mar 2020) | Lockdown Peak (Apr 2020 - Jun 2020) | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **National Unemployment Average** | **9.61%** | **20.19%** (Peaked at >23% in May 2020) | Unemployment more than doubled across India during the initial lockdown. |
| **Urban Unemployment** | ~10.5% | **25.0%** | Hardest hit due to sudden halt of services, hospitality, transport, and informal urban labor. |
| **Rural Unemployment** | ~7.8% | **19.3%** | Partially buffered by agricultural activity and MGNREGA safety nets. |
| **Puducherry** | 1.58% | **57.70%** (+3,548%) | Extreme surge driven by sudden paralysis of tourism and service enterprises. |
| **Jharkhand & Bihar** | ~13.9% | **44.9% & 37.0%** | Massive vulnerability compounded by returning migrant labor. |
| **Tamil Nadu** | 3.16% | **31.74%** (+903%) | Severe manufacturing, textile, and automotive cluster shutdowns. |
| **Post-Lockdown Recovery** | — | Single digits (**8.03% by Oct 2020**) | Sharp "V-shaped" initial rebound under phased national Unlock guidelines. |

---

## 📁 Repository Structure
```
unemployment_analysis_india/
├── README.md                          # Project documentation and summary
├── requirements.txt                    # Python dependencies
├── analysis.py                         # Standalone executable data pipeline
├── build_notebook.py                   # Script to programmatically generate notebook
├── unemployment_analysis.ipynb         # Master Jupyter Notebook with rendered outputs
├── data/                               # Raw datasets
│   ├── Unemployment in India.csv      # May 2019 - Jun 2020 (Rural/Urban granular)
│   └── Unemployment_Rate_upto_11_2020.csv # Jan 2020 - Oct 2020 (Zone/Geographic)
└── charts/                             # High-resolution (300 DPI) exported figures
    ├── 01_top10_unemployment_states.png
    ├── 02_urban_vs_rural_trend.png
    ├── 03_timeseries_major_states.png
    ├── 04_correlation_heatmap.png
    ├── 05_precovid_vs_postcovid_comparison.png
    ├── 06_zone_unemployment_trends.png
    └── 07_state_month_intensity_heatmap.png
```

---

## 🛠️ Installation & Reproduction

### 1. Prerequisites
Ensure Python 3.10+ is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Standalone Script
To execute the automated pipeline and re-generate all charts:
```bash
python3 analysis.py
```

### 4. Open the Jupyter Notebook
Launch Jupyter or open directly in VS Code / Cursor:
```bash
jupyter notebook unemployment_analysis.ipynb
```

---

## 📈 Visualizations Overview

1. **`01_top10_unemployment_states.png`**: Identifies states with highest baseline unemployment (Tripura, Haryana, Jharkhand, Bihar).
2. **`02_urban_vs_rural_trend.png`**: Traces the monthly divergence between Urban and Rural economies, highlighting the April-May 2020 lockdown apex.
3. **`03_timeseries_major_states.png`**: Multi-line tracking of economic powerhouses (Maharashtra, Delhi, Tamil Nadu, UP, Bihar, WB).
4. **`04_correlation_heatmap.png`**: Quantifies Pearson correlation between Unemployment Rate, Employed count, and Labour Participation.
5. **`05_precovid_vs_postcovid_comparison.png`**: Grouped comparative bar chart visualizing the magnitude of the pandemic shock on hardest-hit states.
6. **`06_zone_unemployment_trends.png`**: Regional zone trajectories (North, East, South, West, Northeast) showing the V-shaped unlock recovery.
7. **`07_state_month_intensity_heatmap.png`**: Comprehensive 28-state matrix highlighting the synchronized nationwide shock in April-May 2020.

---

## 💡 Policy Insights
- **Urban Social Safety Nets**: Need for an urban counterpart to MGNREGA to shield informal, daily-wage service labor during localized or systemic disruptions.
- **Portability of Benefits**: Universalization of welfare schemes (e.g., One Nation One Ration Card) to protect inter-state migrant labor forces.
- **Regional Industrial Diversification**: Targeted employment stimulus in states like Haryana, Bihar, and Jharkhand with chronic structural unemployment.
