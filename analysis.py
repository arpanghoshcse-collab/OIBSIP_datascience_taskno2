#!/usr/bin/env python3
"""
Unemployment Analysis with Python: COVID-19 Impact on India
Author: Antigravity Pair Programmer
Tech Stack: Python, pandas, matplotlib, seaborn, numpy
"""

import os
import sys

# Configure cache and headless backend for matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
sns.set_theme(style='whitegrid', font='sans-serif')
palette = sns.color_palette("deep")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300

CHARTS_DIR = 'charts'
DATA_DIR = 'data'
os.makedirs(CHARTS_DIR, exist_ok=True)


def load_and_preprocess_data():
    """Loads, cleans, and pre-processes both unemployment datasets."""
    print("=" * 70)
    print("1. LOADING & PREPROCESSING DATASETS")
    print("=" * 70)

    # --- Dataset 1: Unemployment in India.csv (Includes Rural & Urban data) ---
    path1 = os.path.join(DATA_DIR, 'Unemployment in India.csv')
    df_raw1 = pd.read_csv(path1)
    print(f"[Dataset 1] Raw Shape: {df_raw1.shape}")
    print(f"[Dataset 1] Raw Null Counts:\n{df_raw1.isnull().sum()}\n")

    # Drop blank rows
    df1 = df_raw1.dropna().copy()
    # Strip whitespace from column names
    df1.columns = df1.columns.str.strip()
    # Strip strings from object columns
    for col in ['Region', 'Frequency', 'Area']:
        if col in df1.columns:
            df1[col] = df1[col].astype(str).str.strip()

    # Parse dates
    df1['Date'] = pd.to_datetime(df1['Date'].str.strip(), format='%d-%m-%Y')
    df1 = df1.sort_values(by='Date').reset_index(drop=True)
    df1['Year'] = df1['Date'].dt.year
    df1['Month'] = df1['Date'].dt.month
    df1['Month_Name'] = df1['Date'].dt.strftime('%b %Y')
    df1['Period'] = np.where(
        df1['Date'] < '2020-04-01',
        'Pre-COVID (May 2019 - Mar 2020)',
        'Lockdown Peak (Apr 2020 - Jun 2020)'
    )

    print(f"[Dataset 1] Cleaned Shape: {df1.shape}")
    print(f"[Dataset 1] Date Range: {df1['Date'].min().strftime('%Y-%m-%d')} to {df1['Date'].max().strftime('%Y-%m-%d')}")
    print(f"[Dataset 1] States Count: {df1['Region'].nunique()}")
    print(f"[Dataset 1] Area Categories: {df1['Area'].unique().tolist()}\n")

    # --- Dataset 2: Unemployment_Rate_upto_11_2020.csv (Zone-level 2020 data) ---
    path2 = os.path.join(DATA_DIR, 'Unemployment_Rate_upto_11_2020.csv')
    df_raw2 = pd.read_csv(path2)
    print(f"[Dataset 2] Raw Shape: {df_raw2.shape}")
    print(f"[Dataset 2] Raw Null Counts:\n{df_raw2.isnull().sum()}\n")

    df2 = df_raw2.dropna().copy()
    df2.columns = df2.columns.str.strip()
    for col in ['Region', 'Frequency', 'Region.1']:
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.strip()

    df2.rename(columns={'Region.1': 'Zone'}, inplace=True)
    df2['Date'] = pd.to_datetime(df2['Date'].str.strip(), format='%d-%m-%Y')
    df2 = df2.sort_values(by='Date').reset_index(drop=True)
    df2['Year'] = df2['Date'].dt.year
    df2['Month'] = df2['Date'].dt.month
    df2['Month_Name'] = df2['Date'].dt.strftime('%b %Y')
    df2['Period'] = np.where(
        df2['Date'] < '2020-04-01',
        'Pre-COVID (Jan - Mar 2020)',
        'Lockdown & Unlock (Apr - Oct 2020)'
    )

    print(f"[Dataset 2] Cleaned Shape: {df2.shape}")
    print(f"[Dataset 2] Date Range: {df2['Date'].min().strftime('%Y-%m-%d')} to {df2['Date'].max().strftime('%Y-%m-%d')}")
    print(f"[Dataset 2] Zones: {df2['Zone'].unique().tolist()}\n")

    return df1, df2


def print_statistical_summaries(df1, df2):
    """Outputs comprehensive numerical summaries and comparisons."""
    print("=" * 70)
    print("2. DESCRIPTIVE STATISTICS & INSIGHTS")
    print("=" * 70)

    print("\n--- Summary Statistics (Dataset 1) ---")
    cols1 = ['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']
    print(df1[cols1].describe().round(2))

    print("\n--- Urban vs. Rural Unemployment Comparison ---")
    area_stats = df1.groupby('Area')['Estimated Unemployment Rate (%)'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    print(area_stats)

    print("\n--- Pre-COVID vs. Lockdown Period Unemployment Rates ---")
    period_stats = df1.groupby('Period')['Estimated Unemployment Rate (%)'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    print(period_stats)

    print("\n--- Zone-wise Unemployment Summary (2020) ---")
    zone_stats = df2.groupby('Zone')['Estimated Unemployment Rate (%)'].agg(['mean', 'median', 'std']).round(2).sort_values(by='mean', ascending=False)
    print(zone_stats)

    print("\n--- Top 10 States with Highest Overall Average Unemployment Rate ---")
    top10_states = df1.groupby('Region')['Estimated Unemployment Rate (%)'].mean().round(2).sort_values(ascending=False).head(10)
    for rank, (state, rate) in enumerate(top10_states.items(), 1):
        print(f"  {rank:2d}. {state:20s}: {rate:5.2f}%")

    print("\n--- State-Wise Pre vs. Post COVID Impact Analysis ---")
    pvt = df1.pivot_table(index='Region', columns='Period', values='Estimated Unemployment Rate (%)', aggfunc='mean')
    pre_col = 'Pre-COVID (May 2019 - Mar 2020)'
    post_col = 'Lockdown Peak (Apr 2020 - Jun 2020)'
    pvt['Absolute Increase (% pts)'] = pvt[post_col] - pvt[pre_col]
    pvt['Percentage Increase (%)'] = (pvt['Absolute Increase (% pts)'] / pvt[pre_col]) * 100
    pvt_sorted = pvt.sort_values(by='Absolute Increase (% pts)', ascending=False).round(2)
    print(pvt_sorted.head(10))
    print()


def plot_top10_states(df1):
    """Plot Top 10 states with highest average unemployment rates."""
    plt.figure(figsize=(10, 6))
    top10 = df1.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=True).tail(10)
    
    colors = sns.color_palette("Reds_r", n_colors=10)[::-1]
    bars = plt.barh(top10.index, top10.values, color=colors, edgecolor='black', linewidth=0.7)
    
    plt.title('Top 10 Indian States by Average Unemployment Rate (May 2019 - Jun 2020)', pad=15, weight='bold')
    plt.xlabel('Average Estimated Unemployment Rate (%)')
    plt.ylabel('State / Union Territory')
    plt.xlim(0, max(top10.values) + 5)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%',
                 va='center', ha='left', fontsize=10, weight='semibold', color='#333333')
        
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, '01_top10_unemployment_states.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] {out_path}")


def plot_urban_vs_rural(df1):
    """Plot time-series comparison of Urban vs Rural unemployment rates."""
    plt.figure(figsize=(12, 6))
    
    monthly_area = df1.groupby(['Date', 'Area'])['Estimated Unemployment Rate (%)'].mean().reset_index()
    
    sns.lineplot(
        data=monthly_area,
        x='Date',
        y='Estimated Unemployment Rate (%)',
        hue='Area',
        marker='o',
        linewidth=2.5,
        palette={'Rural': '#2b8a3e', 'Urban': '#d9480f'}
    )
    
    # Highlight lockdown period
    lockdown_start = pd.to_datetime('2020-03-24')
    lockdown_end = pd.to_datetime('2020-06-30')
    plt.axvspan(lockdown_start, lockdown_end, color='gray', alpha=0.2, label='Lockdown Period (Phase 1-4)')
    
    plt.title('Temporal Trend: Urban vs. Rural Unemployment Rate in India (2019 - 2020)', pad=15, weight='bold')
    plt.xlabel('Date')
    plt.ylabel('Estimated Unemployment Rate (%)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.legend(title='Area', loc='upper left', frameon=True)
    
    # Annotate peak
    peak_date = pd.to_datetime('2020-05-31')
    plt.annotate(
        'Lockdown Surge\nUrban peaked at ~25%',
        xy=(peak_date, 25.0),
        xytext=(pd.to_datetime('2020-01-01'), 27),
        arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
        weight='bold', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6)
    )
    
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, '02_urban_vs_rural_trend.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] {out_path}")


def plot_timeseries_major_states(df1):
    """Plot time-series line chart for major states across India."""
    plt.figure(figsize=(13, 6))
    
    major_states = ['Maharashtra', 'Delhi', 'Tamil Nadu', 'Uttar Pradesh', 'Bihar', 'West Bengal']
    df_major = df1[df1['Region'].isin(major_states)].groupby(['Date', 'Region'])['Estimated Unemployment Rate (%)'].mean().reset_index()
    
    palette_states = sns.color_palette("tab10", n_colors=len(major_states))
    
    sns.lineplot(
        data=df_major,
        x='Date',
        y='Estimated Unemployment Rate (%)',
        hue='Region',
        marker='s',
        linewidth=2.2,
        palette=palette_states
    )
    
    # Add vertical line for national lockdown announcement
    plt.axvline(pd.to_datetime('2020-03-24'), color='red', linestyle='--', linewidth=1.5, label='National Lockdown (24 Mar 2020)')
    
    plt.title('Unemployment Rate Trajectory for Major Indian States (May 2019 - Jun 2020)', pad=15, weight='bold')
    plt.xlabel('Date')
    plt.ylabel('Estimated Unemployment Rate (%)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.legend(title='State / UT', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, '03_timeseries_major_states.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] {out_path}")


def plot_correlation_heatmap(df1):
    """Correlation heatmap between Unemployment Rate, Employed count, and Labour Participation Rate."""
    plt.figure(figsize=(8, 6))
    
    cols = ['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']
    short_labels = ['Unemployment Rate (%)', 'Employed Count', 'Labour Participation (%)']
    
    corr = df1[cols].corr()
    corr.columns = short_labels
    corr.index = short_labels
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(
        corr,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        vmin=-0.3,
        vmax=1.0,
        square=True,
        linewidths=1.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient'}
    )
    
    plt.title('Correlation Matrix: Unemployment, Employment & Participation', pad=15, weight='bold')
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, '04_correlation_heatmap.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] {out_path}")


def plot_precovid_vs_postcovid(df1):
    """Side-by-side grouped bar chart comparing Pre-COVID vs Post-COVID lockdown unemployment."""
    pvt = df1.pivot_table(index='Region', columns='Period', values='Estimated Unemployment Rate (%)', aggfunc='mean')
    pre_col = 'Pre-COVID (May 2019 - Mar 2020)'
    post_col = 'Lockdown Peak (Apr 2020 - Jun 2020)'
    pvt['Increase'] = pvt[post_col] - pvt[pre_col]
    top_affected = pvt.sort_values(by='Increase', ascending=False).head(10).reset_index()
    
    df_melted = pd.melt(
        top_affected,
        id_vars=['Region'],
        value_vars=[pre_col, post_col],
        var_name='Period',
        value_name='Unemployment Rate (%)'
    )
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_melted,
        x='Region',
        y='Unemployment Rate (%)',
        hue='Period',
        palette={'Pre-COVID (May 2019 - Mar 2020)': '#4c6ef5', 'Lockdown Peak (Apr 2020 - Jun 2020)': '#fa5252'},
        edgecolor='black',
        linewidth=0.7
    )
    
    plt.title('Pre-COVID vs. Lockdown Peak Unemployment Rate (Top 10 Impacted States)', pad=15, weight='bold')
    plt.xlabel('State / Union Territory')
    plt.ylabel('Estimated Unemployment Rate (%)')
    plt.xticks(rotation=35, ha='right')
    plt.legend(title='Period', frameon=True)
    
    # Label bar heights
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8.5, weight='semibold',
                        xytext=(0, 2), textcoords='offset points')
            
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, '05_precovid_vs_postcovid_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] {out_path}")


def plot_zone_unemployment_trends(df2):
    """Plot zone-wise monthly unemployment evolution in 2020."""
    plt.figure(figsize=(11, 6))
    
    zone_monthly = df2.groupby(['Date', 'Zone'])['Estimated Unemployment Rate (%)'].mean().reset_index()
    
    sns.lineplot(
        data=zone_monthly,
        x='Date',
        y='Estimated Unemployment Rate (%)',
        hue='Zone',
        marker='o',
        linewidth=2.2,
        palette='Set1'
    )
    
    plt.axvline(pd.to_datetime('2020-03-24'), color='red', linestyle='--', linewidth=1.5, label='National Lockdown (Mar 2020)')
    plt.title('Zone-Wise Unemployment Rate Evolution in India (Jan 2020 - Oct 2020)', pad=15, weight='bold')
    plt.xlabel('Date')
    plt.ylabel('Estimated Unemployment Rate (%)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.legend(title='Geographic Zone', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, '06_zone_unemployment_trends.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] {out_path}")


def plot_state_month_heatmap(df1):
    """Heatmap showing unemployment intensity across all states and months."""
    plt.figure(figsize=(14, 10))
    
    df1_sorted = df1.sort_values(by='Date')
    df1_sorted['Month_Year'] = df1_sorted['Date'].dt.strftime('%Y-%m')
    
    pivot_heatmap = df1_sorted.pivot_table(
        index='Region',
        columns='Month_Year',
        values='Estimated Unemployment Rate (%)',
        aggfunc='mean'
    )
    
    sns.heatmap(
        pivot_heatmap,
        cmap='YlOrRd',
        annot=True,
        fmt='.1f',
        linewidths=0.5,
        cbar_kws={'label': 'Unemployment Rate (%)'}
    )
    
    plt.title('State-wise Monthly Unemployment Rate Heatmap (May 2019 - Jun 2020)', pad=15, weight='bold')
    plt.xlabel('Month-Year')
    plt.ylabel('State / Union Territory')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, '07_state_month_intensity_heatmap.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] {out_path}")


def main():
    print("Starting Comprehensive Unemployment Analysis...")
    df1, df2 = load_and_preprocess_data()
    print_statistical_summaries(df1, df2)
    
    print("=" * 70)
    print("3. GENERATING HIGH-RESOLUTION CHARTS")
    print("=" * 70)
    plot_top10_states(df1)
    plot_urban_vs_rural(df1)
    plot_timeseries_major_states(df1)
    plot_correlation_heatmap(df1)
    plot_precovid_vs_postcovid(df1)
    plot_zone_unemployment_trends(df2)
    plot_state_month_heatmap(df1)
    
    print("\nAll charts successfully generated and saved to the 'charts/' folder.")


if __name__ == '__main__':
    main()
