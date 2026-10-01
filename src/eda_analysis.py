import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PROCESSED_CSV_PATH, FIGURES_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Styling configuration
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#0F172A",
    "figure.facecolor": "#0F172A",
    "text.color": "#F8FAFC",
    "axes.labelcolor": "#94A3B8",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#334155"
})

def generate_all_eda(df_path=PROCESSED_CSV_PATH, figures_dir=FIGURES_DIR, reports_dir=REPORTS_DIR):
    figures_dir = Path(figures_dir)
    reports_dir = Path(reports_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {df_path}...")
    df = pd.read_csv(df_path)
    sample_df = df.sample(min(25000, len(df)), random_state=42)

    summary = {}

    # Q01: Property Price Distribution
    logger.info("Generating Q01: Property Price Distribution...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df['Price_in_Lakhs'], kde=True, color="#38BDF8", bins=40, ax=ax)
    ax.set_title("Q1: Property Price Distribution (₹ in Lakhs)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Listing Price (₹ Lakhs)")
    ax.set_ylabel("Listing Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "q01_price_distribution.png", dpi=150)
    plt.close(fig)
    summary["Q01_Price_Distribution"] = {
        "mean": round(float(df['Price_in_Lakhs'].mean()), 2),
        "median": round(float(df['Price_in_Lakhs'].median()), 2),
        "min": float(df['Price_in_Lakhs'].min()),
        "max": float(df['Price_in_Lakhs'].max())
    }

    # Q02: Property Size Distribution
    logger.info("Generating Q02: Property Size Distribution...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df['Size_in_SqFt'], kde=True, color="#10B981", bins=40, ax=ax)
    ax.set_title("Q2: Property Size Distribution (Square Feet)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Property Area (Sq Ft)")
    ax.set_ylabel("Listing Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "q02_size_distribution.png", dpi=150)
    plt.close(fig)
    summary["Q02_Size_Distribution"] = {
        "mean": round(float(df['Size_in_SqFt'].mean()), 1),
        "median": round(float(df['Size_in_SqFt'].median()), 1),
        "min": int(df['Size_in_SqFt'].min()),
        "max": int(df['Size_in_SqFt'].max())
    }

    # Q03: Price per Sq Ft by Property Type
    logger.info("Generating Q03: Price per Sq Ft by Property Type...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=sample_df, x="Property_Type", y="Price_per_SqFt", palette=["#38BDF8", "#818CF8", "#F472B6"], ax=ax)
    ax.set_title("Q3: Price per Sq Ft by Property Type (₹ Lakhs/Sq Ft)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Property Typology")
    ax.set_ylabel("Mean Price per Sq Ft (₹ Lakhs)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q03_price_per_sqft_by_type.png", dpi=150)
    plt.close(fig)

    # Q04: Property Size vs Total Price
    logger.info("Generating Q04: Size vs Price...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=sample_df.sample(2500, random_state=42), x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.35, color="#818CF8", ax=ax)
    ax.set_title("Q4: Property Size vs Listing Price (Cross-Section)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Size in Sq Ft")
    ax.set_ylabel("Price in Lakhs (₹)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q04_size_vs_price.png", dpi=150)
    plt.close(fig)

    # Q05: Outliers Analysis (Price per Sq Ft)
    logger.info("Generating Q05: Outlier Analysis...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(x=sample_df["Price_per_SqFt"], color="#F59E0B", ax=ax)
    ax.set_title("Q5: Price per Sq Ft Outlier Distribution (IQR Method)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Price per Sq Ft (₹ Lakhs)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q05_outliers_analysis.png", dpi=150)
    plt.close(fig)

    # Q06: Average Price per Sq Ft by State
    logger.info("Generating Q06: Price per Sq Ft by State...")
    state_sqft = df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=state_sqft.values, y=state_sqft.index, palette="mako", ax=ax)
    ax.set_title("Q6: Mean Price per Sq Ft by State (Top 12 States)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Price per Sq Ft (₹ Lakhs)")
    ax.set_ylabel("State")
    fig.tight_layout()
    fig.savefig(figures_dir / "q06_price_per_sqft_by_state.png", dpi=150)
    plt.close(fig)

    # Q07: Average Property Price by City
    logger.info("Generating Q07: Average Price by City...")
    city_price = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False)
    top_cities = pd.concat([city_price.head(5), city_price.tail(5)])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=top_cities.values, y=top_cities.index, palette="viridis", ax=ax)
    ax.set_title("Q7: Average Property Price Across Key Municipal Cities", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Price (₹ Lakhs)")
    ax.set_ylabel("City")
    fig.tight_layout()
    fig.savefig(figures_dir / "q07_avg_price_by_city.png", dpi=150)
    plt.close(fig)

    # Q08: Median Age of Properties by Locality
    logger.info("Generating Q08: Median Age by Locality...")
    loc_age = df.groupby('Locality')['Age_of_Property'].median().head(12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=loc_age.values, y=loc_age.index, palette="crest", ax=ax)
    ax.set_title("Q8: Median Age of Properties Across Sample Localities", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Median Age (Years)")
    ax.set_ylabel("Locality")
    fig.tight_layout()
    fig.savefig(figures_dir / "q08_median_age_by_locality.png", dpi=150)
    plt.close(fig)

    # Q09: BHK Distribution Across Cities
    logger.info("Generating Q09: BHK Distribution by City...")
    bhk_city = pd.crosstab(df['City'], df['BHK'], normalize='index').head(8) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bhk_city.plot(kind='barh', stacked=True, colormap='tab10', ax=ax)
    ax.set_title("Q9: BHK Composition Across Major Cities (% Share)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("% Share of Inventory")
    ax.set_ylabel("City")
    ax.legend(title="BHK", bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(figures_dir / "q09_bhk_distribution_by_city.png", dpi=150)
    plt.close(fig)

    # Q10: Top 5 Most Expensive Localities
    logger.info("Generating Q10: Top 5 Expensive Localities...")
    top5_loc = df.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=top5_loc.values, y=top5_loc.index, palette="flare", ax=ax)
    ax.set_title("Q10: Top 5 Most Expensive Residential Localities", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Price (₹ Lakhs)")
    ax.set_ylabel("Locality")
    fig.tight_layout()
    fig.savefig(figures_dir / "q10_top5_expensive_localities.png", dpi=150)
    plt.close(fig)

    # Q11: Numeric Feature Correlation Matrix
    logger.info("Generating Q11: Numeric Correlation...")
    num_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals']
    corr = sample_df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title("Q11: Numeric Feature Correlation Matrix", fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "q11_numeric_correlation.png", dpi=150)
    plt.close(fig)

    # Q12: Nearby Schools vs Price per Sq Ft
    logger.info("Generating Q12: Schools vs Price per Sq Ft...")
    schools_sqft = df.groupby('Nearby_Schools')['Price_per_SqFt'].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(x=schools_sqft.index, y=schools_sqft.values, marker="o", color="#38BDF8", linewidth=2.5, ax=ax)
    ax.set_title("Q12: Nearby Schools Density vs Mean Price per Sq Ft", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Nearby Schools Count (0 to 10)")
    ax.set_ylabel("Mean Price per Sq Ft (₹ Lakhs)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q12_schools_vs_price_per_sqft.png", dpi=150)
    plt.close(fig)

    # Q13: Nearby Hospitals vs Price per Sq Ft
    logger.info("Generating Q13: Hospitals vs Price per Sq Ft...")
    hosp_sqft = df.groupby('Nearby_Hospitals')['Price_per_SqFt'].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(x=hosp_sqft.index, y=hosp_sqft.values, marker="s", color="#10B981", linewidth=2.5, ax=ax)
    ax.set_title("Q13: Nearby Hospitals Density vs Mean Price per Sq Ft", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Nearby Hospitals Count (0 to 10)")
    ax.set_ylabel("Mean Price per Sq Ft (₹ Lakhs)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q13_hospitals_vs_price_per_sqft.png", dpi=150)
    plt.close(fig)

    # Q14: Property Price by Furnished Status
    logger.info("Generating Q14: Price by Furnished Status...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=sample_df, x="Furnished_Status", y="Price_in_Lakhs", palette="Set2", ax=ax)
    ax.set_title("Q14: Property Price Distribution by Furnished Status", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Furnishing Level")
    ax.set_ylabel("Price in Lakhs (₹)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q14_price_by_furnished_status.png", dpi=150)
    plt.close(fig)

    # Q15: Price per Sq Ft by Facing Direction
    logger.info("Generating Q15: Price per Sq Ft by Facing...")
    facing_sqft = df.groupby('Facing')['Price_per_SqFt'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=facing_sqft, x="Facing", y="Price_per_SqFt", palette="Pastel1", ax=ax)
    ax.set_title("Q15: Price per Sq Ft Across Property Facing Directions", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Cardinal Facing Direction")
    ax.set_ylabel("Mean Price per Sq Ft (₹ Lakhs)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q15_price_per_sqft_by_facing.png", dpi=150)
    plt.close(fig)

    # Q16: Properties by Owner Type
    logger.info("Generating Q16: Owner Type Distribution...")
    owner_counts = df['Owner_Type'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=owner_counts.index, y=owner_counts.values, palette=["#38BDF8", "#818CF8", "#F59E0B"], ax=ax)
    ax.set_title("Q16: Inventory Volume Across Owner Types", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Seller / Owner Category")
    ax.set_ylabel("Total Property Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "q16_owner_type_distribution.png", dpi=150)
    plt.close(fig)

    # Q17: Properties by Availability Status
    logger.info("Generating Q17: Availability Status Distribution...")
    avail_counts = df['Availability_Status'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=avail_counts.index, y=avail_counts.values, palette=["#10B981", "#6366F1"], ax=ax)
    ax.set_title("Q17: Property Inventory by Availability Status", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Construction / Availability Status")
    ax.set_ylabel("Total Property Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "q17_availability_status_distribution.png", dpi=150)
    plt.close(fig)

    # Q18: Parking Space vs Property Price
    logger.info("Generating Q18: Parking vs Price...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=sample_df, x="Parking_Space", y="Price_in_Lakhs", palette=["#EF4444", "#10B981"], ax=ax)
    ax.set_title("Q18: Property Price Distribution by Parking Availability", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Dedicated Parking Space Available")
    ax.set_ylabel("Price in Lakhs (₹)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q18_parking_vs_price.png", dpi=150)
    plt.close(fig)

    # Q19: Amenities Available vs Price per Sq Ft
    logger.info("Generating Q19: Amenities vs Price per Sq Ft...")
    amenity_sqft = df.groupby('amenity_count')['Price_per_SqFt'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=amenity_sqft, x="amenity_count", y="Price_per_SqFt", palette="crest", ax=ax)
    ax.set_title("Q19: Amenity Count vs Mean Price per Sq Ft", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Total Number of Amenities (0 to 5)")
    ax.set_ylabel("Mean Price per Sq Ft (₹ Lakhs)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q19_amenities_vs_price_per_sqft.png", dpi=150)
    plt.close(fig)

    # Q20: Public Transport vs Investment Potential
    logger.info("Generating Q20: Transport vs Investment Potential...")
    trans_inv = df.groupby('Public_Transport_Accessibility')['High_Investment_Potential'].mean().reset_index()
    trans_inv['Pct_High_Potential'] = trans_inv['High_Investment_Potential'] * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=trans_inv, x="Public_Transport_Accessibility", y="Pct_High_Potential", palette=["#10B981", "#6366F1", "#F59E0B"], ax=ax)
    ax.set_title("Q20: Public Transit Connectivity vs High Investment Potential Rate", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Public Transport Accessibility Tier")
    ax.set_ylabel("% Qualifying as High Potential (Score >= 70)")
    fig.tight_layout()
    fig.savefig(figures_dir / "q20_transport_vs_investment.png", dpi=150)
    plt.close(fig)

    logger.info("All 20 EDA figures successfully generated.")

if __name__ == "__main__":
    generate_all_eda()
