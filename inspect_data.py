import pandas as pd
import numpy as np

def run_inspection():
    df = pd.read_csv('india_housing_prices.csv')
    print("=== BASIC INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Nulls total: {df.isnull().sum().sum()}")
    print(f"Duplicates total: {df.duplicated().sum()}")
    
    print("\n=== PRICE PER SQFT RELATIONSHIP ===")
    ratio = df['Price_in_Lakhs'] / df['Size_in_SqFt']
    print("Calculated (Price_in_Lakhs / Size_in_SqFt) vs Price_per_SqFt:")
    print("Ratio describe:")
    print(ratio.describe())
    print("Column Price_per_SqFt describe:")
    print(df['Price_per_SqFt'].describe())
    diff = (df['Price_per_SqFt'] - ratio.round(2)).abs()
    print("Abs diff (Price_per_SqFt - round(ratio, 2)):")
    print(diff.describe())
    
    print("\n=== YEAR BUILT VS AGE ===")
    ref_year = df['Year_Built'] + df['Age_of_Property']
    print("Year_Built + Age_of_Property distribution:")
    print(ref_year.value_counts())
    
    print("\n=== FLOOR NO VS TOTAL FLOORS ===")
    floor_mismatch = (df['Floor_No'] > df['Total_Floors']).sum()
    print(f"Rows where Floor_No > Total_Floors: {floor_mismatch} ({floor_mismatch/len(df):.2%})")
    
    print("\n=== AMENITIES ===")
    all_amenities = set()
    for a in df['Amenities'].dropna():
        for item in a.split(','):
            all_amenities.add(item.strip())
    print(f"All individual amenities: {sorted(list(all_amenities))}")
    
    print("\n=== CATEGORICAL VALUES ===")
    for col in ['Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility', 
                'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status']:
        print(f"{col}: {df[col].value_counts().to_dict()}")

    print("\n=== CORRELATIONS WITH PRICE ===")
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(df[num_cols].corr()['Price_in_Lakhs'].sort_values(ascending=False))

    print("\n=== STATISTICAL DISTRIBUTION OF PRICE & SIZE ===")
    for c in ['Price_in_Lakhs', 'Size_in_SqFt', 'Price_per_SqFt', 'BHK', 'Age_of_Property']:
        q25, q75 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers = ((df[c] < lower) | (df[c] > upper)).sum()
        print(f"{c}: min={df[c].min()}, q25={q25}, median={df[c].median()}, q75={q75}, max={df[c].max()}, outliers={outliers} ({outliers/len(df):.2%})")

if __name__ == '__main__':
    run_inspection()
