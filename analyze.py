import os
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# CONFIG (EDIT THIS ONLY)
# =======================

# class, age, sex, pclass, age_group, embarked
DATASET_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
CATEGORY_COL = "age_group"
VALUE_COL = "fare"
OUTPUT_PATH = os.path.join("output", "chart5.png")


def main():
    df = pd.read_csv(DATASET_URL)

    # Terminal summary (required)
    print("Rows, Columns:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head(5))

    # Binning for age groups
    bin_labels = ['0 - 19', '20 - 39', '40 - 59', '60 - 80']
    df["age_group"] = pd.cut(df["age"], bins=4, labels=bin_labels)

    # Clean: keep only needed columns and remove missing values
    df_small = df[[CATEGORY_COL, VALUE_COL]].dropna()

    # Convert numeric column safely
    df_small[VALUE_COL] = pd.to_numeric(df_small[VALUE_COL], errors="coerce")
    df_small = df_small.dropna()

    # Group mean by category
    grouped = df_small.groupby(CATEGORY_COL)[VALUE_COL].mean().sort_values(ascending=False)

    print("\nAverage values by category:")
    print(grouped)
    
    print(df_small.head())

    # Plot
    os.makedirs("output", exist_ok=True)
    ax = grouped.plot(kind="bar")
    ax.set_title(f"Average {VALUE_COL} by {CATEGORY_COL}")
    ax.set_xlabel(CATEGORY_COL)
    ax.set_ylabel(f"Average {VALUE_COL}")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"\nSaved chart to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()