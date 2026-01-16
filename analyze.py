import os
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# CONFIG (EDIT THIS ONLY)
# =======================
DATASET_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
CATEGORY_COL = ["class", "sex", "pclass", "embark_town", "age"]
VALUE_COL = "fare"

def main():
    for x in CATEGORY_COL: 
        chartmake(x)
    



def chartmake(x):
    OUTPUT_PATH = os.path.join("output", "chart" + str(CATEGORY_COL.index(x) + 1) + ".png")
    df = pd.read_csv(DATASET_URL)

    # Binning for age groups
    if x == "age":
        bin_labels = ['0 - 19', '20 - 39', '40 - 59', '60 - 80']
        df["age_group"] = pd.cut(df["age"], bins=4, labels=bin_labels)

    # Terminal summary (required)
    print("Rows, Columns:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head(5))

    # Clean: keep only needed columns and remove missing values, specify binning for age
    col_to_use = "age_group" if x == "age" else x
    df_small = df[[col_to_use, VALUE_COL]].dropna()

    # Convert numeric column safely
    df_small[VALUE_COL] = pd.to_numeric(df_small[VALUE_COL], errors="coerce")
    df_small = df_small.dropna()

    # Group mean by category
    grouped = df_small.groupby(col_to_use)[VALUE_COL].mean().sort_values(ascending=False)

    print("\nAverage values by category:")
    print(grouped)

    # Plot
    os.makedirs("output", exist_ok=True)
    ax = grouped.plot(kind="bar")
    ax.set_title(f"Average {VALUE_COL} by {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(f"Average {VALUE_COL}")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"\nSaved chart to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()