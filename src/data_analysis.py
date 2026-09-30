import matplotlib.pyplot as plt
import os

hist_dir = "outputs/plots/raw_data/hist"
scatter_dir = "outputs/plots/raw_data/scatter"
box_dir = "outputs/plots/raw_data/box"
categorical_plots_dir = "outputs/plots/raw_data/categorical"


os.makedirs(hist_dir, exist_ok= True)
os.makedirs(scatter_dir, exist_ok= True)
os.makedirs(box_dir, exist_ok=True)
os.makedirs(categorical_plots_dir, exist_ok=True)

numeric_features = [
    "bathrooms",
    "bedrooms",
    "price",
    "square_feet",
    "latitude",
    "longitude",
    "time"
]

categorical_features = [
    "category",
    "currency",
    "fee",
    "has_photo",
    "pets_allowed",
    "price_type",
    "cityname",
    "state",
    "source"
]

def numeric_quality(series):
    count = series.count()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bond = q1 - 1.5 * iqr
    upper_bond = q3 + 1.5 * iqr
    lower_outliers = (series < lower_bond).sum()
    upper_outliers = (series > upper_bond).sum()

    lower_outlier_percent = lower_outliers / count * 100
    upper_outlier_percent = upper_outliers / count * 100

    outliers = lower_outliers + upper_outliers
    outliers_percent = outliers / count * 100
    return {
        "Feature": series.name,
        "Count": count,
        "Missing %": series.isna().mean() * 100,
        "Cardinality": series.nunique(),
        "Minimal value": series.min(),
        "Mean value": series.mean(),
        "Median value":series.median(),
        "Max value": series.max(),
        "Q1": q1,
        "Q3": q3,
        "Std": series.std(),
        "IQR": iqr,
        "Lower bound": lower_bond,
        "Lower bonud %": lower_outlier_percent,
        "Upper bound": upper_bond,
        "Upper bound %": upper_outlier_percent,
        "Outliers": outliers,
        "Outliers_%": outliers_percent

        }




def categorical_quality(series):
    count = series.count()
    cardinality = series.nunique()
    value_counts = series.value_counts()

    mode =value_counts.index[0]
    mode_freq = value_counts.iloc[0]
    mode_percent =mode_freq / count * 100

    if cardinality > 1:
        mode_2nd = value_counts.index[1]
        mode_2nd_freq = value_counts.iloc[1]
        mode_2nd_percent = mode_2nd_freq / count * 100
    else: 
        mode_2nd = None
        mode_2nd_freq = None
        mode_2nd_percent = None

    return{
        "Feature": series.name,
        "Count": count,
        "Missing %": series.isna().mean()*100,
        "Cardinality": cardinality,

        "Mode": mode,
        "Mode frequency": mode_freq,
        "Mode %": mode_percent,

        "Mode 2nd": mode_2nd,
        "Mode 2nd frequency": mode_2nd_freq,
        "Mode 2nd %": mode_2nd_percent,        

    }




def plot_histogram(series):
    data =series.dropna()
    plt.figure(figsize=(10,6))
    plt.hist(data, bins = 50)
    plt.title(data.name)
    plt.xlabel(data.name)
    plt.ylabel("Count")
    file_path = os.path.join(
        hist_dir,
        f"{data.name}_hist.png"
        )
    plt.savefig(file_path)
    os.path.join(hist_dir)
    plt.close()




def plot_scatter(df, x_feature, y_feature):
    data = df[[x_feature, y_feature]].dropna()
    plt.figure(figsize=(10,6))
    plt.scatter(
        data[x_feature],
        data[y_feature],
        alpha= 0.2 
        )
    plt.title(f"{x_feature} vs {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    file_path = os.path.join(
        scatter_dir,
        f"{x_feature} vs {y_feature}_scatter.png"
        )
    plt.savefig(file_path)
    os.path.join(scatter_dir)
    plt.close()



def plot_box(series):
    data =series.dropna()
    plt.figure(figsize=(10,6))
    plt.boxplot(
    data,
    tick_labels=[series.name]
    )
    plt.title(data.name)
    plt.ylabel(data.name)
    file_path = os.path.join(
        box_dir,
        f"{data.name}_box.png"
        )
    plt.savefig(file_path)
    os.path.join(box_dir)
    plt.close()

def plot_categorical(series, top_n=15):
    data = series.dropna().value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    plt.bar(data.index.astype(str), data.values)

    plt.title(f"{series.name} distribution")
    plt.xlabel(series.name)
    plt.ylabel("Count")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    file_path = os.path.join(
    categorical_plots_dir,
    f"{series.name}_bar.png"
    )

    plt.savefig(file_path)
    plt.close()
