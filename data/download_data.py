import pandas as pd
import os

def load_and_inspect(data_path="data/inshort_news_data.csv"):
    """
    Charge et inspecte le dataset.
    Args:
        data_path (str): Chemin vers le fichier CSV
    Returns:
        DataFrame pandas
    """
    # Chargement
    df = pd.read_csv(data_path)

    print("=" * 50)
    print("INSPECTION DU DATASET")
    print("=" * 50)

    # Infos générales
    print(f"\nNombre total d'exemples : {len(df)}")
    print(f"Colonnes : {df.columns.tolist()}")

    # Distribution des classes
    print("\nDistribution des classes :")
    print(df["news_category"].value_counts())

    # Longueur des textes
    df["text_len"] = df["news_article"].str.split().str.len()
    print(f"\nLongueur des textes (en mots) :")
    print(df["text_len"].describe())

    # 5 exemples
    print("\n5 exemples du dataset :")
    for _, row in df.sample(5, random_state=42).iterrows():
        print(f"  Catégorie : {row['news_category']}")
        print(f"  Texte     : {row['news_article'][:100]}...")
        print()

    return df


if __name__ == "__main__":
    load_and_inspect()