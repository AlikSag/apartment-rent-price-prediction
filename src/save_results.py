import os
import pandas as pd


def save_results(results, file_path):
    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    df = pd.DataFrame(results)

    df.to_csv(
        file_path,
        index=False
    )
