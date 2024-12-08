# validate_raw_data.py
# Author: Zoe Ren
# 7 December 2024
# Run by following command: python /scripts/validate_raw_data.py --input-path "./data/raw/wine_quality.csv" --processed-data-path "./data/processed"

import os
import click
import pandas as pd
import pandera as pa


@click.command()
@click.option(
    "--input-path",
    default="./data/raw/wine_quality.csv",
    help="Path to the input CSV file.",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--processed-data-path",
    default="./data/processed",
    help="Path to save the processed data.",
    type=click.Path(file_okay=False),
)
def validate_raw_data(input_path, processed_data_path, seed):
    """
    Script to validate and clean wine quality data.
    Validate the raw data is step 1 of data validation done by pandera before data splitting. 
    Validate the training data is step 2 of data validation done by deepchecks after data splitting.
    """
    # Make sure folder exists for output
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Read the data
    print(f"Reading data from {input_path}...")
    raw_data = pd.read_csv(input_path)

    # Define the schema for validation
    schema = pa.DataFrameSchema(
        {
            "color": pa.Column(str, pa.Check.isin(["red", "white"])),
            "fixed_acidity": pa.Column(float, pa.Check.between(0, 16), nullable=True),
            "volatile_acidity": pa.Column(float, pa.Check.between(0, 1.8), nullable=True),
            "citric_acid": pa.Column(float, pa.Check.between(0, 1.4), nullable=True),
            "residual_sugar": pa.Column(float, pa.Check.between(0, 30), nullable=True),
            "chlorides": pa.Column(float, pa.Check.between(0, 0.7), nullable=True),
            "free_sulfur_dioxide": pa.Column(float, pa.Check.between(0, 160), nullable=True),
            "total_sulfur_dioxide": pa.Column(float, pa.Check.between(0, 400), nullable=True),
            "density": pa.Column(float, pa.Check.between(0, 1.5), nullable=True),
            "pH": pa.Column(float, pa.Check.between(0, 5), nullable=True),
            "sulphates": pa.Column(float, pa.Check.between(0, 2.5), nullable=True),
            "alcohol": pa.Column(float, pa.Check.between(9, 15), nullable=True),
            "quality": pa.Column(float, pa.Check.between(1, 10), nullable=True),
        },
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
        ],
        drop_invalid_rows=True,
    )

    # Validate and clean the data
    print("Validating and cleaning data through pandera...")
    try:
        clean_data = schema.validate(raw_data, lazy=True).drop_duplicates().dropna(how="all")
        print("Validation successful.")
    except pa.errors.SchemaErrors as e:
        print("Validation failed. Errors:")
        print(e.failure_cases)
        return

    # Save cleaned data
    output_file = os.path.join(processed_data_path, "cleaned_wine_quality.csv")
    clean_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    print(f"Data validation step 1 is done.")


if __name__ == "__main__":
    validate_raw_data()
