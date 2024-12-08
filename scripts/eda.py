# Run by following command: python scripts/script_eda.py ./data/train_data.csv ./scripts/output/
## maybe different directory

import pandas as pd
import altair as alt
import altair_ally as aly
import os
import click

@click.command()
@click.argument("input_data", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def eda(input_data, output_dir):
    """
    Perform EDA on training data and save the results.

    INPUT_DATA: Path to the input dataset (CSV file).
    OUTPUT_DIR: Directory to save the results (charts and analysis).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    train_df = pd.read_csv(input_data)

    # 1.EDA

    ## 1.1 Distribution of quality scores across numerical features
    aly.alt.data_transformers.enable('vegafusion')
    aly_dist_chart = aly.dist(train_df, color='quality')
    aly_dist_chart_path = os.path.join(output_dir, "dist_wine_scores_by_feature.png")
    aly_dist_chart.save(aly_dist_chart_path)
    print(f"1.1 Distribution chart saved to {aly_dist_chart_path}")

    ## 1.2 Distribution of quality scores by categorical feature (wine color)
    ### Calculate the proportions of each quality score for different wine colors
    proportions = (train_df.groupby(['color', 'quality'])
                  .size()
                  .reset_index(name='count')
                  .assign(proportion=lambda x: x.groupby('color')['count'].transform(lambda y: y / y.sum()))
                  .reset_index(drop=True))

    ### Create a line plot showing the proportions of each quality score for different wine colors
    line_chart = alt.Chart(proportions).mark_line(
        interpolate='monotone',
        point=True,
        tension=0.7,
        strokeWidth=2
    ).encode(
        x=alt.X('quality:Q',
                title='Quality Score',
                scale=alt.Scale(domain=[2.5, 9.5])),
        y=alt.Y('proportion:Q',
                title='Proportion',
                axis=alt.Axis(format='.0%')),
        color=alt.Color('color:N',
                       title='Wine Type',
                       scale=alt.Scale(domain=['red', 'white'],
                                     range=['#1f77b4', '#ff7f0e'])),
        tooltip=[
            alt.Tooltip('quality:Q', title='Quality'),
            alt.Tooltip('proportion:Q', title='Proportion', format='.1%'),
            alt.Tooltip('color:N', title='Wine Type')
        ]
    ).properties(
        width=500,
        height=300
    )

    line_chart_path = os.path.join(output_dir, "density_red_vs_white.png")
    line_chart.save(line_chart_path)
    print(f"1.2 Line chart saved to {line_chart_path}")

    ## 1.3 Correlation matrix
    aly.corr(train_df)
    ### Create scatter plot with regression line
    scatter_chart = alt.Chart(train_df[['free_sulfur_dioxide', 'total_sulfur_dioxide']].sample(600)).mark_circle().encode(
        x='free_sulfur_dioxide',
        y='total_sulfur_dioxide'
    ).properties(
        width=300,
        height=200
    ) + alt.Chart(
        train_df[['free_sulfur_dioxide', 'total_sulfur_dioxide']].sample(600)
    ).mark_line(color='red').encode(
        x='free_sulfur_dioxide',
        y='total_sulfur_dioxide'
    ).transform_regression(
        'free_sulfur_dioxide',
        'total_sulfur_dioxide'
    )

    scatter_chart_path = os.path.join(output_dir, "feature_corrs.png")
    scatter_chart.save(scatter_chart_path)
    print(f"1.3 Scatter correlation chart saved to {scatter_chart_path}")

    ## 1.4 Outlier detection
    ### Get numerical columns only (exclude 'quality' and 'color')
    numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'quality']

    ### Create box plots
    charts = []
    for col in numerical_cols:
        chart = alt.Chart(train_df).mark_boxplot().encode(
            x=alt.X(col + ':Q', scale=alt.Scale(zero=False)),
            y=alt.Y('color:N', title=None),  # keep color but add title
            color=alt.Color('color:N', legend=alt.Legend(title="Wine Type"))
        ).properties(
            title=col,
            width=250,
            height=80
        )
        charts.append(chart)

    ### Display all the box plots together
    n_cols = 3
    grid = alt.vconcat(*[alt.hconcat(*charts[i:i + n_cols]) for i in range(0, len(charts), n_cols)])
    grid_path = os.path.join(output_dir, "red_vs_white_all_features.png")
    grid.save(grid_path)
    print(f"1.4 Outlier boxplots saved to {grid_path}")

    ## 1.5 The distribution of the target variable (quality)
    ### Create a DataFrame with the quality counts
    quality_df = pd.DataFrame({
        'quality': train_df['quality'].value_counts().index,
        'count': train_df['quality'].value_counts().values
    })
    quality_df['percentage'] = (quality_df['count'] / len(train_df) * 100).round(1)
    quality_df = quality_df.sort_values('quality')

    ### Create the bar plot
    bar_chart = alt.Chart(quality_df).mark_bar().encode(
        x=alt.X('quality:O', title='Quality Score'),
        y=alt.Y('percentage:Q', title='Percentage (%)')
    ).properties(
        width=350,
        height=200,
        title='Distribution of Wine Quality Scores'
    )

    bar_chart_path = os.path.join(output_dir, "dist_wine_scores.png")
    bar_chart.save(bar_chart_path)
    print(f"1.5 Bar chart saved to {bar_chart_path}")


if __name__ == "__main__":
    eda()
