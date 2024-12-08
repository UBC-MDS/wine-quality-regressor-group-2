# Analysis of Wine Quality and Prediction Using Logistic Regression

## Author:
Alix, Paramveer, Susannah, Zoe

## About
This analysis investigates the relationship between physicochemical properties and wine quality using the Wine Quality dataset from the UCI Machine Learning Repository, containing data for both red and white wine. Through comprehensive exploratory data analysis, we examined 11 physicochemical features and their correlations with wine quality scores. Our analysis revealed that higher quality wines typically have higher alcohol content and lower volatile acidity, with white wines generally receiving higher quality scores than red wines. Most features showed right-skewed distributions with notable outliers, particularly in sulfur dioxide and residual sugar measurements. The quality scores themselves followed a normal distribution centered around scores 5-6.

We implemented a logistic regression model with standardized features and one-hot encoded categorical variables, using randomized search cross-validation to optimize the regularization parameter. The final model achieved an accuracy of 52.4% on the test set. While this performance suggests room for improvement, the analysis provides valuable insights for future research directions.

The dataset used in this project is the Wine Quality dataset from the UCI Machine Learning Repository (Cortez et al. 2009) and can be found [here](https://archive.ics.uci.edu/dataset/186/wine+quality.) These datasets are related to red and white variants of the Portuguese “Vinho Verde” wine. They contains physicochemical properties (e.g., acidity, sugar content, and alcohol) of different wine samples, alongside a sensory score representing the quality of the wine, rated by experts on a scale from 0 to 10. Each row in the dataset represents a wine sample, with the columns detailing 11 physicochemical attributes and the quality score. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones).

Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

## Report
The final report can be found here: *PUT LINK HERE*

## Dependencies

- `conda` (version 24.9.1 or higher)
- `conda-lock` (version 2.5.7 or higher)
- Python package `ucimlrepo` (version 0.0.7)
- `jupyterlab` (version 4.2.0 or higher)
- `nb_conda_kernels` (version 2.5.1 or higher)
- Python and packages listed in [`environment.yml`](https://github.com/UBC-MDS/wine-quality-regressor-group-2/blob/main/environment.yml)

## Usage

### Setup

> If you are using Windows or Mac, then please ensure that Docker Desktop is running. The user can be check if they have Docker by running the following command in a bash terminal: `docker --version`.

1. Clone this GitHub repository.
2. Make sure `docker-compose.yml` is using the image with the tag you wish to run it with. No changes are necessary if there is not a specific image tag you would like to run.

### Running the analysis

1. Run the following command in a terminal in the root of the local repository to use the Docker image to run the analysis:

    ```bash
    docker compose up
    ```

    This command will automatically start up a Jupyter Lab session using the image listed in the `docker-compose.yml` file and mount the current project in the Docker container.

2. In the terminal, look for the Jupyter Lab link which starts with `http://127.0.0.1:8888/`. Copy and paste the URL into the browser to open up Jupyter Lab.

3. To run the analysis, enter the following commands in the terminal in the project root:

```
# script1: download data
python /scripts/data_download.py \
    --id=186 \
    --raw_data_out=data/raw

# script2: raw data validation
python /scripts/validate_raw_data.py \
    --input-path=data/raw/wine_quality.csv \
    --processed-data-path=data/processed \
    --seed=522

# script3: training data validation
python ./scripts/validate_training_data.py \
    --input-path=data/processed/training_set.csv \
    --output-path=results

#script4: read data
python ./scripts/read_data.py \
    --raw_data_path=data/raw/wine_quality.csv \
    --processed_data_path=data/processed \
    --seed=522 \
    --test_size=0.2

# script4: EDA
python ./scripts/eda.py \
    --input-data=data/processed/training_set.csv \
    --output-dir=results/figures

# script5: model and result
python ./scripts/model_and_results.py \
    --training-data=data/processed/training_set.csv \
    --test-data=data/processed/test_set.csv \
    --seed=522 \
    --results-to=results/tables

# build HTML report and copy build to docs folder
jupyter-book build report
cp -r report/_build/html/* docs
```

### Clean up

Hit `Ctrl + C` in the terminal to end the Jupyter Lab session. Run the following command after the session ends to free up the resources used by Docker: `docker compose rm`.

### Developer notes

Please see CONTRIBUTING.md.

### License

Please see License.

### References

<div id="refs" class="references hanging-indent">

<div id="ref-Jain2023">
Jain, K., Kaushik, K., Gupta, S. K., & Others. 2023. "Machine learning-based predictive modelling for the enhancement of wine quality." *Scientific Reports*, 13:17042. <https://doi.org/10.1038/s41598-023-44111-9>.
</div>

<div id="ref-Cortez2009">
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. 2009. "Wine Quality [Dataset]." *UCI Machine Learning Repository*. <https://doi.org/10.24432/C56S3T>.
</div>

<div id="ref-Kniazieva2023">
Kniazieva, Y. 2023, October 12. "A digital sommelier: Machine learning for wine quality prediction." *Label Your Data*. <https://labelyourdata.com/articles/machine-learning-for-wine-quality-prediction>.
</div>

<div id="ref-Aich2018">
Aich, S., Al-Absi, A. A., Hui, K. L., Lee, J. T., & Sain, M. 2018. "A classification approach with different feature sets to predict the quality of different types of wine using machine learning techniques." In *International Conference on Advanced Communication Technology (ICACT)*, pp. 139–143. <https://doi.org/10.23919/ICACT.2018.8323674>.
</div>

</div>