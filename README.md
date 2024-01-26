# olympic-predictions
Using data from the 1948 - 2020 Olympic Games, I built time series models to predict the winning times for all individual swimming events at the upcoming 2024 competition. Challenges included having limited data and the impact of COVID-19 on the results of the 2020 competition. Based on my models, we can expect a fast summer of racing, with predictions indicating that 5 world records are likely to be broken across the events.

Files included:
2020_mae_olympicproject.py: applying models on the validation set (2020) to determine which model to use for the 2024 prediction (damped or not damped)
2020predictions_final.csv: table containing 2020 predictions, MAE, and final models to be used on the test data (2024)
full_olympic_project.py: complete code used for the project. Some code blocks need to be run more than once, when to run which piece is commented in the code.
final_results_allcolumns.xlsx: table comparing all results on validation/test from the project. Also contains final predictions and world records to be broken.
example_plots.py: code to build line graphs/forecasts for select events. Can be adapted to produce graphs for other events.
Side Project Poster: summary of project work and results
