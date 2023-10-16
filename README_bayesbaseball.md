Bayesbaseball.R README
Bayesbaseball.R uses Statcast data from 2022, 2021, and 2019 to predict a baseball player's On-Base Percentage (OBP) based on their batting average (AVG) and walk rate per plate appearance (BB/PA). Inversely, it can predict AVG and BB/PA based on a User inputted OBP. 

How to Run
Place the script in the same folder as hittingtigers.csv. 
Run the script with the desired argument: 
To predict On-base Percentage                 : Rscript bayesbaseball.R --mode OBP
To predict Batting Average and Walk Rate: Rscript bayesbaseball.R --mode AVG_BB_PA 
Utility:
1. Prediction: Accurately predict OBP based on AVG and BB/PA or vice-versa.
2. Player Analysis: Understand the key factors contributing to a player’s OBP.
3. Strategy Planning: Set achievable goals for players based on their predicted OBP.
4. Talent Scouting: Evaluate new talents using quantitative metrics.
5. Command Line Flexibility: Choose prediction mode directly from the command line.
Requirements:
* R environment
* brms package for Bayesian regression
* dplyr package for data manipulation
* argparser package for command line argument parsing
Features:
1. Data Standardization: Centers and scales AVG and BB/PA.
2. Bayesian Modeling: Uses Bayesian regression to model OBP.
3. Prediction: Offers two prediction modes, for OBP or AVG and BB/PA.
4. Evaluation Metrics: Calculates MAE, MSE, and R-squared to evaluate model performance.
5. User Interaction: Accepts user input for AVG, BB/PA, or OBP based on the prediction mode selected.
Input Data:
A CSV file named hittingtigers.csv with the following columns is required:
* AVG
* BB_PA
* OBP
Output:
* Predicted OBP, AVG, or BB/PA based on user choice.
* Evaluation metrics like MAE, MSE, and R-squared are displayed on the console.

