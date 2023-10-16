Strikeorball.py README

Description
Strikeorball.py uses over 140,000 pitch datapoints to generate a Random Forest Classifier that predicts whether or not a User-designed pitch will be called a ball or strike. The input data is stored in sample-pitch-data.csv which must be in the same location as the script in order to run. The script outputs a ranked feature importance plot indicating the most influential factors related to ball/strike outcomes.

How to Run
Ensure that the file “sample-pitch-dat.csv” is located in the same directory as the script. Run the script in your terminal: python3 strikeorball.py. After the model is trained, the User will be asked to input values for various predictors to determine if a User designed pitch would be called a strike or a ball. The User can input either 1) custom values 2) random values within the range of the predictor by inputting “r” or 3) the median predictor value by leaving the value blank.

Requirements
* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
Features
1. Data Cleaning: Deletes rows with more than 10 consecutive NULL values, fills missing values with column mean.
2. Data Split: Splits the data into training and test sets.
3. Model Training: Trains a Random Forest Classifier.
4. Model Evaluation: Computes accuracy, classification report, confusion matrix, and cross-validation scores.
5. ROC Curve: Plots an ROC curve for the model.
6. Feature Importance: Displays the importance of each feature.
7. Custom Prediction: Allows the user to input custom values or random values for predictors.
Input Data
CSV file named sample-pitch-dat.csv containing:
* IsLefthandedPitcher
* IsLefthandedBatter
* inning
* balls
* strikes
* pitchtypeid
* ReleaseSpeed
* PlateX
* PlateZ
* ReleaseX
* ReleaaseZ
* Extension
* SpinRate
* SpinDirection
* X0
* Y0
* Z0
* VerticalBreak
* InducedVertBreak
Output
* Confusion matrix plot saved as confusion_matrix.png
* ROC curve plot saved as strikeorball_roc_curve.png
* A printed list of ranked feature importance
