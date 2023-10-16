library(brms)
library(dplyr)
library(argparser)

# Function to predict OBP
predict_OBP <- function(trace, avg_std, bb_pa_std) {
  pred_mean <- mean(trace$b_Intercept) + mean(trace$b_AVG_std) * avg_std + mean(trace$b_BB_PA_std) * bb_pa_std
  return(pred_mean)
}

# Function to predict AVG and BB_PA
predict_AVG_BB_PA <- function(trace, obp_target, data) {
  avg_range <- seq(min(data$AVG), max(data$AVG), length.out = 100)
  bb_pa_range <- seq(min(data$BB_PA), max(data$BB_PA), length.out = 100)
  
  best_avg <- 0
  best_bb_pa <- 0
  min_diff <- Inf
  
  for (avg in avg_range) {
    for (bb_pa in bb_pa_range) {
      avg_std <- (avg - mean(data$AVG)) / sd(data$AVG)
      bb_pa_std <- (bb_pa - mean(data$BB_PA)) / sd(data$BB_PA)
      
      predicted_obp <- predict_OBP(trace, avg_std, bb_pa_std)
      diff <- abs(predicted_obp - obp_target)
      
      if (diff < min_diff) {
        min_diff <- diff
        best_avg <- avg
        best_bb_pa <- bb_pa
      }
    }
  }
  return(c(best_avg, best_bb_pa))
}

# Command line argument parsing
arg_parser <- arg_parser("Select prediction mode: OBP or AVG_BB_PA") %>%
  add_argument("--mode", type="character", help='Mode: OBP or AVG_BB_PA')
args <- parse_args(arg_parser)

# Load the data
data <- read.csv("hittingstats.csv")

# Standardize input features
data <- data %>%
  mutate(AVG_std = (AVG - mean(AVG)) / sd(AVG),
         BB_PA_std = (BB_PA - mean(BB_PA)) / sd(BB_PA))

# Fit the Bayesian model
fit <- brm(OBP ~ AVG_std + BB_PA_std, data = data, prior = c(
  prior(normal(0, 0.1), class = b),
  prior(normal(0, 0.1), class = Intercept),
  prior(exponential(0.1), class = sigma)),
  iter = 3000, chains = 4, control = list(adapt_delta = 0.95))

# Get trace
trace <- posterior_samples(fit)

# Calculate evaluation metrics
test_data <- data
test_data$pred_OBP <- apply(test_data, 1, function(row) predict_OBP(trace, as.numeric(row["AVG_std"]), as.numeric(row["BB_PA_std"])))
mae <- mean(abs(test_data$pred_OBP - test_data$OBP))
mse <- mean((test_data$pred_OBP - test_data$OBP)^2)
r_squared <- 1 - (sum((test_data$pred_OBP - test_data$OBP)^2) / sum((test_data$OBP - mean(test_data$OBP))^2))

cat("Mean Absolute Error (MAE):", round(mae, 3), "\n")
cat("Mean Squared Error (MSE):", round(mse, 3), "\n")
cat("Coefficient of Determination (R-squared):", round(r_squared, 3), "\n")

if (args$mode == 'OBP') {
  cat("Enter AVG: ")
  avg <- as.numeric(readLines(con = "stdin", n = 1))
  cat("Enter BB/PA: ")
  bb_pa <- as.numeric(readLines(con = "stdin", n = 1))
  
  avg_std <- (avg - mean(data$AVG)) / sd(data$AVG)
  bb_pa_std <- (bb_pa - mean(data$BB_PA)) / sd(data$BB_PA)
  
  predicted_OBP <- predict_OBP(trace, avg_std, bb_pa_std)
  cat("Predicted OBP for AVG=", avg, " and BB/PA=", bb_pa, ": ", round(predicted_OBP, 3), "\n")
} else if (args$mode == 'AVG_BB_PA') {
  cat("Enter desired OBP: ")
  obp <- as.numeric(readLines(con = "stdin", n = 1))
  predicted_AVG_BB_PA <- predict_AVG_BB_PA(trace, obp, data)
  cat("Predicted AVG and BB/PA for OBP=", obp, ": ", round(predicted_AVG_BB_PA, 3), "\n")
}
