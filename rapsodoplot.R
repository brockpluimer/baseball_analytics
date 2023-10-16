R# Loading required libraries
library(ggplot2)
library(dplyr)

# Sample dataset
data <- read.csv("pitch_movement.csv", stringsAsFactors = FALSE)

# Hard-coded team and player names
team <- "Marlins"
selected_pitcher <- "Alcantara, Sandy"

# Filter data based on the selected team and player
pitcher_data <- data %>% 
  filter(team_name == team, `name` == selected_pitcher) %>%
  arrange(pitch_type_name, pitcher_break_x) %>%
  mutate(pitcher_break_z = pitcher_break_z * -1)  # Inverse the horizontal movement


# Define color settings
colors <- c("4-Seamer" = "red", "Slider" = "blue", "Changeup" = "green", 
            "Curveball" = "purple", "Sinker" = "brown", "Slurve" = "pink", "Sweeper" = "orange", "Cutter" = "gold")

# Calculate the convex hull for the selected pitcher's data
hull_indices <- chull(pitcher_data$pitcher_break_x, pitcher_data$pitcher_break_z)
hull_data <- pitcher_data[hull_indices, ]

# Plot
plot <- ggplot(pitcher_data, aes(x = pitcher_break_x, y = pitcher_break_z)) +
  geom_polygon(data = hull_data, aes(x = pitcher_break_x, y = pitcher_break_z), fill = "grey", alpha = 0.6) + 
  geom_point(aes(color = pitch_type_name), size = 4) +
  coord_fixed(ratio = 1, xlim = c(-30, 30), ylim = c(-80, 0)) +  # Fixed axis range
  scale_color_manual(values = colors) +
  labs(title = paste("Rapsodo Plot for", selected_pitcher),
       x = "Horizontal Movement (inches)", y = "Vertical Movement (inches)") +
  theme_minimal() +
  theme(legend.position = "right", plot.background = element_rect(fill = "white"))

# Display plot
print(plot)

# Save as PNG
ggsave(paste0(selected_pitcher, "_", team, "_Rapsodo_Plot.png"), plot, width = 7, height = 7)
