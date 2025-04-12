setwd("~/CicadaChorusDetection/step3_model_training/")
library(tidyverse)

read_csv("simulation_tuning_results.csv") %>%
  group_by(model_path, dataset) %>%
  summarise(
    f1_macro = mean(f1),
  ) %>%
  pivot_wider(id_cols = model_path, names_from = dataset, values_from = f1_macro) %>%
  arrange(desc(tune)) %>%View()

read_csv("simulation_tuning_results_soorted.csv") %>%
  View()
