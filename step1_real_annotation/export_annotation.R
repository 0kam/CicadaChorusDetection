setwd("~/CicadaChorusDetection/step1_real_annotation/")
library(tidyverse)

bind_rows(
  read_csv("tune_dataset.csv"),
  read_csv("test_dataset.csv")
) %>%
  rename(
    lb = aburazemi,
    ev = higurashi,
    rb = minminzemi,
    kf = niiniizemi,
    wk = tsukutsukuboushi
  ) %>%
  mutate(path = str_remove(path, "/home/okamoto/CicadaChorusDetection/step1_real_annotation/")) %>%
  write_csv("cicadas_in_nies/annotation.csv")
