# Split cicada song data into training and testing sets
# Test ratio: 0.2

library(tidyverse)
setwd("~/CicadaChorusDetection/step2_prepare_training_sources/data/cicada_song")

set.seed(123)
train_test_split <- function(df) {
  test_ratio = 0.2
  test_num <- ceiling(test_ratio * nrow(df))
  dir.create(str_c("train/", df$sp[[1]]))
  dir.create(str_c("test/", df$sp[[1]]))
  df <- df %>%
    sample_frac(1L) %>%
    mutate(train = test_num < row_number())
  df_train <- df %>%
    filter(train) %>%
    mutate(fname_to = str_replace(fname, "segments_preprocessed", "train"))
  df_test <- df %>%
    filter(!train) %>%
    mutate(fname_to = str_replace(fname, "segments_preprocessed", "test"))
  file.copy(df_train$fname, df_train$fname_to)
  file.copy(df_test$fname, df_test$fname_to)
}

tibble(
    fname = list.files("segments_preprocessed/", pattern = ".wav", full.names = T, recursive = T)
) %>%
  mutate(
    sp = str_extract(fname, "//.*/") %>%
      str_remove_all("/")
  ) %>%
  group_by(sp) %>%
  group_split() %>%
  map(train_test_split)
