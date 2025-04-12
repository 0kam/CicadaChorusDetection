setwd("~/CicadaChorusDetection/step1_real_annotation/")
library(tidyverse)
library(lubridate)

# First, we sample 200 audio files from each of 3 datasets to use in model testing
# ------------------------------------------
# Sample data from Akitsu-no-ike
akitsu <- tibble(
  fname = list.files("~/TrueNAS/nies_audio_recording/2024/ChirpArray/Akitsu/", 
                     full.names = TRUE)) %>%
  mutate(datetime = str_extract(fname, "\\d{8}_\\d{4}") %>%ymd_hm()) %>%
  mutate(date = date(datetime)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(file_size = file.size(fname))


set.seed(123)
akitsu %>%
  filter(date >= ymd("2024-07-15")) %>%
  filter(date <= ymd("2024-09-15")) %>%
  filter(hour >= 5) %>%
  filter(hour <= 18) %>%
  filter(file_size > 111300000) %>%
  sample_n(200) %>%
  arrange(datetime) %>%
  write_csv("sample_akitsu_1.csv")

# Sample data from Ryokuchi
ryokuchi <- tibble(
  fname = list.files("~/TrueNAS/nies_audio_recording/2024/ChirpArray/Ryokuchi/", 
                     full.names = TRUE)) %>%
  mutate(datetime = str_extract(fname, "\\d{8}_\\d{4}") %>%ymd_hm()) %>%
  mutate(date = date(datetime)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(file_size = file.size(fname))


set.seed(123)
ryokuchi %>%
  filter(date >= ymd("2024-07-15")) %>%
  filter(date <= ymd("2024-09-15")) %>%
  filter(hour >= 5) %>%
  filter(hour <= 18) %>%
  filter(file_size > 111300000) %>% # Remove files with recording errors
  sample_n(200) %>%
  arrange(datetime) %>%
  write_csv("sample_ryokuchi_1.csv")

# Sample data from Suiri
suiri <- tibble(
  fname = list.files("~/HDD10TB/cicadasong2022/records/NE03/", "*.wav", 
                     full.names = TRUE, recursive = T)) %>%
  mutate(datetime = str_extract(fname, "\\d{8}_\\d{2}_\\d{2}") %>%ymd_hm()) %>%
  mutate(date = date(datetime)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(file_size = file.size(fname))


set.seed(123)
suiri %>%
  filter(date >= ymd("2022-07-15")) %>%
  filter(date <= ymd("2022-09-15")) %>%
  filter(hour >= 5) %>%
  filter(hour <= 18) %>%
  filter(file_size > 28800000) %>%
  sample_n(200) %>%
  arrange(datetime) %>%
  write_csv("sample_suiri_1.csv")


# Next, we sample 50 audio files (not included in the testing dataset)
# from each of 3 datasets to use in simulation tuning
# ------------------------------------------
# Sample data from Akitsu-no-ike
akitsu <- tibble(
  fname = list.files("~/TrueNAS/nies_audio_recording/2024/ChirpArray/Akitsu/", 
                     full.names = TRUE)) %>%
  mutate(datetime = str_extract(fname, "\\d{8}_\\d{4}") %>%ymd_hm()) %>%
  mutate(date = date(datetime)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(file_size = file.size(fname))

akitsu_test <- read_csv("sample_akitsu_1.csv")
"%not.in%" <- Negate("%in%")

set.seed(123)
akitsu %>%
  filter(date >= ymd("2024-07-15")) %>%
  filter(date <= ymd("2024-09-15")) %>%
  filter(hour >= 5) %>%
  filter(hour <= 18) %>%
  filter(file_size > 111300000) %>%
  filter(fname %not.in% akitsu_test$fname) %>%
  sample_n(50) %>%
  arrange(datetime) %>%
  write_csv("sample_akitsu_2.csv")

# Sample data from Ryokuchi
ryokuchi <- tibble(
  fname = list.files("~/TrueNAS/nies_audio_recording/2024/ChirpArray/Ryokuchi/", 
                     full.names = TRUE)) %>%
  mutate(datetime = str_extract(fname, "\\d{8}_\\d{4}") %>%ymd_hm()) %>%
  mutate(date = date(datetime)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(file_size = file.size(fname))

ryokuchi_test <- read_csv("sample_ryokuchi_1.csv")

set.seed(123)
ryokuchi %>%
  filter(date >= ymd("2024-07-15")) %>%
  filter(date <= ymd("2024-09-15")) %>%
  filter(hour >= 5) %>%
  filter(hour <= 18) %>%
  filter(file_size > 111300000) %>% # Remove files with recording errors
  filter(fname %not.in% ryokuchi_test$fname) %>%
  sample_n(50) %>%
  arrange(datetime) %>%
  write_csv("sample_ryokuchi_2.csv")

# Sample data from Suiri
suiri <- tibble(
  fname = list.files("~/HDD10TB/cicadasong2022/records/NE03/", "*.wav", 
                     full.names = TRUE, recursive = T)) %>%
  mutate(datetime = str_extract(fname, "\\d{8}_\\d{2}_\\d{2}") %>%ymd_hm()) %>%
  mutate(date = date(datetime)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(file_size = file.size(fname))

suiri_test <- read_csv("sample_suiri_1.csv")

set.seed(123)
suiri %>%
  filter(date >= ymd("2022-07-15")) %>%
  filter(date <= ymd("2022-09-15")) %>%
  filter(hour >= 5) %>%
  filter(hour <= 18) %>%
  filter(file_size > 28800000) %>%
  filter(fname %not.in% suiri_test$fname) %>%
  sample_n(50) %>%
  arrange(datetime) %>%
  write_csv("sample_suiri_2.csv")
