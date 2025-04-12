setwd("~/CicadaChorusDetection/step1_real_annotation/")
library(tidyverse)
library(ggupset)

df <- bind_rows(
  read_csv("tune_dataset.csv") %>%
    mutate(dataset = "Tune Dataset"),
  read_csv("test_dataset.csv") %>%
    mutate(dataset = "Test Dataset")
  ) %>%
  rename(
    "Large Brown Cicada" = aburazemi,
    "Evening Cicada" = higurashi,
    "Robust Cicada" = minminzemi,
    "Kaempfer Cicada" = niiniizemi,
    "Walker's Cicada" = tsukutsukuboushi
  ) %>%
  pivot_longer(
    cols = -c(path, start, end, dataset),
    names_to = "species",
    values_to = "presense"
  )

df %>%
  filter(presense == 1) %>%
  select(-presense) %>%
  group_by(path, start, end, dataset) %>%
  summarise(species = list(species)) %>%
  ggplot(aes(x = species)) +
  geom_bar() +
  scale_x_upset(n_intersections = 50, scale_name = "t") +
  facet_grid(dataset~., scales = "free") +
  theme_minimal() +
  labs(
    y = "Count",
    x = "Species composition"
  ) +
  theme(
    axis.title = element_text(size = 18),
    axis.text.y = element_text(size = 14),
    strip.text = element_text(size = 18) 
  ) -> p

ggsave("real_dataset_upset.png", width = 8, height = 6)

df %>%
  filter(presense == 1) %>%
  group_by(dataset, species) %>%
  ggplot(aes(x = 1, fill = species)) +
  geom_bar(stat = "count", position = "fill") +
  geom_text(stat = "count", aes(label = ..count..),
            position = position_fill(vjust = 0.5),
            size = 6) +
  coord_polar(theta = "y") +
  theme_minimal() +
  labs(
    fill = "Species",
    y = "Count"
  ) +
  theme(
    axis.title.y = element_blank(),
    axis.title.x = element_text(size = 18),
    axis.text = element_blank(),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 14),
    strip.text = element_blank(),# element_text(size = 18),
    panel.grid = element_blank()
  ) +
  facet_grid(dataset~.) -> p2

ggsave("real_dataset_pie.png", p2, width = 6, height = 8)

df %>%
  group_by(path, start) %>%
  summarise(presense = sum(presense)) %>%
  ungroup() %>%
  group_by(presense) %>%
  summarise(n = n() / 2250 * 100)
