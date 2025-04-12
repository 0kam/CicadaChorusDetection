setwd("~/CicadaChorusDetection/step3_model_training/")
library(tidyverse)

ds <- c(
  "mlruns/769776563097437127/",
  "mlruns/720817091353757188/",
  "mlruns/107176719986165299/"
  )

df <- map(ds, function(d) {list.files(d, recursive = T, full.names = T)}) %>%
  unlist() %>%
  str_subset("model_name") %>%
  tibble(
    run = .,
  ) %>%
  mutate(
    model_name = map_chr(run, function(f) {
      read_lines(f)
    })) %>%
  mutate(
    run_name = map_chr(run, function(f) {
      name <- try(
        f %>%
          str_replace("params.*", "tags/mlflow.runName") %>%
          read_file()
      )
      if (class(name) == "try-error") {name <- 0}
      return(name)
    })) %>%
  mutate(
    f1_mean = map_dbl(run, function(f) {
      f1 <- try(
        f %>%
          str_replace("params.*", "metrics/real_f1_macro") %>%
          read_delim(col_names = F) %>%
          pull(X2)
      )
      if (class(f1) == "try-error") {f1 <- 0}
      return(f1)
    })) %>%
  mutate(
    sr = map_dbl(run, function(f) {
      val <- try(
        f %>%
          str_replace("params.*", "params/dataset.sr") %>%
          read_file() %>%
          as.numeric
      )
      if (class(val) == "try-error") {val <- 0}
      return(val)
    })) %>%
  mutate(
    win_size = map_dbl(run, function(f) {
      val <- try(
        f %>%
          str_replace("params.*", "params/dataset.win_sec") %>%
          read_file() %>%
          as.numeric
      )
      if (class(val) == "try-error") {val <- 0}
      return(val)
    })) %>%
  select(-run) %>%
  filter(f1_mean != 0)
  
df %>%
  mutate(win_size = as.factor(win_size)) %>%
  mutate(
    model_name = model_name %>%
      str_replace_all("_", "") %>%
      str_replace("efficientnet", "EfficientNet") %>%
      str_replace("resnet", "ResNet") %>%
      str_replace("Netb", "NetB") %>%
      str_replace("Netv2s", "NetV2S") %>%
      str_replace("Netv2m", "NetV2M")
  ) %>%
  mutate(sr = sr / 1000) %>%
  ggplot(aes(x = sr, y = f1_mean, color = win_size)) +
  geom_point(size = 3) +
  geom_line() +
  facet_wrap(model_name~.) +
  labs(
    x = "Sampling rate (kHz)",
    y = "F1 mean in tune dataset",
    color = "Window size (s)"
  ) +
  theme_minimal()+
  scale_x_continuous(breaks = c(16, 24, 36, 48)) +
  theme(
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    strip.text = element_text(size = 12)
  ) -> p

p

ggsave("model_selection.png", p, width = 8, height = 5)

