setwd("~/CicadaChorusDetection/step3_model_training/")
library(tidyverse)

d <- "mlruns/940900156616882691/"
df <- d %>% 
  list.files(recursive = T, full.names = T) %>%
  str_subset("generation.cicadas.popsize.max") %>%
  tibble(
    run = .,
  ) %>%
  mutate(
    popsize = map_int(run, function(f) {
      read_lines(f) %>%
        as.integer
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
          str_replace("params.*", "metrics/test_f1_macro") %>%
          read_delim(col_names = F) %>%
          pull(X2)
      )
      if (class(f1) == "try-error") {f1 <- 0}
      return(f1)
    })) %>%
  select(-run) %>%
  filter(str_detect(run_name, "202504")) %>%
  filter(!(run_name %in% c("20250405172803", "20250405222050"))) %>%
  filter(f1_mean != 0)

df %>%
  ggplot(aes(x = popsize, y = f1_mean)) +
  geom_point(size = 3) +
  labs(
    x = "Max. num. of cicada calls in each simulation",
    y = "F1 mean in test dataset"
  ) +
  scale_x_continuous(breaks = c(1, 2, 5, 10, 20, 40)) +
  theme_minimal()+
  theme(
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    panel.grid.minor.x = element_blank()
  ) -> p

p

ggsave("popsize_effect.png", p, width = 6, height = 3)
