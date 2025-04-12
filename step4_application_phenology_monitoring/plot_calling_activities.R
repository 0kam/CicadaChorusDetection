library(tidyverse)
library(lubridate)
library(hms)
library(slider)

setwd("~/CicadaChorusDetection/step4_application_phenology_monitoring/")

plot_spiral <- function(csv_path, th) {
  site_name <- str_remove(basename(csv_path), ".csv")
  df <- read_csv(csv_path) %>%
    rename(
      `Walker's cicada` = tsukutsukuboushi,
      `Kaempfer cicada` = niiniizemi,
      `Robust cicada` = minminzemi,
      `Evening cicada` = higurashi,
      `Large brown cicada` = aburazemi
    ) %>%
    mutate(
      datetime = str_extract(file_name, "\\d{8}_\\d{4}") %>%
        ymd_hm()
    ) %>%
    select(-file_name) %>%
    mutate(datetime = floor_date(datetime, unit = "minutes"))%>%
    group_by(datetime) %>%
    summarise(across(everything(), sum)) %>%
    pivot_longer(cols = -datetime, names_to = "species", values_to = "is_calling") %>%
    mutate(is_calling = (is_calling >= th)) %>%
    mutate(
      month = month(datetime),
      date = date(datetime),
      hour = hour(datetime),
      time = as_hms(datetime)
    ) %>%
    filter(month < 10) %>%
    mutate(is_calling = ifelse(hour > 4, is_calling, FALSE)) %>%
    mutate(is_calling = ifelse(hour < 20, is_calling, FALSE))
  
  species <- df %>%
    pull(species) %>%
    unique()
  
  p <- df %>%
    group_by(species, hour, date) %>%
    summarise(freq = mean(as.integer(is_calling))) %>%
    ungroup() %>%
    ggplot(aes(x = hour, y = date, fill = species, alpha = freq)) +
    geom_tile() +
    scale_x_continuous(
      breaks = 0:7 * 3, minor_breaks = 0:23,
      labels = str_c(0:7 * 3, ":00")
    ) +
    scale_y_date(
      date_breaks = "2 weeks"
    ) +

    labs(
      x = "",
      y = "",
      alpha = "Frequency",
      fill = "Species"
    ) +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 18),
      legend.text = element_text(size = 18),
      legend.title = element_text(size = 22),
      strip.text = element_text(size = 22)
    ) +
    coord_polar(start = -360 / 48 * pi / 180) + # 30分左に回すと真上が0時になる
    facet_wrap(~species)
  
  out_d <- dirname(csv_path)
  ggsave(str_c(out_d, "/", site_name, "_spiral_en.png"), p, width = 20, height = 12)
  p
}

csv_path <- "Akitsu.csv"
plot_spiral(csv_path, th = 0.75)
