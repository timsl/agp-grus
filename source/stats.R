install.packages(c("tidyverse", "broom", "magrittr", "ggplot2"))
install.packages("tidyverse")

library("tidyverse")
library("broom")
library("magrittr")

data <- read_csv("log")

## The held is 0 everywhere, pointless but included for completeness
data %>%
    summarize(min(held), max(held), mean(held))

## View means and sd separately with running particles and no
data %>%
    group_by(updating) %>%
    summarize(mean(update), sd(update),
              mean(display), sd(display),
              mean(all), sd(all))

updating <- data %>%
    filter(updating==1) %>%
    select(-updating, -held)
nonupdating <- data %>%
    filter(updating==0) %>%
    select(-updating, -held)

## A single measurement without update held
nonupdating %>%
    filter(abs(update - median(update)) > sd(update))
nonupdating %>%
    summarize(min(update), max(update), mean(update), sd(update))
