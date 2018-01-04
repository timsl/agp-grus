#!/usr/bin/env Rscript

## install.packages("tidyverse")

library("tidyverse")
library("broom")
library("magrittr")

data <- read_csv("log")

updating <- data %>%
    filter(updating==1) %>%
    select(-updating, -held)
nonupdating <- data %>%
    filter(updating==0) %>%
    select(-updating, -held)

testing <- F
if (testing) {
    ## The held is 0 everywhere, pointless but included for completeness
    data %>%
        summarize(min(held), max(held), mean(held))

    ## View means and sd separately with running particles and no
    data %>%
        group_by(updating) %>%
        summarize(mean(update), sd(update),
                  mean(display), sd(display),
                  mean(all), sd(all))

    ## Update is zero everywhere when not updating
    nonupdating %>%
        summarize(min(update), max(update), mean(update), sd(update))

    ## A few outliers, notably the first display of program run is way slower
    data %>%
        filter(abs(display - median(display)) > sd(display))

    ## Aside from the outliers, no difference in display time
    ## depending on updating. Hence, we seem to be quite free of
    ## confounding variables, and can hereafter only look at updating.
    data %>%
        filter(!(abs(display - median(display)) > sd(display))) %>%
        group_by(updating) %>%
        summarize(mean(display), sd(display))
}

updating %>%
    summarize(mean(update), sd(update),
              mean(display), sd(display))

