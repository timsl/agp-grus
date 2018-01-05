#!/usr/bin/env Rscript

## install.packages("tidyverse")

library("tidyverse")
library("broom")
library("magrittr")

file <- "sumlog"

args = commandArgs(trailingOnly=TRUE)
if (length(args) >= 1){
    file <- args[1]
}

data <- read_csv(file)

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

all <- updating %>%
    group_by(numparts, blocksize) %>%
    summarize(mup=mean(update), sup=sd(update),
              mdi=mean(display), sdi=sd(display))
all %>% print(n = nrow(.))

all %>% select(numparts,blocksize,mup) %>% spread(blocksize, mup)

a <- aes(x=numparts, y=mup, color=blocksize)
logx <- scale_x_continuous(trans="log2")
logy <- scale_y_continuous(trans="log2")

## spline maybe
## data %>%
##     filter(blocksize==256) %>%
##     ggplot(aes(x=numparts, y=update)) +
##     geom_point() +
##     geom_smooth() +
##     logx + logy


all %>%
    mutate(blocksize=as.character(blocksize)) %>%
    ggplot(a) + geom_line() + logx + logy
