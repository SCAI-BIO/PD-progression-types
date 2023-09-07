# general settings, package loading
library(lme4)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(meta)

#' @param x outcome which should be min-max-normalized
#' @return min-max normalized outcome
min_max_norm <- function(x) {
    (x - min(x)) / (max(x) - min(x))
}

#' performs a metaanalysis of all scores of one category including all studies using the meta package
#' @param data Data.frame containing one outcome-study combination per row and the columns n, m, sd and study
#' @param category Name of the category
perform_metaanalysis <- function(data, category) {
    meta_res <- metamean(n = data$n,
                         mean = data$m,
                         sd = data$sd,
                         data = data,
                         subgroup = data$study,
                         studlab = data$study,
                         sm = "MRAW",
                         null.effect = 0,
                         random = TRUE,
                         common = FALSE,
                         title = category)

        forest.meta(meta_res, rightcols = c("effect", "ci"), plotwidth = "12 cm", ref = 0, xlab = "<- Associated with fast-progressing type | associated with slow-progressing type ->", subgroup.hetstat = FALSE, hetstat = FALSE, test.subgroup = FALSE, subgroup.name = "Cohort", header.line = "both")

    ret <- data.frame(
        "study" = unlist(c("overall", meta_res$subgroup.levels)),
        "mean" = unname(c(meta_res$TE.random, meta_res$TE.random.w)),
        "lower" = unname(c(meta_res$lower.random, meta_res$lower.random.w)),
        "upper" = unname(c(meta_res$upper.random, meta_res$upper.random.w)),
        "p" = unname(c(meta_res$pval.random, meta_res$pval.random.w))
    )
    ret$category <- category
    rownames(ret) <- NULL

    return(ret)
}

# load outcome-category mapping
outcomes_categories <- read.csv("categories.csv")

# studies used
study_names <- c("ppmi", "iceberg", "luxpark")

# list used for storing clinical data of all studies
studies <- list()

# load clinical study data
for (study_name in study_names) {
    studies[[study_name]] <- read.csv(paste("../data/", study_name, "_bl.csv", sep = ""))
}

# calculate logistic regression coefficient for predicting the cluster for each outcome in each study

# the output object for all statistics
stats_all <- data.frame()

# loop through each combination of study and outcome;
for (study in study_names){
    # the output object for all statistics
    stats_all <- data.frame()

    for (outcome in outcomes_categories$outcome){
        # skip, if outcome is not measured in this study
        if (!(outcome %in% colnames(studies[[study]]))) {
            next
        }

        # create the outcome dataframe
        data_norm <- studies[[study]][c("Patient_ID", "fit_latenttime", "cluster", outcome)]

        # rename the outcome column
        colnames(data_norm)[colnames(data_norm) == outcome] <- "outcome"

        # remove rows with NA values
        data_norm <- drop_na(data_norm)

        # skip outcomes with less than 30 measurements
        if (nrow(data_norm) < 30) {
            next
        }

        # skip outcomes with less than 5 measurements within a cluster
        if (nrow(data_norm[data_norm$cluster == 0, ]) < 5) {
            next
        }
        if (nrow(data_norm[data_norm$cluster == 1, ]) < 5) {
            next
        }

        # we scale all outcomes to the range 0-1
        data_norm$outcome <- min_max_norm(data_norm$outcome)

        # some scales need to be inverted such that higher values indicate more severe symptoms
        if (1 == outcomes_categories[outcomes_categories$outcome == outcome, ]$invert) {
            data_norm$outcome <- 1 - data_norm$outcome
            log_all <- c(log_all, "...inverted")
        }

        # prepare the dataframe storing the statistics per each cluster
        stats <- data.frame(study = factor(), outcome = factor(), m = numeric(), sd = numeric(), n = numeric())
        stats[1, ] <- NA
        stats$study <- study
        stats$outcome <- outcome

        # cluster as factor factor
        data_norm$cluster <- as.factor(data_norm$cluster)

        # create logistic model
        model <- glm(cluster ~ fit_latenttime + outcome, family = "binomial", data = data_norm)

        # save statistics
        stats$m <- coef(model)[["outcome"]]
        stats$sd <- summary(model)$coefficients["outcome", "Std. Error"]
        stats$n <- nrow(data_norm)
        stats$group <- outcomes_categories[outcomes_categories$outcome == outcome, ]$group

        stats_all <- rbind(stats_all, stats)
    }
}

# perform all meta analyses and create forest plot
data_meta <- data.frame()
pdf(file = "output/bl_forestplots.pdf", width = 13, height = 9)
for (category in unique(stats_all$group)){
    data_category <- stats_all[stats_all$group == category, ]
    if (nrow(data_category) > 0) {
        res <- perform_metaanalysis(data_category, category)
        data_meta <- rbind(data_meta, res)
    }
}
dev.off()