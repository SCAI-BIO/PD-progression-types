library(lme4)
library(ordinal)
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
#' @param data Data.frame containing one outcome-study combination per row and the columns n1, m1, sd1, n2, m2, sd2 and study
#' @param category Name of the category
perform_metaanalysis <- function(data, category) {
    meta_res <- metacont(n.c = data$n0,
                         mean.c = data$m0,
                         sd.c = data$sd0,
                         n.e = data$n1,
                         mean.e = data$m1,
                         sd.e = data$sd1,
                         data = data,
                         subgroup = data$study,
                         studlab = data$study,
                         sm = "SMD",
                         random = TRUE,
                         common = FALSE,
                         label.c = "Fast progressing",
                         label.e = "Slow progressing",
                         title = category)

    forest.meta(meta_res, leftcols = c("studlab", "n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), rightcols=c("effect", "ci"), plotwidth = "12 cm", ref = 0, xlab = "<- Faster progression for fast-progressing subtype | faster progression for slow-progressing subtype ->", subgroup.hetstat = FALSE, hetstat = FALSE, test.subgroup = FALSE, subgroup.name = "Cohort", header.line = "both")

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
    studies[[study_name]] <- read.csv(paste("../data/", study_name, "_visits.csv", sep = ""))
}

# calculate regression coefficient for each outcome in each study and each cluster

# the output object for all statistics
stats_all <- data.frame()

# loop through each combination of study and outcome;
for (study in study_names){

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

        # because we want to calculate slopes, we have to remove all patients with only one measurement
        data_norm <- data_norm %>% group_by(Patient_ID) %>% filter(n() > 1)

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
        }

        # prepare the dataframe storing the statistics per each cluster
        stats <- data.frame(study = factor(), outcome = factor(), m0 = numeric(), sd0 = numeric(), n0 = numeric(), m1 = numeric(), sd1 = numeric(), n1 = numeric())
        stats[1, ] <- NA
        stats$study <- study
        stats$outcome <- outcome
        regression_type <- outcomes_categories[outcomes_categories$outcome == outcome, ]$regression
        stats$group <- outcomes_categories[outcomes_categories$outcome == outcome, ]$group

        # depending on the outcome scale (continuous, binary or ordinal) we have to use different models
        if ("linear" == regression_type) {
            # for continous data: linear mixed model mith random slope + intercept
            tryCatch({
                for (cluster in c(0, 1)){
                    # filter data from one cluster
                    data_lmm <- data_norm[data_norm$cluster == cluster, ]

                    # fit model, calculate slopes from fixed effect + random effect
                    lmm <- lmer(outcome ~ fit_latenttime + (1 + fit_latenttime|Patient_ID), data = data_lmm)
                    slopes <- fixef(lmm)[2] + ranef(lmm)$Patient_ID$fit_latenttime

                    # save statistics
                    stats[[paste("m", cluster, sep = "")]] <- mean(slopes)
                    stats[[paste("sd", cluster, sep = "")]] <- sd(slopes)
                    stats[[paste("n", cluster, sep = "")]] <- length(slopes)
                }
            },
            error = function(e) {
                for (cluster in c(0, 1)){
                    stats[[paste("m", cluster, sep = "")]] <<- NA
                    stats[[paste("sd", cluster, sep = "")]] <<- NA
                    stats[[paste("n", cluster, sep = "")]] <<- NA
                }
            })
        } else if ("binary" == regression_type) {
            # for binary data: binary regression model mith random slope + intercept
            for (cluster in c(0, 1)){
                # filter data from one cluster
                data_bmm <- data_norm[data_norm$cluster == cluster, ]

                # fit model, calculate slopes from fixed effect + random effect
                bmm <- glmer("outcome ~ fit_latenttime + (1+fit_latenttime|Patient_ID)", data = data_bmm, family = "binomial")
                slopes <- fixef(bmm)[2] + ranef(bmm)$Patient_ID$fit_latenttime

                # save statistics
                stats[[paste("m", cluster, sep = "")]] <- mean(slopes)
                stats[[paste("sd", cluster, sep = "")]] <- sd(slopes)
                stats[[paste("n", cluster, sep = "")]] <- length(slopes)
            }
        } else if ("ordinal" == regression_type) {
            # for ordinal data: ordinal regression model mith random slope + intercept
            tryCatch({
                for (cluster in c(0, 1)){
                    # filter data from one cluster
                    data_omm <- data_norm[data_norm$cluster == cluster, ]
                    data_omm$outcome <- as.factor(data_omm$outcome)

                    # fit model, calculate slopes from fixed effect + random effect
                    omm <- clmm(outcome ~ fit_latenttime + (1 + fit_latenttime | Patient_ID), data = data_omm)
                    slopes <- coef(omm)[["fit_latenttime"]] + ranef(omm)$Patient_ID$fit_latenttime

                    # save statistics
                    stats[[paste("m", cluster, sep = "")]] <- mean(slopes)
                    stats[[paste("sd", cluster, sep = "")]] <- sd(slopes)
                    stats[[paste("n", cluster, sep = "")]] <- length(slopes)
                }
            },
            # sometimes the model does not converge, than try a more simple model without random intercepts
            error = function(e) {
                tryCatch({
                    for (cluster in c(0, 1)){
                        # filter data from one cluster
                        data_omm <- data_norm[data_norm$cluster == cluster, ]
                        data_omm$outcome <- as.factor(data_omm$outcome)

                        # fit model, calculate slopes from fixed effect + random effect
                        omm <- clmm(outcome ~ fit_latenttime + (0 + fit_latenttime | Patient_ID), data = data_omm)
                        slopes <- coef(omm)[["fit_latenttime"]] +  + ranef(omm)$Patient_ID$fit_latenttime

                        # save statistics
                        stats[[paste("m", cluster, sep = "")]] <<- mean(slopes)
                        stats[[paste("sd", cluster, sep = "")]] <<- sd(slopes)
                        stats[[paste("n", cluster, sep = "")]] <<- length(slopes)
                    }
                },
                # if no convergence for both attempts: NAN
                error = function(e) {
                    for (cluster in c(0, 1)){
                        stats[[paste("m", cluster, sep = "")]] <<- NA
                        stats[[paste("sd", cluster, sep = "")]] <<- NA
                        stats[[paste("n", cluster, sep = "")]] <<- NA
                    }
                })
            })
        } else {
            stop(paste("Regression type not implemented:", regression_type))
        }

        stats_all <- rbind(stats_all, stats)
    }
}

# perform all meta analyses and save forest plot
data_meta <- data.frame()
pdf(file = "output/slopes_forestplots.pdf", width = 13, height = 8)
for (category in unique(stats_all$group)){
    data_category <- stats_all[stats_all$group == category, ]
    if (nrow(data_category) > 0) {
        res <- perform_metaanalysis(data_category, category)
        data_meta <- rbind(data_meta, res)
    }
}
dev.off()