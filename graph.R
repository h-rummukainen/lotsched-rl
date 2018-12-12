# Usage: Rscript vis.r runlog.csv ...
library(ggplot2)
library(dplyr)

read_log <- function(path, kind=NULL) {
    mutate(read.csv(path),
           case=tools::file_path_sans_ext(basename(path)),
           kind=kind,
           cumul_steps=cumsum(steps),
           cost_rate=-rate)
}

plot_runs <- function(runs) {
    ggplot(runs,aes(cumul_steps,cost_rate,colour=case)) + geom_point() + scale_y_log10()
    ggsave("run.pdf", plot=last_plot(), width=14, height=14, units="cm", dpi=300)
}

running_avg <- function(v, n, ext=FALSE) {
        a <- sapply(1:(length(v)-n+1), function(i) mean(v[i:(i+n-1)]))
            if (ext) {
                        c(a, rep(a[length(a)], n-1))
            } else {
                        a
            }
}

read_logs <- function(paths, kind=NULL) {
    do.call(rbind.data.frame,
            lapply(paths, function(f) read_log(f, kind)))
}

bin_runs <- function(runs, bin_steps=10000) {
    mutate(runs, step_bin=floor(cumul_steps/bin_steps)*bin_steps) %>%
        group_by(step_bin, kind) %>%
        summarize(cost_rate_mean=mean(cost_rate),
                  cost_rate_min=min(cost_rate),
                  cost_rate_max=max(cost_rate))
}

plot_run_bands <- function(runs) {
    ggplot(runs) +
        geom_ribbon(aes(step_bin, ymin=cost_rate_min, ymax=cost_rate_max,
                        fill=kind), alpha=0.5) +
        scale_fill_manual(values=c("#ff0000", "#0000ff")) +
        #geom_line(aes(step_bin, cost_rate_mean, colour=kind)) +
        #scale_color_manual(values=c("#ff8000", "#000080")) +
        scale_y_log10()
}

args <- commandArgs(TRUE)

if (length(args) > 0) {
    runs <- read_logs(args, "any")
    plot_runs(runs)
} else {
    loga <- c("abac.csv")
    count <- 11
    log1 <- paste(paste("ppo1.", seq(1,count), sep=""), ".csv", sep="")
    log2 <- paste(paste("ppo2.", seq(1,count), sep=""), ".csv", sep="")
    runs1 <- read_logs(log1, "PPO1")
    runs2 <- read_logs(log2, "PPO2")
    r1band <- bin_runs(runs1, 5000)
    r2band <- bin_runs(runs2, 5000)
    rband <- rbind.data.frame(r1band, r2band)

    r2s <- runs2
    # XXX set these manually
    r2s[r2s$case=="ppo2.4","case"] <- "PPO2.best"
    r2s[r2s$case=="ppo2.3","case"] <- "PPO2.median"
    r2s[r2s$case=="ppo2.10","case"] <- "PPO2.worst"
    ggplot(filter(r2s,case=="PPO2.best"),aes(cumul_steps/1e6,-rate)) + geom_point(size=0.5) + scale_y_log10() + xlab("Simulation steps (millions)") + ylab("Average cost rate ($/h)") + theme_bw() + theme(axis.title.x=element_text(size=8), axis.text.x=element_text(size=8), axis.text.y=element_text(size=8), axis.title.y=element_text(size=8))
    ggsave("ppo2_best_v.pdf", plot=last_plot(), width=10, height=10, units="cm", dpi=300)
    ggplot(filter(r2s,case=="PPO2.worst"),aes(cumul_steps/1e6,-rate)) + geom_point(size=0.5) + scale_y_log10() + xlab("Simulation steps (millions)") + ylab("Average cost rate ($/h)") + theme_bw() + theme(axis.title.x=element_text(size=8), axis.text.x=element_text(size=8), axis.text.y=element_text(size=8), axis.title.y=element_text(size=8))
    ggsave("ppo2_worst_v.pdf", plot=last_plot(), width=10, height=10, units="cm", dpi=300)
}
