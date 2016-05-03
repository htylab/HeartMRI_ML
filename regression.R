setwd("~/Documents/CSML/DSBC/")

dat2 <- read.csv(file = "results2.csv", header = FALSE, sep = ",")
dat4 <- read.csv(file = "results4.csv", header = FALSE, sep = ",")
dat5 <- read.csv(file = "results5.csv", header = FALSE, sep = ",")

dat <- rbind(dat2, dat4, dat5)
head(dat)
colnames(dat) <- c("pat_id", "edv", "esv", "EDA", "ESA")
attach(dat)
plot(edv, log(EDA), pch = 20)
points(esv, log(ESA), pch = 22, col = 2)
legend(250,11000, legend <- c("edv/EDA", "esv/ESA"), col = c(1,2), pch = c(20,22))


lm.vols_ED <- lm(edv ~ log(EDA))
lm.vols_ES <- lm(esv ~ log(ESA))
summary(lm.vols_ED)
summary(lm.vols_ES)

lm.vols_ED <- lm(edv ~ EDA)
lm.vols_ES <- lm(esv ~ ESA)
summary(lm.vols_ED)
summary(lm.vols_ES)

