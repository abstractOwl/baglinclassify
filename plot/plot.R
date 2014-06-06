#!/usr/bin/Rscript

library("ggplot2")

png(file="plot.png", units="in", width=8, height=6, res=300)
#pdf(file="plot.pdf")

# Plot training points
pos <- read.table("data/positive.dat", col.names = c("X", "Y"))
neg <- read.table("data/negative.dat", col.names = c("X", "Y"))

pos["Class"] <- "Positive" # 2 = black
neg["Class"] <- "Negative" # 4 = blue

points <- rbind(pos, neg)

p <- qplot(X, Y, data=points, colour=Class)

# Plot classifiers
classifiers <- read.table("data/classifiers.dat", col.names = c("s", "yi"))
p <- p + geom_abline(data=classifiers, mapping=aes(slope=s, intercept=yi),
                     colour="grey", linetype="dashed")

# Plot 
segs <- read.table("data/segments.dat",
                   col.names = c("X1", "Y1", "X2", "Y2", "Bound"))

# Draw bagging classifier decision boundary
p + geom_segment(
    data=segs,
    mapping=aes(
        x=X1,
        y=Y1,
        xend=X2,
        yend=Y2
    ),
    colour=rgb(100/100, 70/100, 20/100),
    size=2,
    lineend="round"
)

dev.off()
