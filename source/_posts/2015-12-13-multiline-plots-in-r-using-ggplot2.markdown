---
layout: post
title: "Multiline plots in R using ggplot2"
date: 2015-12-13 15:04
comments: true
categories: 
- R
- ggplot2
- graphing
- plotting

---


{% img /images/multiple-line-graph-r.png %}


With the help of ggplot2, creating beautiful charts is an easy task in R. However it can get a little bit tricky when you're trying to plot a set of data on a single chart, over a shared x axis.

<!-- more -->

A neat trick is using the library ```reshape2``` which is a very useful data manipulation library for R.
With the help of ```melt``` function of this library, we can combine our data into a single data frame in the format that ggplot2 wants from us in order to draw different lines over the same axis.

In this example, in data.csv I have function values of y=x, y=x^2 and y=x^3 for x values from 1 to 10 and i'm trying to draw these 3 charts on the same axis.


Note: if you haven't installed ```ggplot2``` and ```reshape2``` make sure to run
```
install.packages("ggplot2")
``` and
```
install.packages("reshape2")
```

``` js script.R https://github.com/DanialK/multiple-line-graph-r/blob/master/script.R
library(ggplot2)
library(reshape2)

data <- read.csv('./data.csv')

chart_data <- melt(data, id='x')
names(chart_data) <- c('x', 'func', 'value')

ggplot() +
  geom_line(data = chart_data, aes(x = x, y = value, color = func), size = 1)+
  xlab("x axis") +
  ylab("y axis")


```


You can find the source code [here on github](https://github.com/DanialK/multiple-line-graph-r)
