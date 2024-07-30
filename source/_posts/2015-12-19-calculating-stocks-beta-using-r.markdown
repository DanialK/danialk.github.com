---
layout: post
title: "Calculating Stocks' Beta Using R"
date: 2015-12-19 20:13
comments: true
categories: 
- R
- Trading
- Finance

---



According to [Investopedia](http://www.investopedia.com/terms/b/beta.asp), Beta is a measure of the volatility, or systematic risk, of a security or a portfolio in comparison to the market as a whole. In this post, we're going to learn how to calculate beta coefficient of our desired stocks using historical price data that is publicly available.

Below we will be covering

* R script to calculate beta for Goldman Sachs
* Maths behind the beta coefficient

<!-- more -->

In this tutorial we are using S&P500 as our market and we're calculating beta of Goldman Sachs in relation to S&P500. It's essentially the correlation of price movements of stocks within the market/index, to individual stocks within the market/index.

In order to calculate beta, we need to obtain historical data for both S&P500 and Goldman Sachs. I'm using Yahoo Finance's public API in order to get historical quotes.

Follow the code and make sure you read the comments !


``` r script.R https://github.com/DanialK/multiple-line-graph-r/blob/master/


# We're obtaining the weekly data
# More information about Yahoo Finance Historical Quotes API
# https://code.google.com/p/yahoo-finance-managed/wiki/csvHistQuotesDownload
indexUrl <- 'http://ichart.yahoo.com/table.csv?s=%5EGSPC&a=00&b=3&c=1950&d=11&e=19&f=2015&g=w'
stockUrl <- 'http://ichart.yahoo.com/table.csv?s=GS&a=04&b=4&c=1999&d=11&e=19&f=2015&g=w'
# stockUrl <- 'http://ichart.yahoo.com/table.csv?s=MSFT&a=02&b=13&c=1986&d=11&e=19&f=2015&g=w'


# read.csv fetches the csv data from an api endpoint if a valid url is provided 
indexData <- read.csv(indexUrl) 
stockData <- read.csv(stockUrl)

# Making sure R knows the Date column is Date
indexData$Date <- as.Date(indexData$Date)
stockData$Date <- as.Date(stockData$Date)

# Usually index contains more values
# Data received form yahoo endpoints are ordered from latest to oldest so 
# we only subset the part of index data that contains stock information in it
indexData <- indexData[1:length(stockData$Date), ]

# Making sure dates are matching and then we grab the weekly close prices of both index and the stock
range <- indexData$Date == stockData$Date
indexPrices <- indexData$Close[range]
stockPrices <- stockData$Close[range]

# Calculating the weekly return, e.g (x2-x1)/x1
indexReturns <- ( indexPrices[1:(length(indexPrices) - 1)] - indexPrices[2:length(indexPrices)] ) / indexPrices[2:length(indexPrices)]
stockReturns <- ( stockPrices[1:(length(stockPrices) - 1)] - stockPrices[2:length(stockPrices)] ) / stockPrices[2:length(stockPrices)]

# using R's lm function, we run a  regression analysis 
# we're using stockReturns as our y value
# and using indexReturns as our x value
# y ~ x is our formula
fit <- lm(stockReturns ~ indexReturns)
result <- summary(fit)
# summary gives us a lot of useful information, but we're mostly in beta value !!
beta <- result$coefficients[2,1]
print(beta)


```

Mathematically, beta is the covariance of stocks percentage daily/weekly change and index/markets daily/weekly change divided by the variance of market's percentage daily/weekly changes:

```Beta = Covariance(Stock's % Change, Index's % Change)/Variance(Index % Change) ```

You can replace the ```stockUrl``` with some other company's data that is in S&P500. Make sure you read the [Yahoo Finance API](https://code.google.com/p/yahoo-finance-managed/wiki/csvHistQuotesDownload) before doing so.

The main purpose of this tutorial is to see how regression can and ```lm()``` can be used in R, but it is also helpful to be able to calculate beta coefficient if you're running your own portfolio. The value obtained using this script (1.453472) at the time of writing was a little bit off from the values provided by [Google Finance](https://www.google.com/finance?q=NYSE%3AGS&ei=ihJ1VsmIH5eb0ATC84yIBA)(1.67) and [Yahoo Finance](https://au.finance.yahoo.com/q/ks?s=GS)(1.38) which can be due to using different time spans for their beta calculations. But since those two numbers don't match, I'd say you should trust the value that you've calculated yourself !!

If you made to here, I hope you found it useful !

Please leave a comment if there are any questions, tips or other concerns.
