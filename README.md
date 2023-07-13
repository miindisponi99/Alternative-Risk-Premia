# Value-at-Risk of minimum variance Alternative Risk Premia portfolio
The objective of this code is to estimate the Value-at-Risk of a portfolio composed of various academic and trading alternative risk premia. The first step of the Value-at-Risk Alternative Risk Premia portfolio estimation is to calculate the minimum variance among the seven equity strategies. This strategy aims to build an optimal portfolio without making any assumptions on the returns but only minimizing the variance of the portfolio using historical data, such as the volatility of the securities and the correlations that exist among them. As regards the minimum variance optimizer it does not follow a constant correlation strategy so as to have a long/short portfolio characterized by optimal weights determined in the MV portfolio.<br>
In the second step, the ARP portfolio is built to which the Minimum Variance strategy is applied. Among the initial seven allocations, the ARP strategy allows us to select only four weighted factors, the first three which are part of the academic risk premia and the last one which is a trading risk premia. These factors are Value, Quality, Momentum, and Short Volatility. Therefore, the calculation of a long/short portfolio follows these steps:
1. Carry out the stock ranking, calculating the average Z-score between that corresponding to a volatility of 1 year and that corresponding to a volatility of 2 years
2. Build the ARP portfolio selecting the stocks that outperformed (considering as such those belonging to the first 2 deciles of the distribution) and those that underperformed (in this case considering that of the last 2 deciles of the distribution), the former ones are part of the “long” allocation and the latter are included in the “short” allocation. Then, using these stocks the next step is to:
    - Calculate the weights of the long portfolio and those of the short portfolio by applying the Minimum Variance strategy computed before
    - Determine the allocation between the long and short portfolios, applying a leverage constraint of 3x and using the hedge ratio to be beta neutral (ratio between the beta of the best performing factors and that of the least performing ones). In this example, Eurostoxx 50 is the benchmark and allows us to compute beta
3. What we get is a portfolio that weights 4 factors out of the 7 total. We then calculate the Value-at-Risk on a monthly basis considering a confidence interval of 99% (therefore 1% left on the distribution tails) and using the corresponding Z (2.576) to find a VaR of the Minimum Variance ARP portfolio of -1.72%. For the different factors, we find a VaR of:
    - Value: -1.85%
    - Quality: -2.88%
    - Momentum: -0.69%
    - Short Volatility: -0.83%<br>
    
These results show that the level of maximal loss in a time horizon of 1 month and a probability of 1% is not too high. The highest loss probability we could encounter is in the quality factor but overall the portfolio is well distributed. One drawback of the VaR is that we cannot compute the magnitude of these losses that could be recorded beyond the confidence level of the VaR<br>


Then, using a non-parametric Monte-Carlo simulation we are able to estimate the distribution of the Value-at-Risk. Here, we consider 1000 simulations without any distribution assumption since bootstrap Monte-Carlo simulation is agnostic by nature.<br>


Indeed, we have not generated new returns based on a normal distribution as the aim is to resample existing returns to take into consideration the serial correlation properties and the true statistical properties of the unknown joint distributions. As aforementioned, the aim is to compute VaR for our portfolio and we reached 1000 simulations of Value-at-Risk with similar results to that computed before: -1.55% (average of the 1000 simulations). Obviously, this results always change due to random resampling but it is a valuable estimate of the VaR. After having computed the VaR we wanted to check the normality assumption with Jarque- Bera statistics and we reached quite high results in Value, Momentum and Short Volatility Factors. Furthermore, we compared the simulated correlation matrix to the real one and we saw similar results meaning that simulated returns were correctly associated to the original sample. Finally, we calculated the simulated and theoretical mean and variance of estimation errors with a 99% confidence interval and the results are almost matched meaning that overall the simulation is really valuable to compute our Value-at-Risk for the Minimum Variance ARP portfolio.<br>

Afterwards, so as to achieve a time-varying Value-at-Risk, we computed the ARMA General Auto Regressive Conditional Heteroskedastic (GARCH) Model because we wanted to include the lags of the volatility. We made the following steps:
1. Determine ARMA model with AR and MA orders to remove autocorrelation from returns (using 10 lags). We started with the identification of the model (that is, determination of q from the ACF and determination of p from the PACF)
2. Then, we did a return diagnostic by computing the information criteria AIC and BIC in order to minimize them to then being able to start estimating the GARCH for the variance in order to remove the autocorrelation from the volatility too
3. Subsequently we performed some specification tests on the residuals of the model, namely:
    - Normality test using Jarque-Bera
    - Serial correlation test using Ljung-Box<br>
    
and then we calculated the standardized residuals with the annualized conditional volatility<br>


We therefore calculated the time-varying VaR on the base of an ARMA GARCH model and we reached an average VaR of -4.34%. Ultimately, we then measured the accuracy of this Value- at-Risk and finally calculated the 95% confidence interval (low being: -6.71%, and high being: -2.67%).
