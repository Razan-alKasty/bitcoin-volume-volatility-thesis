# ============================================================
# THESIS: Trading Volume and Bitcoin Volatility
# Rennes School of Business — Dr. Taoufik Bouraoui
# R Code: GJR-GARCH(1,1) + VAR + Diagnostic Tests
# Data: Binance BTCUSDT Daily 2019-2024
# ============================================================




setwd("C:/Users/HCES_1/Desktop/gp_code")
#install.packages(c("rugarch", "vars", "tseries", "FinTS", "moments", "ggplot2", "zoo"))
 
library(rugarch)
library(vars)
library(tseries)
library(FinTS)
library(moments)
library(ggplot2)
library(zoo)
 
#STEP 2: Load and prepare data 
#setwd("C:/Users/HCES_1/Desktop/gp_code")
df <- read.csv("Binance_BTCUSDT_d.csv", skip = 1, stringsAsFactors = FALSE)
 
# Rename columns
colnames(df) <- c("Unix", "Date", "Symbol", "Open", "High", "Low",
                  "Close", "Volume_BTC", "Volume_USDT", "tradecount")
 
# Parse dates and sort ascending
df$Date <- as.Date(df$Date, format = "%m/%d/%Y")
df <- df[order(df$Date), ]
#df 
# Filter to January 2019 - December 2024
df <- df[df$Date >= as.Date("2019-01-01") & df$Date <= as.Date("2024-12-31"), ]
 
cat("Number of observations:", nrow(df), "\n")
cat("Date range:", as.character(min(df$Date)), "to", as.character(max(df$Date)), "\n")
 
# TEP 3: Compute variables
 
# Log returns: rt = ln(Pt / Pt-1)
df$log_return <- c(NA, diff(log(df$Close)))
 
# Volatility proxy: sigma_t = |rt|
df$volatility <- abs(df$log_return)
 
# Log of trading volume (BTC)
df$log_volume <- log(df$Volume_BTC)
 
# Remove first row (NA from differencing)
df <- df[!is.na(df$log_return), ]
 
cat("Final observations after differencing:", nrow(df), "\n")
 
#STEP 4: Market regime classification (200-day MA) 
 
# Compute 200-day moving average on Close price
# We use the full price series (before differencing) then align
close_prices <- df$Close
ma200 <- rollmean(close_prices, k = 200, fill = NA, align = "right")
df$MA200 <- ma200
 
# Regime: 1 = bullish (price > MA200), 0 = bearish (price < MA200)
df$regime <- ifelse(df$Close > df$MA200, 1, 0)
 
cat("\nRegime distribution (excluding NA MA period):\n")
print(table(df$regime, useNA = "ifany"))
 
# Subset for regime analysis (remove NA MA200 rows)
df_regimes <- df[!is.na(df$MA200), ]
df_bull <- df_regimes[df_regimes$regime == 1, ]
df_bear <- df_regimes[df_regimes$regime == 0, ]
 
cat("Bullish observations:", nrow(df_bull), "\n")
cat("Bearish observations:", nrow(df_bear), "\n")
 
# ---- STEP 5: Descriptive Statistics ----
 
cat("\n========== DESCRIPTIVE STATISTICS ==========\n")
 
vars_desc <- c("log_return", "volatility", "Volume_BTC", "log_volume")
 
desc_stats <- function(x, name) {
  x <- na.omit(x)
  cat(sprintf("\n--- %s ---\n", name))
  cat(sprintf("  N         : %d\n", length(x)))
  cat(sprintf("  Mean      : %.6f\n", mean(x)))
  cat(sprintf("  Std Dev   : %.6f\n", sd(x)))
  cat(sprintf("  Min       : %.6f\n", min(x)))
  cat(sprintf("  Max       : %.6f\n", max(x)))
  cat(sprintf("  Skewness  : %.4f\n", skewness(x)))
  cat(sprintf("  Kurtosis  : %.4f\n", kurtosis(x)))
}
 
desc_stats(df$log_return,  "Log Return (full sample)")
desc_stats(df$volatility,  "Volatility |rt| (full sample)")
desc_stats(df$Volume_BTC,  "Volume BTC (full sample)")
desc_stats(df$log_volume,  "Log Volume BTC (full sample)")
 
desc_stats(df_bull$log_return, "Log Return — BULLISH")
desc_stats(df_bull$volatility, "Volatility — BULLISH")
desc_stats(df_bull$Volume_BTC, "Volume BTC — BULLISH")
 
desc_stats(df_bear$log_return, "Log Return — BEARISH")
desc_stats(df_bear$volatility, "Volatility — BEARISH")
desc_stats(df_bear$Volume_BTC, "Volume BTC — BEARISH")
 
# ---- STEP 6: Stationarity Tests (ADF) ----
 
cat("\n========== ADF STATIONARITY TESTS ==========\n")
 
adf_result <- function(x, name) {
  result <- adf.test(na.omit(x))
  cat(sprintf("ADF Test — %s\n", name))
  cat(sprintf("  Test statistic : %.4f\n", result$statistic))
  cat(sprintf("  p-value        : %.4f\n", result$p.value))
  cat(sprintf("  Result         : %s\n\n",
              ifelse(result$p.value < 0.05, "STATIONARY (reject H0)", "NON-STATIONARY (fail to reject H0)")))
}
 
adf_result(df$log_return, "Log Returns")
adf_result(df$volatility, "Volatility (|rt|)")
adf_result(df$Volume_BTC, "Volume BTC")
adf_result(df$log_volume, "Log Volume BTC")
 
# ---- STEP 7: ARCH Effects Test ----
 
cat("\n========== ARCH EFFECTS TEST ==========\n")
 
arch_test <- ArchTest(df$log_return, lags = 12)
cat("ARCH-LM Test on Log Returns (12 lags):\n")
cat(sprintf("  Chi-squared statistic : %.4f\n", arch_test$statistic))
cat(sprintf("  p-value               : %.4f\n", arch_test$p.value))
cat(sprintf("  Result                : %s\n\n",
            ifelse(arch_test$p.value < 0.05,
                   "ARCH effects present — GARCH model justified",
                   "No ARCH effects detected")))
 
# ---- STEP 8: GJR-GARCH(1,1) — Full Sample ----
 
cat("\n========== GJR-GARCH(1,1) — FULL SAMPLE ==========\n")
 
# Specification: GJR-GARCH(1,1) with normal distribution
# Mean equation includes log volume as external regressor
garch_spec_full <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0),
                        external.regressors = matrix(df$log_volume, ncol = 1)),
  distribution.model = "norm"
)
 
garch_fit_full <- ugarchfit(spec = garch_spec_full, data = df$log_return)
 
cat("\nGJR-GARCH(1,1) Full Sample Results:\n")
print(garch_fit_full)
 
# Extract key coefficients
coef_full <- coef(garch_fit_full)
cat("\nCoefficients:\n")
print(round(coef_full, 6))
 
# Persistence = alpha + beta + 0.5*gamma
alpha <- coef_full["alpha1"]
beta  <- coef_full["beta1"]
gamma <- coef_full["gamma1"]
persistence <- alpha + beta + 0.5 * gamma
cat(sprintf("\nVolatility Persistence (alpha + beta + 0.5*gamma): %.4f\n", persistence))
 
# Information criteria
ic_full <- infocriteria(garch_fit_full)
cat("\nInformation Criteria:\n")
print(round(ic_full, 4))
 
# ---- STEP 9: GJR-GARCH(1,1) — Bullish Regime ----
 
cat("\n========== GJR-GARCH(1,1) — BULLISH REGIME ==========\n")
 
garch_spec_bull <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0),
                        external.regressors = matrix(df_bull$log_volume, ncol = 1)),
  distribution.model = "norm"
)
 
garch_fit_bull <- ugarchfit(spec = garch_spec_bull, data = df_bull$log_return)
 
cat("\nGJR-GARCH(1,1) Bullish Regime Results:\n")
print(garch_fit_bull)
 
coef_bull <- coef(garch_fit_bull)
alpha_b <- coef_bull["alpha1"]
beta_b  <- coef_bull["beta1"]
gamma_b <- coef_bull["gamma1"]
persistence_bull <- alpha_b + beta_b + 0.5 * gamma_b
cat(sprintf("\nVolatility Persistence (Bullish): %.4f\n", persistence_bull))
 
ic_bull <- infocriteria(garch_fit_bull)
cat("\nInformation Criteria (Bullish):\n")
print(round(ic_bull, 4))
 
# ---- STEP 10: GJR-GARCH(1,1) — Bearish Regime ----
 
cat("\n========== GJR-GARCH(1,1) — BEARISH REGIME ==========\n")
 
garch_spec_bear <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0),
                        external.regressors = matrix(df_bear$log_volume, ncol = 1)),
  distribution.model = "norm"
)
 
garch_fit_bear <- ugarchfit(spec = garch_spec_bear, data = df_bear$log_return)
 
cat("\nGJR-GARCH(1,1) Bearish Regime Results:\n")
print(garch_fit_bear)
 
coef_bear <- coef(garch_fit_bear)
alpha_e <- coef_bear["alpha1"]
beta_e  <- coef_bear["beta1"]
gamma_e <- coef_bear["gamma1"]
persistence_bear <- alpha_e + beta_e + 0.5 * gamma_e
cat(sprintf("\nVolatility Persistence (Bearish): %.4f\n", persistence_bear))
 
ic_bear <- infocriteria(garch_fit_bear)
cat("\nInformation Criteria (Bearish):\n")
print(round(ic_bear, 4))
 
# ---- STEP 11: VAR Model — Full Sample ----
 
cat("\n========== VAR MODEL — FULL SAMPLE ==========\n")
 
# Prepare bivariate system: volatility and log volume
var_data_full <- na.omit(data.frame(
  volatility  = df_regimes$volatility,
  log_volume  = df_regimes$log_volume
))
 
# Select optimal lag length using AIC
var_select_full <- VARselect(var_data_full, lag.max = 10, type = "const")
cat("Optimal lag selection (AIC):\n")
print(var_select_full$selection)
 
optimal_lag_full <- var_select_full$selection["AIC(n)"]
cat(sprintf("Selected lag order: %d\n", optimal_lag_full))
 
# Estimate VAR
var_fit_full <- VAR(var_data_full, p = optimal_lag_full, type = "const")
cat("\nVAR Full Sample Summary:\n")
summary(var_fit_full)
 
# ---- STEP 12: Granger Causality Tests ----
 
cat("\n========== GRANGER CAUSALITY TESTS — FULL SAMPLE ==========\n")
 
# Does volume Granger-cause volatility?
gc_vol_to_vola <- causality(var_fit_full, cause = "log_volume")
cat("H0: Log Volume does NOT Granger-cause Volatility\n")
cat(sprintf("  F-statistic : %.4f\n", gc_vol_to_vola$Granger$statistic))
cat(sprintf("  p-value     : %.4f\n", gc_vol_to_vola$Granger$p.value))
cat(sprintf("  Result      : %s\n\n",
            ifelse(gc_vol_to_vola$Granger$p.value < 0.05,
                   "REJECT H0 — Volume Granger-causes Volatility",
                   "FAIL TO REJECT H0 — No Granger causality")))
 
# Does volatility Granger-cause volume?
gc_vola_to_vol <- causality(var_fit_full, cause = "volatility")
cat("H0: Volatility does NOT Granger-cause Log Volume\n")
cat(sprintf("  F-statistic : %.4f\n", gc_vola_to_vol$Granger$statistic))
cat(sprintf("  p-value     : %.4f\n", gc_vola_to_vol$Granger$p.value))
cat(sprintf("  Result      : %s\n\n",
            ifelse(gc_vola_to_vol$Granger$p.value < 0.05,
                   "REJECT H0 — Volatility Granger-causes Volume",
                   "FAIL TO REJECT H0 — No Granger causality")))
 
# ---- STEP 13: VAR — Bullish Regime ----
 
cat("\n========== VAR MODEL — BULLISH REGIME ==========\n")
 
var_data_bull <- na.omit(data.frame(
  volatility = df_bull$volatility,
  log_volume = df_bull$log_volume
))
 
var_select_bull <- VARselect(var_data_bull, lag.max = 10, type = "const")
cat("Optimal lag (Bullish, AIC):", var_select_bull$selection["AIC(n)"], "\n")
optimal_lag_bull <- var_select_bull$selection["AIC(n)"]
 
var_fit_bull <- VAR(var_data_bull, p = optimal_lag_bull, type = "const")
 
gc_bull_vol_to_vola <- causality(var_fit_bull, cause = "log_volume")
cat("Granger: Volume -> Volatility (BULLISH)\n")
cat(sprintf("  F-statistic : %.4f\n", gc_bull_vol_to_vola$Granger$statistic))
cat(sprintf("  p-value     : %.4f\n", gc_bull_vol_to_vola$Granger$p.value))
cat(sprintf("  Result      : %s\n\n",
            ifelse(gc_bull_vol_to_vola$Granger$p.value < 0.05,
                   "SIGNIFICANT", "NOT SIGNIFICANT")))
 
gc_bull_vola_to_vol <- causality(var_fit_bull, cause = "volatility")
cat("Granger: Volatility -> Volume (BULLISH)\n")
cat(sprintf("  F-statistic : %.4f\n", gc_bull_vola_to_vol$Granger$statistic))
cat(sprintf("  p-value     : %.4f\n", gc_bull_vola_to_vol$Granger$p.value))
cat(sprintf("  Result      : %s\n\n",
            ifelse(gc_bull_vola_to_vol$Granger$p.value < 0.05,
                   "SIGNIFICANT", "NOT SIGNIFICANT")))
 
# ---- STEP 14: VAR — Bearish Regime ----
 
cat("\n========== VAR MODEL — BEARISH REGIME ==========\n")
 
var_data_bear <- na.omit(data.frame(
  volatility = df_bear$volatility,
  log_volume = df_bear$log_volume
))
 
var_select_bear <- VARselect(var_data_bear, lag.max = 10, type = "const")
cat("Optimal lag (Bearish, AIC):", var_select_bear$selection["AIC(n)"], "\n")
optimal_lag_bear <- var_select_bear$selection["AIC(n)"]
 
var_fit_bear <- VAR(var_data_bear, p = optimal_lag_bear, type = "const")
 
gc_bear_vol_to_vola <- causality(var_fit_bear, cause = "log_volume")
cat("Granger: Volume -> Volatility (BEARISH)\n")
cat(sprintf("  F-statistic : %.4f\n", gc_bear_vol_to_vola$Granger$statistic))
cat(sprintf("  p-value     : %.4f\n", gc_bear_vol_to_vola$Granger$p.value))
cat(sprintf("  Result      : %s\n\n",
            ifelse(gc_bear_vol_to_vola$Granger$p.value < 0.05,
                   "SIGNIFICANT", "NOT SIGNIFICANT")))
 
gc_bear_vola_to_vol <- causality(var_fit_bear, cause = "volatility")
cat("Granger: Volatility -> Volume (BEARISH)\n")
cat(sprintf("  F-statistic : %.4f\n", gc_bear_vola_to_vol$Granger$statistic))
cat(sprintf("  p-value     : %.4f\n", gc_bear_vola_to_vol$Granger$p.value))
cat(sprintf("  Result      : %s\n\n",
            ifelse(gc_bear_vola_to_vol$Granger$p.value < 0.05,
                   "SIGNIFICANT", "NOT SIGNIFICANT")))
 
# ---- STEP 15: Save cleaned data for Python ----
 
df_export <- df_regimes[, c("Date", "Close", "log_return", "volatility",
                             "Volume_BTC", "log_volume", "MA200", "regime")]
df_export$Date <- as.character(df_export$Date)
 
write.csv(df_export, "BTC_clean_for_python.csv", row.names = FALSE)
cat("\nCleaned data saved as BTC_clean_for_python.csv\n")
cat("Upload this file to Google Colab for the Python (RF + LSTM) code.\n")
 

cat("\n========== R CODE COMPLETED SUCCESSFULLY ==========\n")
cat("Copy ALL output above and paste it to Claude for Chapter 4 writing.\n")



# Extract conditional volatility
cond_vol <- as.numeric(sigma(garch_fit_full))

# Use df directly since garch was fitted on df$log_return
plot_data <- data.frame(
  Date = as.Date(df$Date),
  CondVol = cond_vol,
  Regime = ifelse(is.na(df$regime), 0, df$regime)
)

ggplot(plot_data, aes(x = Date, y = CondVol)) +
  geom_line(colour = "black", linewidth = 0.5) +
  geom_rect(data = subset(plot_data, Regime == 1),
            aes(xmin = Date, xmax = Date + 1, ymin = 0, ymax = Inf),
            fill = "green", alpha = 0.02) +
  geom_rect(data = subset(plot_data, Regime == 0),
            aes(xmin = Date, xmax = Date + 1, ymin = 0, ymax = Inf),
            fill = "red", alpha = 0.02) +
  labs(title = "Figure 1. Conditional Volatility — GJR-GARCH(1,1)",
       subtitle = "Green = Bullish Regime | Red = Bearish Regime",
       x = "Date", y = "Conditional Volatility") +
  theme_minimal()
