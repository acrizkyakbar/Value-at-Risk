# IMPORT PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
from dateutil.relativedelta import relativedelta  

np.random.seed(42)

# ===========================================================================================================================================

# DATA PREPARATION

def dataPrep(tickers, start, end):
    """Funtion for Data Preparation

    Args:
        tickers (List)  : list of Asset Tickers.
        start (String)  : Start Date (Format: 'YYYY-MM-DD').
        end (String)    : End Date   (Format: 'YYYY-MM-DD').

    Returns:
        returns (Series or DataFrame)   : Daily Returns of Asset(s).
        e (Float or Series)             : Daily Mean Return (Expected Return) of Asset(s).
        std (Float or Series)           : Daily Standard Deviation (Volatility) of Asset(s).
        V (DataFrame)                   : Daily Variance-Covariance Matrix of Asset Returns.
    """
    # Get Asset(s) Adjusted Close Data
    data = yf.download(tickers, start, end, rounding=True)['Adj Close']
    
    # Get Daily Returns of Asset(s)
    returns = data.pct_change().dropna()
    
    # Single Asset Portfolio
    if isinstance(returns, pd.Series):
        # Get Daily Mean Return and Standard Deviation of the Asset
        e = returns.mean()
        std = returns.std()
        return returns, e, std

    # Multiple Assets Portfolio
    elif isinstance(returns, pd.DataFrame):
        # Get Daily Mean Return and Standard Deviation and Variance-Covariance Matrix of the Assets
        e = returns.mean()
        std = returns.std()
        V = returns.cov()
        return returns, e, std, V

# ===========================================================================================================================================

# HISTORICAL VaR

def Historical_VaR(portfolio, returns, weights, cl, t):
    """Function to Determine Value-at-Risk according to Historical Approach

    Args:
        portfolio (Float)                       : Value of Portfolio.
        returns (Series or DataFrame, Float)    : Returns of Asset(s) in Portfolio.
        weights (List or array)                 : Weight of Individual Asset in Portfolio (Sum up to 1).
        cl (Integer)                            : Confidence Level in % (Preferred: 90, 95, 99).
        t (Integer)                             : Time Horizon (in days).

    Returns:
        pct_VaR (Float): Percentage Value-at-Risk.
        abs_VaR (Float): Absolute Value-at-Risk.
    """
    # Determine Alpha of Confidence Level
    alpha = 1 - cl/100
    
    # Convert 'weights' Data Type to Array in Case It is Input as a List
    weights = np.array(weights)
    
    # Single Asset Case    
    if isinstance(returns, pd.Series):
        weights = 1        
        pct_VaR = np.quantile(returns, alpha) * np.sqrt(t)
        abs_VaR = np.quantile(returns * portfolio, alpha) * np.sqrt(t)
        
        return pct_VaR.round(4), abs_VaR.round(2)
    
    # Multiple Assets Case
    elif isinstance(returns, pd.DataFrame):
        returns['portfolio'] = np.dot(returns, weights)
        pct_VaR = np.quantile(returns.portfolio, alpha) * np.sqrt(t)
        abs_VaR = np.quantile(returns.portfolio * portfolio, alpha) * np.sqrt(t)
        
        return pct_VaR.round(4), abs_VaR.round(2)

# =======
# EXAMPLE
# =======

# 1. Single Asset Case

assets = ['MSFT']
returns, e, std = dataPrep(tickers = assets, start = '2020-09-20', end = '2023-09-20')

portfolio = 1000000
weights = np.array([1])

pct_VaR, abs_VaR = Historical_VaR(portfolio = portfolio, returns = returns, weights = weights, cl = 95, t = 1)
print(f'1-Day 95% Percentage Historical VaR of {assets[0]}   = {(-1 * pct_VaR*100).round(2)}%')
print(f'1-Day 95% Absolute Historical VaR of {assets[0]}     = $ {-1 * abs_VaR}')

# 2. Multiple Assets Case

assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ']
returns, e, std, V = dataPrep(tickers = assets, start = '2020-09-20', end = '2023-09-20')

portfolio = 1000000
weights = np.array([1/len(assets)] * len(assets))

pct_VaR, abs_VaR = Historical_VaR(portfolio = portfolio, returns = returns, weights = weights, cl = 95, t = 1)
print(f'1-Day 95% Percentage Historical VaR of Portfolio   = {-1 * pct_VaR*100}%')
print(f'1-Day 95% Absolute Historical VaR of Portfolio     = $ {-1 * abs_VaR}')

# # ===========================================================================================================================================

# PARAMETRIC APPROACH: STOCK PORTFOLIO

def Parametric_VaR(portfolio, returns, weights, cl, t):
    """Function to Determine Value-at-Risk according to Parametric Approach for Stock Portfolio.

    Args:
        portfolio (Float)                       : Value of Portfolio.
        returns (Series or DataFrame, Float)    : Returns of Asset(s) in Portfolio.
        weights (List or array)                 : Weight of Individual Asset in Portfolio (Sum up to 1).
        cl (Integer)                            : Confidence Level (Preferred: 90, 95, 99).
        t (Integer)                             : Time Horizon (in days).

    Returns:
        pct_VaR (Float): Percentage Value-at-Risk.
        abs_VaR (Float): Absolute Value-at-Risk.
    """
    # Determine Alpha of Confidence Level
    alpha = 1 - cl/100
    
    # Convert 'weights' Data Type to Array in Case It is Input as a List
    weights = np.array(weights)
    
    # Single Asset Portfolio
    if isinstance(returns, pd.Series):
        # Get Daily Mean Return and Standard Deviation of the Asset
        e = returns.mean()
        std = returns.std()

        # VaR
        pct_VaR = norm.ppf(alpha) * std * np.sqrt(t) - e * t
        abs_VaR = norm.ppf(alpha) * std * portfolio * np.sqrt(t) - e * portfolio * t
        
        return pct_VaR.round(4), abs_VaR.round(2)
    
    # Multiple Assets Portfolio
    elif isinstance(returns, pd.DataFrame):
        # Get Daily Mean Return, Standard Deviation and Variance-Covariance Matrix of the Assets
        e = returns.mean()
        std = returns.std()
        V = returns.cov()
        
        # Mean Return of the Portfolio
        eP = np.dot(weights.T, e)
        
        # Variance of the Portfolio
        varP = np.dot(np.dot(weights.T, V), weights)
        
        # Standard Deviation of the Portfolio
        stdP = np.sqrt(varP)
        
        # VaR
        pct_VaR = norm.ppf(alpha) * stdP * np.sqrt(t) - eP * t
        abs_VaR = norm.ppf(alpha) * stdP * portfolio * np.sqrt(t) - eP * portfolio * t
        
        return pct_VaR.round(4), abs_VaR.round(2)

# # =======
# # EXAMPLE
# # =======

# 1. Single Asset Case

assets = ['MSFT']
returns, e, std = dataPrep(tickers = assets, start = '2020-09-20', end = '2023-09-20')

portfolio = 1000000
weights = np.array([1])

pct_VaR, abs_VaR = Parametric_VaR(portfolio = portfolio, returns = returns, weights = weights, cl = 95, t = 1)
print(f'1-Day 95% Percentage Parametric VaR of {assets[0]}   = {-1 * pct_VaR*100}%')
print(f'1-Day 95% Absolute Parametric VaR of {assets[0]}     = $ {-1 * abs_VaR}')

# 2. Multiple Assets Case

assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ']
returns, e, std, V = dataPrep(tickers = assets, start = '2020-09-20', end = '2023-09-20')

portfolio = 1000000
weights = np.array([1/len(assets)] * len(assets))

pct_VaR, abs_VaR = Parametric_VaR(portfolio = portfolio, returns = returns, weights = weights, cl = 95, t = 1)
print(f'1-Day 95% Percentage Parametric VaR of Portfolio   = {-1 * pct_VaR*100}%')
print(f'1-Day 95% Absolute Parametric VaR of Portfolio     = $ {-1 * abs_VaR}')

# # ===========================================================================================================================================

# PARAMETRIC APPROACH: OPTION PORTFOLIO

def bs(S, K, t, r, std, option_type = 'c'):
    """Option Price and Greeks according to Black-Scholes Model

    Args:
        S (Float)                       : Underlying Price.
        K (Float)                       : Strike Price.
        t (Integer)                     : Time to Expiration.
        r (Float)                       : Risk-free Rate.
        std (FLoat)                     : Daily Standard Deviation of Underlying Asset.
        option_type (String, optional)  : Type of Option (Call or Put). Defaults to 'c'.

    Raises:
        ValueError: Raise an error if the option_type is not correctly input.

    Returns:
        option_price (Float)    : Price of Option
        delta (Float)           : Delta of Option
        theta (Float)           : Theta of Option
        gamma (Float)           : Gamma of Option
        vega (Float)            : Vega of Option
        rho (Float)             : Rho of Option
    """
    # Convert Daily to Annual Time to Expiration
    T = t / 252
    
    # Convert Daily to Annual Standard Deviation
    annualized_std = std * np.sqrt(252)

    # Option Price
    d1 = (np.log(S / K) + (r + (annualized_std ** 2) / 2) * T) / (annualized_std * np.sqrt(T))
    d2 = d1 - annualized_std * np.sqrt(T)

    try:
        if option_type == 'c':
            norm_d1 = norm.cdf(d1)
            norm_d2 = norm.cdf(d2)
            option_price = S * norm_d1 - K * np.exp(-r * T) * norm_d2
        elif option_type == 'p':
            norm_minus_d1 = norm.cdf(-d1)
            norm_minus_d2 = norm.cdf(-d2)
            option_price = K * np.exp(-r * T) * norm_minus_d2 - S * norm_minus_d1
        else:
            raise ValueError("Invalid option type. Use 'c' for call and 'p' for put.")
        
    # Option Greeks
        delta = norm_d1 if option_type == 'c' else -norm_minus_d1
        theta = ((-S * norm.pdf(d1) * annualized_std / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm_d2)) / 252 if option_type == 'c' else \
                ((-S * norm.pdf(d1) * annualized_std / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm_minus_d2)) / 252
        gamma = norm.pdf(d1) / (S * annualized_std * np.sqrt(T))
        vega  = S * np.sqrt(T) * norm.pdf(d1) * annualized_std
        rho   = K * T * np.exp(-r * T) * norm_d2 if option_type == 'c' else \
                -K * T * np.exp(-r * T) * norm_minus_d2
            
    except:
        print('Invalid Input, Check the Parameters')
        return None

    return option_price.round(2), delta.round(6), theta.round(6), gamma.round(6), vega.round(2), rho.round(2)


def OptionPortfolio(tickers, K, r, t, end = datetime.now(), option_type = 'c'):
    """Function for Storing Option Portfolio in a DataFrame

    Args:
        tickers (List, String)      : List of Underlying Asset(s).
        K (List, Float)             : Strike Price.
        r (Float)                   : Risk-free Rate.
        t (List, Integer)           : Time to Expiration
        end (String)                : End Date   (Format: 'YYYY-MM-DD'). Defaults to Today's Date.
        option_type (List, String)  : Type of Option. Defaults to 'c'.

    Returns:
        df (DataFrame)  : Option DataFrame
        std             : Daily Standard Deviation of Underlying Asset(s)' Returns
    """
    # Get 20 Years Underlying Daily Adjusted Close Price Data
    data = yf.download(tickers, start = (datetime.strptime(end, '%Y-%m-%d') - relativedelta(years=20)), end = end, rounding = True)['Adj Close']
    
    # Get The Latest Underlying Price
    S = data.tail(1).values[0]
    
    # Daily Standard Deviation of Underlying Returns
    returns = data.pct_change()
    std = np.array(returns.std())
    
    # Option Parameters DataFrame
    option_parameters = {'underlying_asset' : tickers,
            'underlying_price' : S,
            'strike_price' : K,
            'expiration' : t,
            'rf' : r,
            'annualized_std' : std * np.sqrt(252),
            'option_type': option_type
            }
    df_parameters = pd.DataFrame(option_parameters)

    # Option Price and Greeks Data
    options = []
    for i in range(len(df_parameters)):
        option_data = bs(S = df_parameters.iloc[i][1], K = df_parameters.iloc[i][2], t = df_parameters.iloc[i][3], r = df_parameters.iloc[i][4], std = df_parameters.iloc[i][5], option_type = df_parameters.iloc[i][6])
        options.append([*option_data])
        
    # Option Price and Greeks DataFrame
    df_options = pd.DataFrame(options, columns = ['option_price', 'delta', 'theta', 'gamma', 'vega', 'rho'])
    
    # Concat Both DataFrame
    df = pd.concat([df_parameters, df_options], axis=1)
    
    return df, std

df, std = OptionPortfolio(tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ'],
                            end = '2023-09-20',
                            K = [179.22, 369.87, 442.71,  92.8 ,  80.71],
                            t = 25,
                            r = 0.05,
                            option_type = ['c', 'c', 'p', 'c', 'p']
                            )

print(df)
print(std)

# DELTA NORMAL VaR

def delta_VaR(S, delta, std, cl):
    """Function for Determining Value-at-Risk of Option Portfolio according Delta Normal Approach

    Args:
        S (List/ Series/ Array, Float)      : Underlying Price.
        delta (List/ Series/ Array, Float)  : Delta of Option.
        std (List/ Series/ Array, Float)    : Daily Standard Deviation of Risk Factor (In this case: Underlying Price).
        cl (List/ Series/ Array, Integer)   : Confidence Level (Preferred: 90, 95, 99).

    Returns:
        pct_VaR (Float): Percentage Value-at-Risk.
        abs_VaR (Float): Absolute Value-at-Risk.
    """
    # Determine Alpha of Confidence Level
    alpha = 1 - cl/100
    
    # Convert S and delta to Array in Case They are Input as List or Series
    S       = np.array(S)
    delta   = np.array(delta)
    
    # The Change of Risk Factor (Underlying Price) = Alpha-Quantile
    dx = np.array(std * norm.ppf(alpha))
    
    # Percentage VaR
    pct_VaR = np.dot(delta, dx)
    
    # Absolute VaR
    abs_VaR = np.dot(S * delta, dx)
    
    return pct_VaR.round(4), abs_VaR.round(2)

# # =======
# # EXAMPLE
# # =======

# 1. Single Option Portfolio
pct_VaR, abs_VaR = delta_VaR(S = df.underlying_price[0], delta = df.delta[0], std = std[0], cl = 95)
print(f'1-Day 95% Percentage Delta Normal VaR of {df.underlying_asset[0]} Option = {(pct_VaR*100).round(2)}%')
print(f'1-Day 95% Absolute Delta Normal VaR of {df.underlying_asset[0]} Option   = $ {abs_VaR}')

# 2. Multiple Option Portfolio
pct_VaR, abs_VaR = delta_VaR(S = df.underlying_price, delta = df.delta, std = std, cl = 95)
print(f'1-Day 95% Percentage Delta Normal VaR of Option Portfolio = {pct_VaR*100}%')
print(f'1-Day 95% Absolute Delta Normal VaR of Option Portfolio   = $ {abs_VaR}')
print()

# DELTA-GAMMA NORMAL VaR

def delta_gamma_VaR(S, delta, gamma, std, cl):
    """Function for Determining Value-at-Risk of Option Portfolio according Delta-Gamma Normal Approach

    Args:
        S (List/ Series/ Array, Float)      : Underlying Price.
        delta (List/ Series/ Array, Float)  : Delta of Option.
        gamma (List/ Series/ Array, Float)  : Gamma of Option.
        std (List/ Series/ Array, Float)    : Daily Standard Deviation of Risk Factor (In this case: Underlying Price).
        cl (List/ Series/ Array, Integer)   : Confidence Level (Preferred: 90, 95, 99).

    Returns:
        pct_VaR (Float): Percentage Value-at-Risk.
        abs_VaR (Float): Absolute Value-at-Risk.
    """
    # Determine Alpha of Confidence Level
    alpha = 1 - cl/100
    
    # Convert S, delta and gamma to Array in Case They are Input as List or Series    
    S       = np.array(S)
    delta   = np.array(delta)
    gamma   = np.array(gamma)
    
    # The Change of Risk Factor (Underlying Price) = Alpha-Quantile
    dx = np.array(std * norm.ppf(alpha))
    
    # The Change of Risk Factor (Underlying Price) Square = Alpha-Quantile Square
    dx_sqr = np.array(dx**2)
    
    # Percentage VaR
    pct_VaR = np.dot(delta.T, dx) + 0.5 * np.dot(gamma.T, dx_sqr)
    
    # Absolute VaR
    abs_VaR = np.dot(S * delta.T, dx) + 0.5 * np.dot(S**2 * gamma.T, dx_sqr)
    
    return pct_VaR.round(4), abs_VaR.round(2)

# # =======
# # EXAMPLE
# # =======

# 1. Single Option Portfolio
pct_VaR, abs_VaR = delta_gamma_VaR(S = df.underlying_price[0], delta = df.delta[0], gamma = df.gamma[0], std = std[0], cl = 95)
print(f'1-Day 95% Percentage Delta-Gamma Normal VaR of {df.underlying_asset[0]} Option = {(pct_VaR*100).round(2)}%')
print(f'1-Day 95% Absolute Delta-Gamma Normal VaR of {df.underlying_asset[0]} Option   = $ {abs_VaR}')

# 2. Multiple Option Portfolio
pct_VaR, abs_VaR = delta_gamma_VaR(S = df.underlying_price, delta = df.delta, gamma = df.gamma, std = std, cl = 95)
print(f'1-Day 95% Percentage Delta-Gamma Normal VaR of Option Portfolio = {pct_VaR*100}%')
print(f'1-Day 95% Absolute Delta-Gamma Normal VaR of Option Portfolio   = $ {abs_VaR}')
print()

# # ===========================================================================================================================================

# NON-PARAMETRIC APPROACH: OPTION PORTFOLIO

# CORNISH-FISHER VaR
def cornish_fisher_VaR(S, delta, gamma, covMatrix, cl):
    """Function for Determining Value-at-Risk of Option Portfolio according Cornish-Fisher Approach

    Args:
        S (List/ Series/ Array, Float)      : Underlying Price.
        delta (List/ Series/ Array, Float)  : Delta of Option.
        gamma (List/ Series/ Array, Float)  : Gamma of Option.
        covMatrix (DataFrame, Float)        : Variance-Covariance of Risk Factor (In this Case: Underlying Return)
        cl (List/ Series/ Array, Integer)   : Confidence Level (Preferred: 90, 95, 99).

    Returns:
        pct_VaR (Float): Percentage Value-at-Risk.
        abs_VaR (Float): Absolute Value-at-Risk.
    """
     # Determine Alpha of Confidence Level
    alpha = 1 - cl/100
    
    # Convert S to Array in Case They are Input as List or Series
    S       = np.array(S)
    
    # Stack S Vertically with an Array consists of Ones to Determine the Percentage VaR
    S       = np.vstack([S/S, S])
    
    # Create a Diagonal Matrix of Gamma
    gamma   = np.diag(gamma)
    
    # Create an Empty List for Storing Percentage and Absolute VaR
    VaR_list = []
    
    # Iterate S to Determine Percentage and Absolute VaR
    for i in range(len(S)):
        
        delta   = S[i] * delta
        gamma   = S[i]**2 * gamma
        V       = covMatrix
        
        # Parameters
        myu_1 = 0.5 * np.trace(np.dot(gamma, V))
        myu_2 = np.dot(np.dot(delta.T, V), delta) + 0.5 * (np.trace(np.dot(gamma, V)))**2
        myu_3 = 3 * np.dot(np.dot(np.dot(np.dot(delta.T, V), gamma), V), delta) + (np.trace(np.dot(gamma, V)))**3
        myu_4 = 12 * np.dot(np.dot(np.dot(delta.T, V), (np.dot(gamma, V)**2)), delta) + 3 * (np.trace(np.dot(gamma, V)))**4 + 3 * myu_2**2
        
        rho_3 = myu_3 / myu_2**1.5
        rho_4 = (myu_4 / myu_2**2) - 3
        
        # Critical Value of Z
        z_a = norm.ppf(alpha)
        
        # Alpha percentile of N(0,1) Distribution
        za = z_a + 1/6 * (z_a**2 - 1) * rho_3 + 1/24 * (z_a**3 - 3 * z_a) * rho_4 - 1/36 * (2 * z_a**3 - 5 * z_a) * rho_3**2
        
        # VaR
        VaR = za * np.sqrt(myu_2) + myu_1
        
        VaR_list.append(VaR)
    
    pct_VaR = VaR_list[0]
    abs_VaR = VaR_list[1]
    
    return pct_VaR.round(4), abs_VaR.round(2)

covMatrix = dataPrep(tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ'],
                            start = '2020-09-20',
                            end = '2023-09-20')[3]

# # =======
# # EXAMPLE
# # =======

# Multiple Option Portfolio
pct_VaR, abs_VaR = cornish_fisher_VaR(S = df.underlying_price, delta = df.delta, gamma = df.gamma, covMatrix = covMatrix, cl = 95)
print(f'1-Day 95% Percentage Cornish-Fisher VaR of Option Portfolio = {pct_VaR*100}%')
print(f'1-Day 95% Absolute Cornish-Fisher VaR of Option Portfolio   = $ {abs_VaR}')
print()

# MONTE CARLO SIMULATION

def MC_Simulation(mu, std, T, n, nsims, S0):
    """Stock Price using Monte Carlo Simulation
    Args:
        mu (Float)      : Mean Return of Stock Price
        std (Float)     : Standard Deviation of Stock Price
        T (Integer)     : Time Horizon
        n (Integer)     : Time Step(s)
        nsims (Integer) : Number of Simulation
        S0 (Float)      : Initial Price

    Returns:
        St      : Process of Stock Price
        returns : Returns of Stock Price
    """
    # Determine delta t
    dt = T/n

    # Stock Price
    St = np.exp(
        (mu - std ** 2 / 2) * dt + std * np.random.normal(0, np.sqrt(dt), size = (nsims,n)).T
    )
    
    # Stack Vertically np.array([1]) for Storing the Initial Price of Stock
    St = np.vstack([np.ones(nsims), St])
    
    # Store Initial Price of Stock in the First Row, then Cumulate Vertically Obtaining Stock Price Process.
    St = S0 * St.cumprod(axis = 0)
    
    # Returns of the Stock Price
    returns = np.diff(St, axis = 0) / St[:-1:]
    
    return St, returns

# Data Preparation: Returns, Expected Return, Standard Deviation, Variance Covariance Matrix of Asset in Portfolio

returns, e, std, V = dataPrep(tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ'],
                              start = '2020-09-20',
                              end = '2023-09-20')

# Portfolio Weight: Assume equal weight
weights = np.array([1/len(e)]* len(e))

# Mean Return of the Portfolio
eP = np.dot(weights.T, e)

# Variance of the Portfolio
varP = np.dot(np.dot(weights.T, V), weights)

# Standard Deviation of the Portfolio
stdP = np.sqrt(varP)

St, St_returns = MC_Simulation(mu = eP,         # Mean Return of Portfolio
                               std = stdP,      # Standard Deviation of Portfolio
                               T = 1,           # Time Horizon: 1 year
                               n = 252,         # Time Steps: 252 trading days
                               nsims = 5000,    # Number of Simulations
                               S0 = 1000000     # Initial Portfolio Value
                               )

# plt.plot(St)
# plt.xlabel('Time')
# plt.ylabel('Portfolio Value')
# plt.title('Monte Carlo Simulation of Portfolio Value')
# print(plt.show());

def MC_VaR(portfolio, returns, cl, t):
    """Monte Carlo VaR

    Args:
        portfolio (float)   : Portfolio Value
        returns (array)     : Portfolio Returns
        cl (integer)        : Confidence Level (90, 95, 99)
    
    Returns:
        pct_VaR (Float): Percentage Value-at-Risk.
        abs_VaR (Float): Absolute Value-at-Risk.
    """
    # Alpha
    alpha = 1 - cl/100
    
    # Alpha-quantile of Returns Distribution
    
    # Percentage VaR
    pct_VaR = np.quantile(returns, alpha)
    
    # Absolute VaR
    abs_VaR = np.quantile(returns * portfolio, alpha)
    
    return pct_VaR.round(4), abs_VaR.round(2)

# # =======
# # EXAMPLE
# # =======

pct_VaR, abs_VaR = MC_VaR(portfolio = portfolio, returns = St_returns[1], cl = 95, t = 1)
print(f'1-Day 95% Percentage Monte Carlo VaR of Portfolio   = {-1 * pct_VaR*100}%')
print(f'1-Day 95% Absolute Monte Carlo VaR of Portfolio     = $ {-1 * abs_VaR}')

