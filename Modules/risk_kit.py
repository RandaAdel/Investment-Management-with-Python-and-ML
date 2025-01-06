import pandas as pd 
import scipy.stats

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns 
    computes and returns a DataFrame that contains:
    wealth index
    previous peaks 
    percent drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns=wealth_index/previous_peaks-1

    return pd.DataFrame({
        "Wealth":wealth_index,
        "Peaks":previous_peaks,
        "Drawdown":drawdowns
    })

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom 
    deciles by MarketCap 
    """
    me_m = pd.read_csv("C:/Users/OMAR/Desktop/Randa/Investment-Management-with-Python-and-ML/data/Portfolios_Formed_on_ME_monthly_EW.csv"
                                , header=0
                                , index_col=0 
                                , parse_dates=True
                                , na_values=-99.99)
    
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets

def get_hfi_returns():
    """
    Load the EDHEC Hedge Fund Index Return
    deciles by MarketCap 
    """
    hfi = pd.read_csv("C:/Users/OMAR/Desktop/Randa/Investment-Management-with-Python-and-ML/data/edhec-hedgefundindices.csv"
                                , header=0
                                , index_col=0 
                                , parse_dates=True
                                , na_values=-99.99)
    
    hfi = hfi/100
    hfi.index = hfi.index.to_period("M")
    return hfi


def skewness(r):
    """
    Compute skewness of supplied series 
    """
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Compute kurtosis of supplied series 
    """
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01): 
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level 

