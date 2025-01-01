import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from typing import Optional, Tuple, Dict
from data_preprocessing import save_to_pkl, load_from_pkl

def construct_eigenportfolios(pca_results: Dict,
                            returns_data: pd.DataFrame, 
                            save_prefix: Optional[str] = None) -> Dict:
    """
    Construct eigenportfolios with adjustments for volatility and normalized weights
    using rolling window PCA results.
    
    Parameters
    ----------
    pca_results : Dict
        Dictionary containing PCA results for each window
    returns_data : pd.DataFrame
        DataFrame of daily stock returns
    save_prefix : str, optional
        If provided, saves the eigenportfolios with this prefix
        
    Returns
    -------
    dict
        Dictionary containing:
        - dates: np.ndarray of window dates
        - eigenvectors: array of shape (n_dates, n_stocks, max_components)
        - eigenweights: array of shape (n_dates, n_stocks, max_components)
    """
    # Get dimensions
    dates = np.array(sorted([np.datetime64(date, 'D') for date in pca_results.keys()]))
    n_dates = len(dates)
    n_stocks = returns_data.shape[1]
    
    # Find the maximum number of components across all windows
    max_components = max(window_pca['principal_components'].shape[0] 
                        for window_pca in pca_results.values())
    
    # Initialize arrays with max_components
    eigenvectors_array = np.zeros((n_dates, n_stocks, max_components))
    eigenweights_array = np.zeros((n_dates, n_stocks, max_components))
    
    # Process each window's PCA results
    for i, date in enumerate(dates):
        window_pca = pca_results[date]
        principal_components = window_pca['principal_components']
        n_components = principal_components.shape[0]  # Actual number of components in this window
        
        # Get the window of returns data corresponding to this date
        window_returns = returns_data.loc[:np.datetime_as_string(date, unit='D')].tail(252)
        
        # Calculate standard deviations of stock returns for volatility adjustment
        stock_volatilities = window_returns.std()
        
        # Compute volatility-adjusted weights
        volatility_adjusted_weights = principal_components / stock_volatilities.values.reshape(1, -1)
        
        # Normalize weights to ensure they sum to 1 for each eigenportfolio
        normalized_weights = volatility_adjusted_weights / np.abs(volatility_adjusted_weights).sum(axis=1)[:, np.newaxis]
        
        # Store results in arrays (only up to n_components)
        eigenvectors_array[i, :, :n_components] = principal_components.T
        eigenweights_array[i, :, :n_components] = normalized_weights.T
    
    # Create results dictionary
    results = {
        'dates': dates,
        'eigenvectors': eigenvectors_array,
        'eigenweights': eigenweights_array
    }
    
    # Save results if prefix is provided
    if save_prefix:
        try:
            save_to_pkl(results, f'eigenportfolios_{save_prefix}.pkl')
        except Exception as e:
            print(f"Error saving eigenportfolios: {e}")
        
    return results

def plot_eigenportfolio_weights(eigenportfolio: dict,
                              stock_tickers: list,
                              target_date: Optional[np.datetime64] = None,
                              n_components: Optional[int] = None) -> None:
    """
    Visualize the eigenportfolio weights for selected components at a specific date.
    
    Parameters
    ----------
    eigenportfolio : dict
        Dictionary containing dates, eigenvectors, and eigenportfolios
    stock_tickers : list
        List of stock tickers
    target_date : numpy.datetime64, optional
        The specific date to plot. If None, uses the first date.
    n_components : int, optional
        Number of components to plot. If None, plots all components.
    """
    dates = eigenportfolio['dates']
    eigenportfolios_array = eigenportfolio['eigenweights']
    
    if target_date is None:
        date_idx = 0
        target_date = dates[0]
    else:
        try:
            date_idx = np.where(dates == target_date)[0][0]
        except IndexError:
            raise ValueError(f"Date {target_date} not found in the results.")
    
    weights = eigenportfolios_array[date_idx]
    total_components = weights.shape[1]
    
    # Determine number of components to plot
    if n_components is None:
        n_components = total_components
    else:
        n_components = min(n_components, total_components)
    
    # Create subplots only for selected components
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 6 * n_components))
    if n_components == 1:
        axes = [axes]
        
    for idx in range(n_components):
        component_weights = weights[:, idx]
        weights_series = pd.Series(component_weights, index=stock_tickers)
        sorted_weights = weights_series.sort_values(ascending=False)
        
        sorted_weights.plot(
            kind='line',
            ax=axes[idx],
            color='blue',
            linestyle='-',
            linewidth=2,
            alpha=1
        )
        axes[idx].axhline(y=0, color='black', linewidth=2, linestyle='--')
        axes[idx].set_title(f'Eigenvector {idx+1} Sorted by Coefficient Size')
        axes[idx].set_xlabel('Stocks')
        axes[idx].set_ylabel('Weight')
        axes[idx].set_xticks(range(len(sorted_weights.index)))
        axes[idx].set_xticklabels(sorted_weights.index, rotation=90)
    
    plt.suptitle(f'Eigenvector Sorted by Coefficient Size\nDate: {target_date}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def compute_eigenportfolio_returns(eigenportfolio_results: Dict,
                                 returns_data: pd.DataFrame,
                                 save_prefix: Optional[str] = None) -> Dict:
    """
    Calculate daily returns for each eigenportfolio using the formula:
    Fjt = Σ(Qijt * Rit) where:
    - Fjt is the return of eigenportfolio j on day t
    - Qijt is the weight of stock i in eigenportfolio j on day t
    - Rit is the return of stock i on day t
    
    Parameters
    ----------
    eigenportfolio_results : Dict
        Dictionary containing eigenportfolio data:
        - dates: np.ndarray of window dates
        - eigenportfolios: array of shape (n_dates, n_stocks, max_components)
    returns_data : pd.DataFrame
        DataFrame of daily stock returns
    save_prefix : Optional[str]
        If provided, saves the results with this prefix
        
    Returns
    -------
    Dict
        Dictionary containing:
        - returns: array of shape (n_valid_dates, max_factors) with eigenportfolio returns
        - dates: array of dates corresponding to the returns
    """
    # Extract arrays and dates
    window_dates = eigenportfolio_results['dates']
    eigenweights_array = eigenportfolio_results['eigenweights']
    
    # Ensure returns_data has datetime index and sort it
    returns_data = returns_data.copy()
    returns_data.index = pd.to_datetime(returns_data.index)
    returns_data = returns_data.sort_index()
    
    # Filter returns_data to start from first window date
    valid_returns = returns_data[returns_data.index >= window_dates[0]]
    valid_dates = np.array([np.datetime64(d) for d in valid_returns.index])
    
    # Get maximum number of factors
    max_factors = eigenweights_array.shape[2]
    
    # Initialize array for eigenportfolio returns
    ep_returns = np.zeros((len(valid_returns), max_factors))
    
    # Process each window
    for i in range(len(window_dates)-1):
        # Get dates in this window
        mask = (valid_dates >= window_dates[i]) & (valid_dates < window_dates[i+1])
        window_returns = valid_returns.loc[mask].values
        
        # Compute returns using all available components for this window
        window_weights = eigenweights_array[i]  # This includes all components
        window_ep_returns = np.dot(window_returns, window_weights)
        
        # Store in array (automatically handles varying number of components)
        ep_returns[mask] = window_ep_returns
    
    # Handle the last window
    mask = valid_dates >= window_dates[-1]
    window_returns = valid_returns.loc[mask].values
    window_ep_returns = np.dot(window_returns, eigenweights_array[-1])
    ep_returns[mask] = window_ep_returns
    
    # Create results dictionary
    results = {
        'returns': ep_returns,
        'dates': valid_dates
    }
    
    # Save if prefix provided
    if save_prefix:
        try:
            save_to_pkl(results, f'eigenportfolio_returns_{save_prefix}.pkl')
        except Exception as e:
            print(f"Error saving eigenportfolio_returns: {e}")
    
    return results

def plot_eigenportfolio_returns(
    eigenportfolio_returns_results: Dict, 
    nifty_returns: pd.Series,
    start_date: str = None,
    end_date: str = None,
    n_portfolios: int = 1
) -> None:
    """
    Plot cumulative returns of eigenportfolios alongside Nifty benchmark.
    Both start at 1.0 on the eigenportfolio start date.
    Only dates common to both series are used.
    
    Parameters
    ----------
    eigenportfolio_returns_results : Dict
        Dictionary containing:
        - returns: array of eigenportfolio returns
        - dates: array of corresponding dates
    nifty_returns : pd.Series or pd.DataFrame
        Daily returns for Nifty index with datetime index
    start_date : str, optional
        Start date for plotting in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for plotting in 'YYYY-MM-DD' format
    """
    # Extract dates and returns from results dict
    eigenportfolio_dates = eigenportfolio_returns_results['dates']
    eigenportfolio_returns = eigenportfolio_returns_results['returns']
    
    # Convert eigenportfolio returns to DataFrame with dates
    eigenportfolio_returns_df = pd.DataFrame(
        eigenportfolio_returns[:, :n_portfolios],  # Only take first n_portfolios columns
        index=eigenportfolio_dates,
        columns=['Principal Eigenportfolio' if i == 0 else f'Eigenportfolio {i+1}'
                for i in range(min(eigenportfolio_returns.shape[1], n_portfolios))]
    )


    
    # Ensure nifty returns is a Series
    if isinstance(nifty_returns, pd.DataFrame):
        nifty_returns = nifty_returns.iloc[:, 0]
    
    # Ensure nifty returns has datetime index
    nifty_returns.index = pd.to_datetime(nifty_returns.index)
    
    # Get common dates starting from eigenportfolio_start_date
    eigenportfolio_start_date = eigenportfolio_dates[0]
    common_dates = eigenportfolio_returns_df.index.intersection(nifty_returns.index)
    
    # Convert start_date and end_date to datetime if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        common_dates = common_dates[common_dates >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        common_dates = common_dates[common_dates <= end_date]
    
    # Align both series on common dates
    aligned_eigenportfolio_returns = eigenportfolio_returns_df.loc[common_dates]
    aligned_nifty = nifty_returns.loc[common_dates]
    
    # Combine returns
    combined_returns = pd.concat([
        aligned_eigenportfolio_returns, 
        pd.Series(aligned_nifty, name='Nifty 100')
    ], axis=1)
    
    # Convert to cumulative returns starting at 1
    cumulative_returns = (1 + combined_returns).cumprod() - 1
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot eigenportfolios
    for col in aligned_eigenportfolio_returns.columns:
        plt.plot(common_dates, cumulative_returns[col], label=col)
    
    # Plot Nifty
    plt.plot(common_dates, cumulative_returns['Nifty 100'], 
            label='Nifty 100', linestyle='--', linewidth=2, color='black')
 
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Load the preprocessed data
    daily_returns = load_from_pkl('daily_returns.pkl')
    volume_weighted_returns = load_from_pkl('volume_weighted_returns.pkl')
    pca_dr1=load_from_pkl('pca_dr1.pkl')
    pca_dr15=load_from_pkl('pca_dr15.pkl')
    pca_dr55=load_from_pkl('pca_dr55.pkl')
    pca_dr75=load_from_pkl('pca_dr75.pkl')
    pca_vw15=load_from_pkl('pca_vw15.pkl')
    pca_vw55=load_from_pkl('pca_vw55.pkl')
    pca_vw75=load_from_pkl('pca_vw75.pkl')
   
     # Construct and save eigenportfolios from daily returns using 1 principal component
    eigenportfolios_dr1 = construct_eigenportfolios(
        pca_dr1,
        daily_returns,
        save_prefix="dr1"
    )

    # Construct and save eigenportfolios from daily returns using 15 principal components
    eigenportfolios_dr15 = construct_eigenportfolios(
        pca_dr15,
        daily_returns,
        save_prefix="dr15"
    )


    # Construct and save eigenportfolios from daily returns using 55% variance threshold
    eigenportfolios_dr55 = construct_eigenportfolios(
        pca_dr55,
        daily_returns,
        save_prefix="dr55"
    )

    # Construct and save eigenportfolios from daily returns using 75% variance threshold
    eigenportfolios_dr75 = construct_eigenportfolios(
        pca_dr75,
        daily_returns,
        save_prefix="dr75"
    )

    # Construct and save eigenportfolios from volume-weighted returns using 15 principal components
    eigenportfolios_vw15 = construct_eigenportfolios(
        pca_vw15,
        volume_weighted_returns,
        save_prefix="vw15"
    )

    # Construct and save eigenportfolios from volume-weighted returns using 55% variance threshold
    eigenportfolios_vw55 = construct_eigenportfolios(
        pca_vw55,
        volume_weighted_returns,
        save_prefix="vw55"
    )
    # Construct and save eigenportfolios from volume-weighted returns using 75% variance threshold
    eigenportfolios_vw75 = construct_eigenportfolios(
        pca_vw75,
        volume_weighted_returns,
        save_prefix="vw75"
    )
   
    print("\nEigenportfolios constructed and saved successfully.")

    # Compute the eigenportfolio returns for daily returns with 15 components
    ep_return_dr1 = compute_eigenportfolio_returns(
        eigenportfolios_dr1, 
        daily_returns,
        save_prefix="dr1"
    )

    # Compute the eigenportfolio returns for daily returns with 15 components
    ep_return_dr15 = compute_eigenportfolio_returns(
        eigenportfolios_dr15, 
        daily_returns, 
        save_prefix="dr15"
    )
    
    # Compute the eigenportfolio returns for daily returns with variance threshold of 55%
    ep_return_dr55 = compute_eigenportfolio_returns(
        eigenportfolios_dr55, 
        daily_returns,
        save_prefix="dr55"
    )

    # Compute the eigenportfolio returns for daily returns with variance threshold of 75%
    ep_return_dr75 = compute_eigenportfolio_returns(
        eigenportfolios_dr75, 
        daily_returns, 
        save_prefix="dr75"
    )

     # Compute the eigenportfolio returns for volume_weighted_returns with 15 components
    ep_return_vw15 = compute_eigenportfolio_returns(
        eigenportfolios_vw15, 
        volume_weighted_returns, 
        save_prefix="vw15"
    )
    
    # Compute the eigenportfolio returns for volume_weighted_returns with variance threshold of 55%
    ep_return_vw55 = compute_eigenportfolio_returns(
        eigenportfolios_vw55, 
        volume_weighted_returns, 
        save_prefix="vw55"
    )

     # Compute the eigenportfolio returns for volume_weighted_returns with variance threshold of 75%
    ep_return_vw75 = compute_eigenportfolio_returns(
        eigenportfolios_vw75, 
        volume_weighted_returns,
        save_prefix="vw75"
    )

    print("\nEigenportfolio returns computed and saved successfully.")


if __name__ == "__main__":
    main()