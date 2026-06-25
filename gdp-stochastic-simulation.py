import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_gdp_simulation(n_countries: int = 100, num_years: int = 51) -> pd.DataFrame:
    """Simulates GDP growth paths for multiple countries using stochastic modeling."""
    np.random.seed(42)
    years = np.arange(2024, 2024 + num_years)
    
    # Start countries at different initial GDPs (Log-normal distribution)
    start_gdp = np.random.lognormal(mean=10, sigma=1.0, size=n_countries)
    # Annual growth rates with built-in volatility
    growth_rates = np.random.normal(loc=0.03, scale=0.02, size=(num_years, n_countries))
    
    # Vectorized compounding: GDP_t = GDP_0 * product(1 + growth)
    gdp_paths = start_gdp * np.cumprod(1 + growth_rates, axis=0)
    
    countries = [f"Country_{i}" for i in range(n_countries)]
    return pd.DataFrame(gdp_paths, index=years, columns=countries)

def plot_economic_projections(df: pd.DataFrame):
    """Generates a side-by-side analysis of growth trends and distribution shifts."""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    top_performers = df.iloc[-1].nlargest(5).index

    # Plot A: The Spaghetti Plot (with focus on top 5 outliers)
    for col in df.columns:
        if col in top_performers:
            ax1.plot(df.index, df[col], alpha=0.9, lw=2.5, label=col)
        else:
            ax1.plot(df.index, df[col], alpha=0.1, lw=0.6, color='gray')

    ax1.set_title("50-Year Economic Projections (Top 5 Highlighted)", fontsize=14, pad=15)
    ax1.set_ylabel("GDP (Trillions $)")
    ax1.set_yscale('log')  
    ax1.grid(True, which='both', linestyle='--', alpha=0.2)
    ax1.legend(loc='upper left')

    # Plot B: The Distribution Evolution (Log scale bins for better spread view)
    bins = np.logspace(np.log10(df.min().min()), np.log10(df.max().max()), 20)
    ax2.hist(df.iloc[0], bins=bins, alpha=0.5, label="2024 (Start)", color='cyan')
    ax2.hist(df.iloc[-1], bins=bins, alpha=0.5, label="2074 (End)", color='magenta')
    
    ax2.set_title("Wealth Distribution Shift", fontsize=14, pad=15)
    ax2.set_xlabel("GDP Value (Log Scale)")
    ax2.set_xscale('log')
    ax2.grid(True, which='both', linestyle='--', alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    gdp_data = generate_gdp_simulation()
    plot_economic_projections(gdp_data)
