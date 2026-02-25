[README.md](https://github.com/user-attachments/files/25545930/README.1.md)
# 📈 Real Time Options Pricing & Risk Dashboard

A quantitative finance web application that prices equity options in real time using the **Black-Scholes model** and **Monte Carlo simulation**, with full Greeks calculation and live market data.

**🔗 Live Demo: [vrajvyas-options-dashboard.streamlit.app](https://vrajvyas-options-dashboard.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What It Does

Input any stock ticker, strike price, and expiration date — the app pulls live market data and instantly calculates:

- **Option price** via Black-Scholes and Monte Carlo simulation
- **The Greeks** — Delta, Gamma, Theta, Vega, Rho
- **P&L diagram** showing payoff at expiry vs today's theoretical value
- **Monte Carlo paths** — 10,000+ simulated future price trajectories
- **3D Greek surfaces** across spot price and volatility dimensions
- **Historical market data** — candlestick chart, returns distribution, rolling volatility

---

## Screenshots

| Greeks Dashboard | Monte Carlo Paths | P&L Diagram |
|---|---|---|
| Live Delta, Gamma, Theta, Vega, Rho with sensitivity charts | 10,000 simulated GBM price paths | Payoff at expiry vs theoretical curve |

---

## The Maths

### Black-Scholes Model
Prices European call/put options using the closed-form solution:

```
C = S·N(d₁) - K·e^(-rT)·N(d₂)
P = K·e^(-rT)·N(-d₂) - S·N(-d₁)

d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d₂ = d₁ - σ·√T
```

Where `S` = spot price, `K` = strike, `T` = time to expiry, `r` = risk-free rate, `σ` = volatility.

### Monte Carlo Simulation
Simulates 10,000+ future price paths using **Geometric Brownian Motion**:

```
S(t) = S₀ · exp[(r - σ²/2)·t + σ·√t·Z]   where Z ~ N(0,1)
```

The option price is the discounted average payoff across all simulated paths.

### The Greeks
| Greek | Measures | Formula |
|---|---|---|
| Δ Delta | Price sensitivity to spot | ∂V/∂S |
| Γ Gamma | Delta sensitivity to spot | ∂²V/∂S² |
| Θ Theta | Price decay per day | ∂V/∂t |
| 𝜈 Vega | Sensitivity to volatility | ∂V/∂σ |
| ρ Rho | Sensitivity to interest rates | ∂V/∂r |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `yfinance` | Live market data from Yahoo Finance |
| `numpy` | Numerical computation & Monte Carlo |
| `scipy` | Normal distribution functions |
| `plotly` | Interactive charts & 3D surfaces |
| `pandas` | Data manipulation |

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/vrajvyas/options-dashboard.git
cd options-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run options_dashboard.py
```

The app will open automatically at `http://localhost:8501`

---

## Project Structure

```
options-dashboard/
│
├── options_dashboard.py   # Main application
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Key Features

- **Live data** - automatically fetches current price and historical volatility for any ticker
- **Interactive sliders** - adjust volatility, interest rate, strike in real time
- **Model comparison** - see Black-Scholes vs Monte Carlo side by side
- **Scenario analysis** - option price matrix across spot × volatility combinations
- **Implied volatility smile** - visualises the volatility surface by strike

---

## Disclaimer

This project is for **educational purposes only** and does not constitute financial advice. Black-Scholes assumes log-normal returns, constant volatility and interest rates, and European-style exercise. Real-world options pricing involves additional complexity.

---

*Built by [Vraj-kishor Vyas](https://www.linkedin.com/in/vrajkishor-vyas/) — MEng Electronic and Computer Engineering student at the University of Nottingham*
