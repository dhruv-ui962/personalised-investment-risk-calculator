# Generated from: Untitled5.ipynb
# Converted at: 2025-12-31T11:04:36.893Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# mask the kmeans warning
import os
os.environ["OMP_NUM_THREADS"] = "1"


# making sure user input is one of the options provided

def get_valid_choice(prompt, min_val, max_val):
    while True:
        try:
            choice = int(input(prompt))
            if min_val <= choice <= max_val:
                return choice
            else:
                print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_yes_no(prompt):
    while True:
        ans = input(prompt).lower().strip()
        if ans in ['y', 'n']:
            return ans
        print("Please enter 'y' or 'n'.")

def get_mcq(prompt, options):
    options = [o.lower() for o in options]
    while True:
        ans = input(prompt).lower().strip()
        if ans in options:
            return ans
        print(f"Please choose one of {options}.")

def get_number(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(prompt))
            if (min_val is not None and val < min_val) or \
               (max_val is not None and val > max_val):
                print(f"Enter a value between {min_val} and {max_val}.")
            else:
                return val
        except ValueError:
            print("Please enter a valid number.")

def get_currency(prompt):
    while True:
        try:
            return float(input(prompt).replace(",", ""))
        except ValueError:
            print("Please enter a valid amount.")


# seed
np.random.seed(42)

# section1:
# database
# constants set for inflation etc
"""
 Assumptions:
- Inflation: 5% (Indian Long-term Average)
- Risk-Free Rate: 6.5% (India 10Y G-Sec Yield)
- Student t distribution (more accurate than a probability distribution for volatility simulations) df=5 
"""
yearly_inflation = 0.05   
safe_yield = 0.065  
tails_df = 5                  

asset_database = {
    "NIFTY 50 Index": {"mu": 0.00045, "sigma": 0.012}, 
    "Midcap Equity": {"mu": 0.00060, "sigma": 0.018},
    "Smallcap Equity": {"mu": 0.00075, "sigma": 0.022},
    "Government Bonds": {"mu": 0.00025, "sigma": 0.004},
    "Gold": {"mu": 0.00035, "sigma": 0.010},
    "Fixed Deposits": {"mu": 0.00022, "sigma": 0.002}
}

asset_options = list(asset_database.keys())

# inputs for asset choice
print("\n INVESTMENT RISK CALCULATOR ")

# select choice of asset
print("\n Available Asset choices:")
for i, name in enumerate(asset_options, 1):
    print(f"{i}. {name}")

user_pick = get_valid_choice(
    "\nSelect the Asset you want to buy (Number from the list): ",
    1, len(asset_options)
)
main_asset = asset_options[user_pick - 1]

# select asset to compare as benchmark
print(f"\n Select a Benchmark to compare '{main_asset}' with:")
for i, name in enumerate(asset_options, 1):
    print(f"{i}. {name}")

bench_pick = get_valid_choice(
    "\nSelect a benchmark asset to compare with your primary choice : ",
    1, len(asset_options)
)
comparison_asset = asset_options[bench_pick - 1]

if main_asset == comparison_asset:
    print("\n  You selected the same asset for both, so the comparision will be identical and non-satisfactory.")

# financial inputs
print("\n Financial Details:")
time_span = int(get_number("Investment period in years: ", 1, 60))
min_return_goal = get_number("Target Annual  Return % : ", 0, 100) / 100
yearly_pay = get_currency("Annual Income (₹): ")
total_bank_balance = get_currency("Total Savings (₹): ")
current_age = int(get_number("Age: ", 18, 100))

#  risk profile assess-6qs 
print("\n RISK PROFILE QUESTIONNAIRE")
risk_points = 0

q1 = get_mcq(
    "1. If your portfolio dropped 20% in a month, would you (a) Sell all, (b) Hold, (c) Buy more? ",
    ['a', 'b', 'c']
)
risk_points += 0 if q1 == 'a' else 1 if q1 == 'b' else 2

q2 = get_mcq(
    "2. What is your primary goal? (a)  Safety, (b) Balanced , (c) Maximising Wealth? ",
    ['a', 'b', 'c']
)
risk_points += 0 if q2 == 'a' else 1 if q2 == 'b' else 2

q3 = get_mcq(
    "3. How stable is your monthly income? (a) Volatile, (b) Steady, (c) Very Secure? ",
    ['a', 'b', 'c']
)
risk_points += 0 if q3 == 'a' else 1 if q3 == 'b' else 2

q4 = get_yes_no("4. Do you have an emergency fund for 6+ months? (y/n): ")
risk_points += 2 if q4 == 'y' else 0

q5 = get_yes_no("5. Have you invested in stocks/mutual funds before? (y/n): ")
risk_points += 1 if q5 == 'y' else 0

q6 = int(get_number(
    "6. On a scale of 1-5, how much do you worry about inflation? (1-Low, 5-High): ",
    1, 5
))
risk_points += (q6 // 2)

# Calculate a multiplier based on points
stomach_for_risk = 0.6 + (risk_points / 15)

#  simulate paths and daily+ annual returns using monte carlo simulation
def run_monte_carlo(profit_avg, volatility, total_years):
    trading_days = total_years * 252
    iterations = 5000
    
    # daily
    daily_changes = t.rvs(tails_df, loc=profit_avg, scale=volatility, size=(iterations, trading_days))
    # -ve asset value issue fixed!!!!!!!
    daily_changes = np.maximum(daily_changes, -0.999) 
    
    growth_curves = np.cumprod(1 + daily_changes, axis=1) 
    
    # annual
    closing_values = growth_curves[:, -1]
    raw_cagr = (closing_values**(1/total_years)) - 1
    inflation_adj_cagr = ((1 + raw_cagr) / (1 + yearly_inflation)) - 1
    
    # sharpe ratio part to calculate efficiency
    excess_profit = daily_changes - (safe_yield / 252)
    reward_to_risk = (np.mean(excess_profit) / np.std(excess_profit)) * np.sqrt(252)
    
    # paths simulation
    high_water_mark = np.maximum.accumulate(growth_curves, axis=1)
    crash_depth = ((growth_curves - high_water_mark) / high_water_mark).min()
    
    return inflation_adj_cagr, reward_to_risk, crash_depth, growth_curves

# simulation 1   (primary)
main_real_rets, main_efficiency, main_max_loss, main_history = run_monte_carlo(
    asset_database[main_asset]["mu"], 
    asset_database[main_asset]["sigma"], 
    time_span
)

# simulation 2  (benchmark)
bench_real_rets, bench_efficiency, bench_max_loss, _ = run_monte_carlo(
    asset_database[comparison_asset]["mu"], 
    asset_database[comparison_asset]["sigma"], 
    time_span
)

# fixed variables
findings = []
win_rate = np.mean(main_real_rets >= min_return_goal)

# risk capacity using kmeans
user_features = np.array([
    min(total_bank_balance / yearly_pay, 3) / 3,   
    time_span / 30,                                
    stomach_for_risk,                              
    1 - (current_age / 75),                        
    1 if q4 == 'y' else 0                          
]).reshape(1, -1)

# sample p
reference_users = np.array([
    [0.2, 0.2, 0.4, 0.3, 0],   
    [0.4, 0.4, 0.5, 0.5, 0],
    [0.6, 0.6, 0.6, 0.6, 1],   
    [0.8, 0.8, 0.7, 0.7, 1],
    [1.0, 0.9, 0.8, 0.8, 1]    
])

# standardization
scaler = StandardScaler()
scaled_reference = scaler.fit_transform(reference_users)
scaled_user = scaler.transform(user_features)

# k-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_reference)
user_cluster = kmeans.predict(scaled_user)[0]

#  multipliers
cluster_risk_map = {
    0: 0.30,   # Risk-constrained
    1: 0.55,   # Balanced
    2: 0.80    # Risk-absorbing
}

risk_tolerance_score = cluster_risk_map[user_cluster]

# whether simulated return matches the goal
if win_rate < 0.4:
    findings.append(f"  Only {win_rate:.1%} chance of hitting your {min_return_goal:.1%} target.")
elif win_rate > 0.75:
    findings.append(f"  High chances ({win_rate:.1%}) of meeting your target.")

# benchmark Comparison
if main_efficiency < bench_efficiency:
    findings.append(f"  {comparison_asset} has a higher chance of achieving better returns that have been adjusted for risks like inflation and volatility.")
else:
    findings.append(f" Your primary chosen asset outperforms {comparison_asset} based on our simulated returns.")

# check for potential crashes + inflation risk
if abs(main_max_loss) > 0.30 and risk_tolerance_score < 0.5:
    findings.append("  VOLATILITY: the potential risks of crash exceeds your financial capacity.")

if np.mean(main_real_rets) < 0.01:
    findings.append(" INFLATION : Expected real returns are likely to fall considerably due to inflation.")

# Final Decision-IMP
final_verdict = "INVEST" if (win_rate > 0.5 and risk_tolerance_score > 0.35) else "DO NOT INVEST"

# if decision to invest: amount to invest???
recommended_amt = 0
if final_verdict == "INVEST":
    recommended_amt = total_bank_balance * risk_tolerance_score * 0.5

#   SUMMARY 
print(f"Investment simulation summary")
print(f"Primary Asset:   {main_asset}")
print(f"Benchmark Asset: {comparison_asset}")

header_main = f"Your Asset ({main_asset[:10]}..)"
header_bench = f"Benchmark ({comparison_asset[:10]}..)"

print(f"{'Metric':<28} | {header_main:<20} | {header_bench:<20}")
print(f"{'Expected Real Return (CAGR)':<28} | {np.mean(main_real_rets):.2%}              | {np.mean(bench_real_rets):.2%}")
print(f"{'Effficiency ':<28} | {main_efficiency:.2f}                | {bench_efficiency:.2f}")
print(f"{'Crash Risk ':<28} | {main_max_loss:.2%}              | {bench_max_loss:.2%}")
print(f"{'Probability of Achieving Target':<28} | {win_rate:.2%}              | {np.mean(bench_real_rets >= min_return_goal):.2%}")

print(f"\nFINAL DECISION: {final_verdict}")

if final_verdict == "INVEST":
    print(f"RECOMMENDED CAPITAL AMOUNT: ₹{recommended_amt:,.2f} (approx {recommended_amt/total_bank_balance:.0%} of savings)")

print("\n INSIGHTS:")
for note in findings:
    print(note)

# limitations of this program
print("\nTHINGS TO CONSIDER: ")
print(" HISTORICAL BIAS: This calculator uses past and standard return metrics (Mean/Sigma).")
print(" SIMULATION MODEL: We use a Student-t distribution to predict mark crashes.")
print(" TAXES & FEES: Returns are 'Pre-Tax'. In India, Equity gains >1L are taxed at 12.5% (LTCG).")

#  graphs
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(main_real_rets, bins=40, alpha=0.6, label=main_asset, density=True)
plt.hist(bench_real_rets, bins=40, alpha=0.4, label=comparison_asset, density=True)
plt.axvline(min_return_goal, linestyle='--', linewidth=2)
plt.title("Real Return Distribution")
plt.xlabel("Annual Real Return")
plt.ylabel("Probability Density")
plt.legend()

plt.subplot(1, 2, 2)
middle_path_main = np.median(main_history, axis=0)
simulated_bench_data = t.rvs(
    tails_df,
    loc=asset_database[comparison_asset]["mu"],
    scale=asset_database[comparison_asset]["sigma"],
    size=(100, time_span * 252)
)
middle_path_bench = np.median(np.cumprod(1 + simulated_bench_data, axis=1), axis=0)

plt.plot(middle_path_main, linewidth=2, label=main_asset)
plt.plot(middle_path_bench, linewidth=2, label=comparison_asset)
plt.title(f"Growth Projector ({time_span} Years)")
plt.xlabel("Days")
plt.ylabel("Multiplier")
plt.legend()

plt.tight_layout()
plt.show()