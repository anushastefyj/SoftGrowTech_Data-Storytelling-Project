# ============================================================
#  DATA STORYTELLING — Data Analysis (Sample Sales Dataset)
# ============================================================
#  pip install pandas numpy scipy tabulate

import pandas as pd
import numpy as np
from scipy import stats
from io import StringIO

# ─────────────────────────────────────────
# 1. BUILT-IN SAMPLE DATA
# ─────────────────────────────────────────
RAW = """date,region,category,orders,revenue,discount_pct
2023-01-05,North,Electronics,45,135000,5
2023-01-12,South,Clothing,78,31200,10
2023-01-20,East,Grocery,210,42000,2
2023-02-03,North,Electronics,52,156000,5
2023-02-14,South,Clothing,65,26000,15
2023-02-22,East,Grocery,198,39600,2
2023-03-08,North,Electronics,60,180000,0
2023-03-15,South,Clothing,90,36000,10
2023-03-25,East,Grocery,220,44000,2
2023-04-10,North,Electronics,40,120000,10
2023-04-18,South,Clothing,55,22000,20
2023-04-28,East,Grocery,185,37000,3
2023-05-05,North,Electronics,70,210000,0
2023-05-12,South,Clothing,95,38000,5
2023-05-22,East,Grocery,230,46000,2
2023-06-08,North,Electronics,55,165000,5
2023-06-16,South,Clothing,70,28000,10
2023-06-25,East,Grocery,215,43000,2
2023-07-04,North,Electronics,48,144000,8
2023-07-14,South,Clothing,60,24000,15
2023-07-24,East,Grocery,200,40000,3
2023-08-02,North,Electronics,35,105000,12
2023-08-10,South,Clothing,50,20000,20
2023-08-20,East,Grocery,190,38000,3
2023-09-06,North,Electronics,62,186000,3
2023-09-15,South,Clothing,88,35200,8
2023-09-25,East,Grocery,225,45000,2
2023-10-10,North,Electronics,75,225000,0
2023-10-18,South,Clothing,100,40000,5
2023-10-28,East,Grocery,245,49000,1
2023-11-05,North,Electronics,90,270000,0
2023-11-11,South,Clothing,130,52000,0
2023-11-22,East,Grocery,260,52000,1
2023-12-08,North,Electronics,110,330000,0
2023-12-15,South,Clothing,160,64000,0
2023-12-24,East,Grocery,280,56000,0
"""

# ─────────────────────────────────────────
# 2. LOAD & INSPECT
# ─────────────────────────────────────────
df = pd.read_csv(StringIO(RAW), parse_dates=['date'])

print("=" * 55)
print("  STEP 1 — LOAD & INSPECT")
print("=" * 55)
print(f"\nShape      : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Date range : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"\nColumn types:\n{df.dtypes.to_string()}")
print(f"\nMissing values:\n{df.isnull().sum().to_string()}")
print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")

# ─────────────────────────────────────────
# 3. CLEAN & ENGINEER FEATURES
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 2 — CLEAN & FEATURE ENGINEERING")
print("=" * 55)

df['month']          = df['date'].dt.to_period('M')
df['quarter']        = df['date'].dt.to_period('Q')
df['month_name']     = df['date'].dt.strftime('%b')
df['rev_per_order']  = (df['revenue'] / df['orders']).round(2)
df['discount_amt']   = (df['revenue'] * df['discount_pct'] / 100).round(2)
df['net_revenue']    = (df['revenue'] - df['discount_amt']).round(2)

print("\nEngineered columns added:")
print("  rev_per_order  — average revenue per order")
print("  discount_amt   — absolute discount given")
print("  net_revenue    — revenue after discount")
print(f"\nSample (first 3 rows):\n{df[['date','category','revenue','rev_per_order','net_revenue']].head(3).to_string(index=False)}")

# ─────────────────────────────────────────
# 4. SUMMARY STATISTICS
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 3 — SUMMARY STATISTICS")
print("=" * 55)

total_rev   = df['revenue'].sum()
total_net   = df['net_revenue'].sum()
total_ord   = df['orders'].sum()
avg_monthly = df.groupby('month')['revenue'].sum().mean()

print(f"\n  Total gross revenue : ₹{total_rev:,.0f}")
print(f"  Total net revenue   : ₹{total_net:,.0f}")
print(f"  Total orders        : {total_ord:,}")
print(f"  Avg monthly revenue : ₹{avg_monthly:,.0f}")
print(f"  Avg rev per order   : ₹{df['rev_per_order'].mean():,.0f}")

print(f"\nDescriptive stats (revenue):\n{df['revenue'].describe().round(0).to_string()}")

# ─────────────────────────────────────────
# 5. MONTHLY TREND
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 4 — MONTHLY REVENUE TREND")
print("=" * 55)

monthly = (
    df.groupby('month')
    .agg(revenue=('revenue', 'sum'), orders=('orders', 'sum'))
    .reset_index()
)
monthly['mom_growth'] = monthly['revenue'].pct_change().mul(100).round(1)

print(f"\n{'Month':<10} {'Revenue':>12} {'Orders':>8} {'MoM %':>8}")
print("-" * 42)
for _, row in monthly.iterrows():
    growth = f"{row['mom_growth']:+.1f}%" if pd.notna(row['mom_growth']) else "  —"
    print(f"{str(row['month']):<10} ₹{row['revenue']:>10,.0f} {row['orders']:>8,} {growth:>8}")

best_month  = monthly.loc[monthly['revenue'].idxmax()]
worst_month = monthly.loc[monthly['revenue'].idxmin()]
print(f"\n  Best month  : {best_month['month']} — ₹{best_month['revenue']:,.0f}")
print(f"  Worst month : {worst_month['month']} — ₹{worst_month['revenue']:,.0f}")

# ─────────────────────────────────────────
# 6. CATEGORY BREAKDOWN
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 5 — CATEGORY BREAKDOWN")
print("=" * 55)

by_cat = (
    df.groupby('category')
    .agg(
        total_revenue=('revenue', 'sum'),
        total_orders=('orders', 'sum'),
        avg_rev_per_order=('rev_per_order', 'mean'),
        avg_discount=('discount_pct', 'mean'),
    )
    .sort_values('total_revenue', ascending=False)
)
by_cat['revenue_share_%'] = (by_cat['total_revenue'] / total_rev * 100).round(1)

print(f"\n{by_cat.round(1).to_string()}")

# ─────────────────────────────────────────
# 7. REGIONAL BREAKDOWN
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 6 — REGIONAL BREAKDOWN")
print("=" * 55)

by_region = (
    df.groupby('region')
    .agg(
        total_revenue=('revenue', 'sum'),
        total_orders=('orders', 'sum'),
        net_revenue=('net_revenue', 'sum'),
    )
    .sort_values('total_revenue', ascending=False)
)
by_region['revenue_share_%'] = (by_region['total_revenue'] / total_rev * 100).round(1)
print(f"\n{by_region.to_string()}")

# ─────────────────────────────────────────
# 8. QUARTERLY COMPARISON
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 7 — QUARTERLY COMPARISON")
print("=" * 55)

by_q = (
    df.groupby('quarter')['revenue']
    .sum()
    .reset_index()
)
by_q['qoq_growth'] = by_q['revenue'].pct_change().mul(100).round(1)
by_q['share_%']    = (by_q['revenue'] / total_rev * 100).round(1)

print(f"\n{'Quarter':<10} {'Revenue':>12} {'QoQ %':>8} {'Share':>8}")
print("-" * 42)
for _, row in by_q.iterrows():
    growth = f"{row['qoq_growth']:+.1f}%" if pd.notna(row['qoq_growth']) else "  —"
    print(f"{str(row['quarter']):<10} ₹{row['revenue']:>10,.0f} {growth:>8} {row['share_%']:>7.1f}%")

# ─────────────────────────────────────────
# 9. CORRELATION ANALYSIS
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 8 — CORRELATION ANALYSIS")
print("=" * 55)

corr_cols = ['orders', 'revenue', 'discount_pct', 'rev_per_order']
corr_matrix = df[corr_cols].corr().round(2)
print(f"\nCorrelation matrix:\n{corr_matrix.to_string()}")

r, p = stats.pearsonr(df['discount_pct'], df['revenue'])
print(f"\n  Discount % vs Revenue → r = {r:.3f}, p = {p:.4f}")
print(f"  Interpretation: {'negative' if r < 0 else 'positive'} correlation, "
      f"{'statistically significant' if p < 0.05 else 'not significant'} (α=0.05)")

# ─────────────────────────────────────────
# 10. KEY INSIGHTS SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 9 — KEY INSIGHTS SUMMARY")
print("=" * 55)

top_cat    = by_cat['total_revenue'].idxmax()
top_region = by_region['total_revenue'].idxmax()
q4_rev     = by_q[by_q['quarter'].astype(str).str.contains('Q4')]['revenue'].values[0]
q1_rev     = by_q[by_q['quarter'].astype(str).str.contains('Q1')]['revenue'].values[0]

print(f"""
  1. Electronics dominates revenue ({by_cat.loc['Electronics','revenue_share_%']}% share)
     with ₹{by_cat.loc['Electronics','total_revenue']:,.0f} gross revenue across the year.

  2. North region leads with {by_region.loc['North','revenue_share_%']}% of total revenue,
     driven entirely by its high Electronics sales.

  3. Q4 is the strongest quarter — ₹{q4_rev:,.0f} revenue,
     {((q4_rev/q1_rev - 1)*100):.0f}% higher than Q1 (₹{q1_rev:,.0f}).

  4. Discount has a {'negative' if r < 0 else 'positive'} correlation (r={r:.2f}) with revenue,
     suggesting higher discounts are used in slower months, not as a revenue driver.

  5. August is the worst-performing month across all categories —
     worth investigating for seasonality or supply issues.
""")

print("=" * 55)
print("  Analysis complete. Ready for visualisation!")
print("=" * 55)

# ─────────────────────────────────────────
# 11. DATA VISUALIZATION
# ─────────────────────────────────────────

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ============================================================
# 1. MONTHLY REVENUE TREND
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    monthly['month'].astype(str),
    monthly['revenue'],
    marker='o',
    linewidth=2
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("monthly_revenue_trend.png")

plt.show()


# ============================================================
# 2. CATEGORY-WISE REVENUE
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x=by_cat.index,
    y=by_cat['total_revenue']
)

plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Total Revenue")

plt.tight_layout()

plt.savefig("category_revenue.png")

plt.show()


# ============================================================
# 3. REGIONAL REVENUE SHARE
# ============================================================

plt.figure(figsize=(7,7))

plt.pie(
    by_region['total_revenue'],
    labels=by_region.index,
    autopct='%1.1f%%'
)

plt.title("Regional Revenue Share")

plt.savefig("regional_share.png")

plt.show()


# ============================================================
# 4. QUARTERLY REVENUE COMPARISON
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x=by_q['quarter'].astype(str),
    y=by_q['revenue']
)

plt.title("Quarterly Revenue Comparison")
plt.xlabel("Quarter")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig("quarterly_revenue.png")

plt.show()


# ============================================================
# 5. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(8,5))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()


# ============================================================
# 6. ORDERS VS REVENUE
# ============================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x=df['orders'],
    y=df['revenue'],
    hue=df['category'],
    s=100
)

plt.title("Orders vs Revenue")
plt.xlabel("Orders")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig("orders_vs_revenue.png")

plt.show()
