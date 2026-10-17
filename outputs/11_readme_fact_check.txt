==========================================================================================
README FACT CHECK — every headline number recomputed from the raw CSV
==========================================================================================

[1] DATA CHECKS
Line items 51,177 | transactions 33,887 | days 181 (2026-01-01 to 2026-06-30) | branches 3 | categories 9 | products 45
Revenue $305,925.14 | gross profit $179,430.49 | margin 58.65%
Max |revenue - qty x price x (1 - discount)| = $0.005; rows with gross profit <= 0: 0
Transactions without Customer ID: 36.45%; distinct Customer IDs: 3,494
Exact duplicate rows: 132 (revenue $741.43); removing them changes revenue by 0.24% and margin by -0.000% points
Distinct prices/costs per branch-product over 181 days: max 1 (1 = never changed)

[2] GARDEN COST GAP
Gross margin: Garden 46.91%, Harbour 64.96%, Uptown 62.63%
Price Harbour/Garden across 45 products: 1.049 to 1.051
Price Uptown/Garden across 45 products: 0.958 to 0.961
Cost Garden / avg(Harbour, Uptown): mean 1.460, range 1.455 to 1.465 (all 45 products)
Example Still Water cost: Garden $0.93, Harbour $0.65, Uptown $0.62
Garden excess cost vs two-branch average at Garden's volume (17,833 units): $16,666.62 in 6 months
Garden margin at Harbour's +5% price list, same cost: 49.4%

[3] DISCOUNTS
Total discount given: $2,179.36
  15%: 504 lines (0.98% of lines), $472.08; branches ['Harbour']; categories ['Cold Coffee', 'Desserts']
  10%: 2,861 lines (5.59% of lines), $1,707.28; branches ['Garden', 'Harbour', 'Uptown']; categories all 9
  15% dates: 03-12..03-19 and 05-12..05-19 (16 days)
  Share of Harbour Cold Coffee + Desserts lines discounted 15% on those days: 100.0%
  10% rate on lines with vs without Customer ID: 5.58% vs 5.61%

[4] DEMAND PATTERNS
Average orders/day: Tuesday 284.5, other days 170.9 (ratio 1.66), Saturday 128.7
Tuesday ratio in each complete week: 1.44 to 1.81, mean 1.64 (25 weeks)
Orders/day by month: M1 146.6, M2 165.0, M3 181.5, M4 191.7, M5 207.7, M6 230.2 (Jun/Jan +57%)
Busiest hour 21:00 (4,273 orders); 20:00-22:59 share 31.6%; quietest 7:00 (724)
Share of orders from 18:00 on: 50.4%; share of gross profit from 18:00 on: 50.8%
Takeaway share before 18:00 54.0%, from 18:00 28.0%

[5] BASKETS & ADD-ONS
Orders with 2+ items: 46.37% (15,712); single-item: 18,175
  First item Matcha & Tea -> second item Bakery: 53.5%
  First item Sandwiches -> second item Cold Drinks: 26.6%
  First item Meals -> second item Cold Drinks: 26.3%
  Unit gross profit Still Water          Harbour $1.40 | Garden $1.02
  Unit gross profit Sparkling Water      Harbour $1.54 | Garden $0.82
  Unit gross profit Lemonade             Harbour $3.71 | Garden $2.98
  Unit gross profit Fresh Orange Juice   Harbour $3.92 | Garden $2.72
  Unit gross profit Chocolate Cookie     Harbour $2.40 | Garden $1.85
  Unit gross profit Butter Croissant     Harbour $2.69 | Garden $2.02
  Unit gross profit Almond Croissant     Harbour $3.18 | Garden $2.12
  Unit gross profit Banana Bread         Harbour $3.10 | Garden $2.20
Single-item orders at June pace: 44,968/yr. Add-on prompt value at $3.04 GP per add-on: 1% -> $1,367/yr, 3% -> $4,101/yr, 5% -> $6,835/yr
10% bundle discount on every 2nd/3rd item: subsidises existing multi-item orders by $24,980/yr (June pace); a new add-on nets $2.89 GP after the discount, so 8,651 extra add-ons/yr (19.2% of single-item orders) are needed just to break even

[6] BRANCH MENU TILT (first item of each order)
  Sandwiches: Garden 6.76%, Harbour 6.86%, Uptown 11.83%
  Desserts:   Garden 13.25%, Harbour 8.48%, Uptown 7.42%

[7] CUSTOMER ID BEHAVIOUR
Visits per ID: mean 6.16, variance/mean 0.99 (1.00 = pure chance)
New IDs first seen by month: M1 1,979, M2 843, M3 433, M4 159, M5 60, M6 20
Average order value with vs without ID: $9.02 vs $9.04
Orders with a Customer ID: 21,534
Is a missing Customer ID linked to... (chi-square p-values; high = no link): Branch p=0.71, Hour p=0.35, first_cat p=0.94, size p=0.79

[8] OTHER CHECKS QUOTED IN THE README
Hot Coffee share of items: Jan-Mar 15.9%, Apr-Jun 15.9% (p=0.94)
Product share by month (do products trend?): chi-square p=0.22
