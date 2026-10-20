# Rainwater harvesting design: 4-person household, 120 m² metal roof, 600 mm/yr (80 % in a 5-month wet season)

## Problem framing

**Goal.** Supply non-potable demand (toilet flushing, laundry, garden) from roof runoff. Answer four questions: sensible tank size, expected annual yield, how long the tank lasts in the dry season, and a rough cost range.

**Knowns (given).** Roof 120 m² (taken as plan area), metal. Rain 600 mm/yr, so 480 mm in 5 wet months (96 mm/month) and 120 mm in 7 dry months (17.1 mm/month). 4 occupants.

**Unknowns the user did not give (I assumed them, all editable in the code).**

| Item | Value used | Basis |
|---|---|---|
| Runoff coefficient, metal roof | 0.90 | Literature range 0.85–0.95 for metal sheet/standing seam |
| First flush diverted | 1 mm per rain event (sensitivity: 2 mm) | Sources give 0.8–5 mm; 1 mm and 2 mm are the commonly cited design values |
| Rain events/month | 8 wet, 3 dry | My assumption (needed to convert first-flush depth into monthly loss) |
| Toilet | 30 L/person/day (about 5 flushes × 6 L) | NZ BRANZ study, about 31 L/p/d at 6.2 L/flush; a modern dual-flush (4.5/3 L) would give less |
| Laundry | 10 L/person/day (40 L/day, about 5 loads/week × 50 L) | Front-loaders 42–60 L/load; top-loaders 170+ L would be about 3× larger |
| Garden | 20 m² of beds × 25 L/m²/week, dry-season months only (2.1–2.2 m³/month) | 25 mm/week is the standard vegetable-garden figure; assumed wet-season rain covers it |
| Indoor non-potable total | 40 L/p/d, 160 L/day, 58.4 m³/yr | Sum of the above (about 31 % of a 130 L/p/d total, within the 40 % "toilet + laundry" share reported for Finland) |
| Month ordering | Months 1–5 wet, 6–12 dry; rain uniform within each season | Simplification; real distribution is peaky |
| Priority | Indoor use served first each time step; mains is the backup | Garden is the discretionary use |

**Two design approaches considered.**
1. **Rule-of-thumb sizing** (annual demand ÷ 12 months, or "dry-period demand × days"): fast, but ignores the 7-month dry season and spill during the wet season, so it can't give an honest yield.
2. **Time-stepped mass balance (chosen).** A monthly yield-after-spillage water balance (as is standard in tank-sizing studies) for a range of tank volumes, cross-checked by a stochastic daily simulation because monthly models overstate performance (they assume rain and demand are spread evenly inside the month). This answers all four questions from one model and exposes the dry-season constraint.

**Success criteria (written before computing).**
- Mass balance closes (inflow = supplied + spill + Δstorage) to numerical precision.
- Yield ≤ inflow ≤ rainfall × area × runoff coefficient (72 m³ ceiling for 600 mm × 120 m²; 64.8 m³ with CD).
- Units consistent (1 mm on 1 m² = 1 L).
- At least 5 m³ and 10 m³ compared, with spill, reliability and dry-season duration.
- Recommendation states honestly whether it "covers" the stated needs; if it cannot, say by how much and why.

## Solution

**Recommendation: one 10 m³ polyethylene tank (or 2 × 5 m³ linked), pumped to toilets and washing machine, with an automatic mains top-up/changeover, a first-flush diverter on each downpipe, and the garden on drip/mulch with low priority.**

| Quantity | 5 m³ tank | **10 m³ tank** | 20 m³ (for reference) |
|---|---|---|---|
| Rain on roof (600 mm × 120 m²) | 72 m³ | 72 m³ | 72 m³ |
| After CD 0.9 and 1 mm first flush (tank inflow) | 58.2 m³ | 58.2 m³ | 58.2 m³ |
| **Annual yield, monthly model** | 39.9 m³ | **44.9 m³** | 54.9 m³ |
| **Annual yield, daily stochastic model** | 39.0 m³ | **43.8 m³** | 52.4 m³ |
| Indoor (toilet + laundry) demand met, monthly / daily | 65 % / 59 % | **71 % / 65 %** | 83 % / 75 % |
| Garden demand met (daily model; garden is low priority) | 29 % | **39 %** | 57 % |
| Spill (overflow), monthly model | 18.4 m³ | 13.4 m³ | 3.4 m³ |
| Months needing mains for indoor use (monthly model) | 6 | **5** | 3 |
| Tank life into dry season, full at start, no rain, indoor only | 31 days | **62 days** | 125 days |
| ... indoor + garden drawn | 22 days | **43 days** | 86 days |
| ... indoor only, with average dry-season inflow (50 L/day) | 45 days | **91 days** | 182 days |
| Rough installed cost (USD-equivalent) | 3,000–8,300 | **3,300–9,000** | not costed |

**Headline answers.**
- **Tank size.** 10 m³. This is about 2 months of indoor demand (62 days). A 5 m³ tank holds only one month (31 days) and spills much of the wet-season surplus. The wet season delivers about 9.5 m³/month against 5 m³/month of indoor demand, so a 10 m³ tank gains about 4.5 m³/month net and refills from empty in about 2 wet months, then spills for the remaining 2–3.
- **Annual yield.** About 44 m³/yr (range 42–45 m³ across sensitivities). That is about 60 % of the 73.7 m³/yr total demand (58.4 indoor + 15.3 garden), and about 65–71 % of the indoor demand alone.
- **Dry season.** The 10 m³ tank, full at the end of the wet season, lasts about 6–9 weeks if the garden is also watered from it, or about 9–13 weeks (62–91 days) for indoor use only. The dry season is 7 months (about 213 days), so mains backup is needed for about 4–5 months. In the monthly table the tank is empty from month 7 onward.
- **Cost.** Roughly USD 3,300–9,000 installed for the 10 m³ system. Tank only is about USD 1,000–1,650. A plain 5 m³ system is about 3,000–8,300. Most of the spread is plumbing, pump and site work, not tank volume. Going from 5 to 10 m³ adds only about USD 350–700. A cross-check from a US guide is USD 2,000–8,000 and from UK guides GBP 3,000–6,000+.
- **Does it "cover" the stated needs? Not fully, and no tank size can.** Even an infinitely large tank only reaches about 85 % of indoor demand in the monthly model, because the dry-season inflow is only 10.7 m³ against 34 m³ of indoor demand in those 7 months. Mains backup is a design requirement. The garden can only be partly served; see the caveats on strategy.
- **Economics.** The 10 m³ yield (44 m³/yr) is worth about USD 66–175/yr at an assumed tariff of USD 1.5–4/m³. Payback on water savings alone is therefore many decades. The case rests on resilience, stormwater reduction and mains-load reduction, not on bills.

## How I got there

1. Restated the problem and set assumptions (table above), searched for reference values (runoff coefficient, first flush, per-person demand, garden need, costs).
2. Computed the supply side: 600 mm × 120 m² = 72 m³ gross, × 0.9 = 64.8 m³, minus first flush (1 mm × events/month) = 58.2 m³/yr, of which 47.5 m³ falls in the 5 wet months and only 10.7 m³ in the 7 dry months.
3. Computed demand: 160 L/day indoor = 58.4 m³/yr, plus 15.3 m³/yr garden in the dry season. Indoor demand alone nearly equals total annual supply, so the whole problem is timing and storage.
4. Ran a monthly balance for 2, 5, 10, 15, 20, 30 m³, then a 40-year daily stochastic check, a draw-down calculation, sensitivities and a bottom-up cost.
5. Verified with assertions, then compared results with the success criteria.

Full code (single file, runs with `python3`, numpy only), followed by its output:

```python
"""Rainwater harvesting design: 120 m2 metal roof, 600 mm/yr, 5-month wet season (80 % of rain), 4 people."""
import numpy as np

# ---------------- ASSUMPTIONS (all editable) ----------------
A, P_ANNUAL = 120.0, 600.0           # roof plan area m2 ; mm/yr
WET_FRAC, WET_N = 0.80, 5            # share of rain in wet season ; wet months (months 1-5, uniform within season)
CD = 0.90                            # runoff coefficient, metal roof (0.85-0.95 in literature)
FF_MM = 1.0                          # first-flush diverted per rain event, mm (1-2 mm in literature)
EV_WET, EV_DRY = 8, 3                # assumed rain events per month (wet / dry)  [assumption]
PEOPLE = 4
TOILET_LPD = 30.0                    # L/person/day  (5 flushes x ~6 L; NZ BRANZ ~31)
LAUNDRY_LPD = 10.0                   # L/person/day  (~5 loads/wk x ~50 L front-loader / 4 people)
GARDEN_M2, GARDEN_LPM2_WK = 20.0, 25.0   # irrigated beds ; 25 mm/week when no rain, dry-season months only
MDAYS = np.array([31,28,31,30,31,30,31,31,30,31,30,31])   # 365-day year

rain = np.array([P_ANNUAL*WET_FRAC/WET_N]*WET_N + [P_ANNUAL*(1-WET_FRAC)/(12-WET_N)]*(12-WET_N))  # mm/month
events = np.array([EV_WET]*WET_N + [EV_DRY]*(12-WET_N))
assert abs(rain.sum()-P_ANNUAL) < 1e-9 and abs(rain[:WET_N].sum()/P_ANNUAL-WET_FRAC) < 1e-9

def demands(garden_m2=GARDEN_M2, scale=1.0):
    """m3/month: indoor (toilet+laundry), garden (dry-season months only)."""
    indoor = PEOPLE*(TOILET_LPD+LAUNDRY_LPD)*scale*MDAYS/1000
    garden = np.where(np.arange(12) < WET_N, 0.0, garden_m2*GARDEN_LPM2_WK*MDAYS/7/1000)
    return indoor, garden

def inflow(ff_mm=FF_MM, cd=CD):
    net = np.maximum(rain - ff_mm*events, 0)*A*cd/1000       # m3/month into tank
    return net, rain*A/1000                                  # (net inflow, gross rain volume)

# ---------------- 1. MONTHLY WATER BALANCE (yield-after-spillage, indoor priority) ----------------
def monthly(cap, garden_m2=GARDEN_M2, ff_mm=FF_MM, cd=CD, scale=1.0, years=4):
    Y, G = inflow(ff_mm, cd); di, dg = demands(garden_m2, scale); S = 0.0
    for _ in range(years):                                    # iterate to periodic steady state
        rows = []
        for m in range(12):
            avail = S + Y[m]
            si = min(di[m], avail); avail -= si
            sg = min(dg[m], avail); avail -= sg
            sp = max(avail-cap, 0); Snew = avail-sp
            rows.append((G[m], Y[m], di[m], si, dg[m], sg, sp, S, Snew)); S = Snew
    r = np.array(rows)
    out = dict(cap=cap, gross=r[:,0].sum(), inflow=r[:,1].sum(), indoor_dem=r[:,2].sum(), indoor=r[:,3].sum(),
               garden_dem=r[:,4].sum(), garden=r[:,5].sum(), spill=r[:,6].sum(), dS=r[-1,8]-r[0,7], table=r)
    out['yield'] = out['indoor']+out['garden']
    out['closure'] = out['inflow']-out['yield']-out['spill']-out['dS']   # must be ~0
    out['indoor_rel'] = out['indoor']/out['indoor_dem']
    out['mains_months'] = int(((r[:,2]-r[:,3]) > 1e-9).sum())
    return out

# ---------------- 2. DAILY STOCHASTIC CHECK (random event timing, interannual rain CV) ----------------
def daily(cap, years=40, seed=1, cv=0.25, garden_m2=GARDEN_M2, ff_mm=FF_MM, cd=CD, scale=1.0):
    rng = np.random.default_rng(seed); S = 0.0; res = []
    di_d = PEOPLE*(TOILET_LPD+LAUNDRY_LPD)*scale/1000
    dg_d = garden_m2*GARDEN_LPM2_WK/7/1000
    for yr in range(years+2):                                 # first 2 yrs discarded (spin-up)
        k = rng.gamma(1/cv**2, cv**2)                         # year wetness factor, mean 1
        t = dict(gross=0, inflow=0, di=0, si=0, dg=0, sg=0, spill=0); S0 = S; first_empty = None; dry_day = 0
        for m in range(12):
            n = int(MDAYS[m]); wet = m < WET_N
            ne = min(max(1, rng.poisson(events[m])), n)
            w = rng.gamma(0.7, 1, ne); r = np.zeros(n)
            r[rng.choice(n, ne, replace=False)] = rain[m]*k*w/w.sum()
            for d in range(n):
                t['gross'] += r[d]*A/1000
                inf = max(r[d]-ff_mm, 0)*A*cd/1000; S += inf; t['inflow'] += inf
                dI, dG = di_d, (0.0 if wet else dg_d)
                sI = min(dI, S); S -= sI; sG = min(dG, S); S -= sG
                sp = max(S-cap, 0); S -= sp
                t['di'] += dI; t['si'] += sI; t['dg'] += dG; t['sg'] += sG; t['spill'] += sp
                if not wet:
                    dry_day += 1
                    if first_empty is None and S < 1e-9: first_empty = dry_day
        t['close'] = t['inflow']-t['si']-t['sg']-t['spill']-(S-S0); t['empty'] = first_empty
        if yr >= 2: res.append(t)
    f = lambda k: np.array([x[k] for x in res], float)
    e = [x['empty'] for x in res if x['empty'] is not None]
    return dict(cap=cap, yield_=(f('si')+f('sg')).mean(), indoor=f('si').mean(), indoor_rel=(f('si')/f('di')).mean(),
                indoor_rel_p10=np.percentile(f('si')/f('di'), 10), garden_rel=(np.nan if f('dg').sum()==0 else (f('sg')/np.maximum(f('dg'),1e-12)).mean()),
                spill=f('spill').mean(), gross=f('gross').mean(), max_close=np.abs(f('close')).max(),
                empty_day_median=float(np.median(e)) if e else None, n_empty=len(e), n=len(res))

# ---------------- 3. TANK DRAW-DOWN DAYS ----------------
def drawdown_days(cap):
    di = PEOPLE*(TOILET_LPD+LAUNDRY_LPD)/1000; dg = GARDEN_M2*GARDEN_LPM2_WK/7/1000
    Y, _ = inflow(); dry_in = Y[WET_N:].sum()/MDAYS[WET_N:].sum()          # m3/day average dry-season inflow
    return dict(indoor_no_rain=cap/di, indoor_plus_garden_no_rain=cap/(di+dg), indoor_with_avg_dry_inflow=cap/(di-dry_in))

# ---------------- 4. COST (USD-equivalent; ranges from cited sources, see Sources) ----------------
AUD_USD = 0.65   # ASSUMED exchange rate, only to put Australian retail tank prices on a USD basis
def cost(cap):
    tank = {5: (800*AUD_USD, 1000*AUD_USD), 10: (1250*AUD_USD, 1695*AUD_USD)}[cap]       # poly tank retail (Camden/Wollondilly guide)
    tank = (tank[0]*1.2, tank[1]*1.5)                                                     # + delivery/margin/regional uplift [assumption]
    items = {'tank': tank,
             'base slab / foundation': (300, 1000),                                       # [assumption; flagged by source as extra]
             'gutter repair+leaf mesh, downpipe works': (300, 800),                       # [assumption]
             'first-flush diverters (2-4 downpipes, AUD66-95 each + fittings)': (150, 500),
             'pump + filter + mains-backup/changeover': (600, 1500),                       # US guide avg ~600 pump+filtration
             'non-potable plumbing to toilets+washer, labelling, plumber': (1000, 3500)}   # US install avg 2000, high 5000
    lo = sum(v[0] for v in items.values()); hi = sum(v[1] for v in items.values())
    return items, lo, hi

if __name__ == "__main__":
    Y, G = inflow(); di, dg = demands()
    print(f"Gross rain on roof {G.sum():.1f} m3 | x CD {G.sum()*CD:.1f} | after first flush {Y.sum():.1f} m3/yr")
    print(f"Wet-season inflow {Y[:WET_N].sum():.1f}, dry-season {Y[WET_N:].sum():.1f} m3")
    print(f"Indoor demand {di.sum():.1f} m3/yr ({PEOPLE*(TOILET_LPD+LAUNDRY_LPD):.0f} L/d); garden {dg.sum():.1f} m3/yr (dry season)")
    print("\nMONTHLY balance (steady state)")
    for cap in (2, 5, 10, 15, 20, 30):
        s = monthly(cap)
        print(f"{cap:>3} m3: yield {s['yield']:5.1f} = indoor {s['indoor']:.1f}/{s['indoor_dem']:.1f} ({s['indoor_rel']:.0%}) + garden {s['garden']:.1f}/{s['garden_dem']:.1f}; "
              f"spill {s['spill']:5.1f}; dS {s['dS']:+.2f}; closure {s['closure']:+.1e}; mains-months {s['mains_months']}")
    print("\nDAILY stochastic (40 yr, CV 0.25)")
    for cap in (2, 5, 10, 15, 20, 30):
        d = daily(cap)
        print(f"{cap:>3} m3: yield {d['yield_']:5.1f}; indoor {d['indoor']:.1f} ({d['indoor_rel']:.0%}, P10 {d['indoor_rel_p10']:.0%}); garden {d['garden_rel']:.0%}; "
              f"spill {d['spill']:.1f}; empty after {d['empty_day_median']} dry-season days (n={d['n_empty']}/{d['n']}); closure {d['max_close']:.1e}")
    print("\nDRAW-DOWN (full tank at start of dry season, days)")
    for cap in (5, 10, 15, 20):
        print(cap, {k: round(v) for k, v in drawdown_days(cap).items()})
    print("\nMonthly table for 10 m3 (month, rain mm, net inflow, indoor dem, indoor supplied, garden dem, garden supplied, spill, S_start, S_end)")
    t = monthly(10)['table']
    for m in range(12): print(m+1, round(rain[m],1), *[round(x,2) for x in t[m][1:]])
    print("\nSENSITIVITY (10 m3 tank; monthly | daily indoor reliability, yield m3/yr)")
    cases = {'base': {}, 'FF 2 mm': dict(ff_mm=2.0), 'CD 0.80': dict(cd=0.80), 'CD 0.95': dict(cd=0.95),
             'demand -25%': dict(scale=0.75), 'demand +25%': dict(scale=1.25), 'no garden': dict(garden_m2=0), 'garden 50 m2': dict(garden_m2=50)}
    for k, kw in cases.items():
        m_ = monthly(10, **kw); d_ = daily(10, **kw)
        print(f"{k:13s} monthly {m_['indoor_rel']:.0%} yield {m_['yield']:.1f} | daily {d_['indoor_rel']:.0%} yield {d_['yield_']:.1f}")
    print("\nCOST (USD-equivalent)")
    for cap in (5, 10):
        items, lo, hi = cost(cap); print(cap, "m3 total", round(lo,-1), "-", round(hi,-1))
        for k, v in items.items(): print("   ", k, round(v[0],-1), "-", round(v[1],-1))
    s5, s10 = daily(5), daily(10)
    print("marginal 5->10 m3: extra yield (daily) %.1f m3/yr" % (s10['yield_']-s5['yield_']))
    for tariff in (1.5, 4.0): print(f"annual value of 10 m3 yield at ${tariff}/m3: ${s10['yield_']*tariff:.0f}")

    # ---------------- 5. CHECKS ----------------
    print("\nCHECKS")
    ceiling = P_ANNUAL*A/1000                                   # 1 mm on 1 m2 = 1 L  ->  600 mm x 120 m2 = 72 m3
    for cap in (2, 5, 10, 15, 20, 30):
        s = monthly(cap); d = daily(cap)
        assert abs(s['closure']) < 1e-9 and d['max_close'] < 1e-9           # mass balance closes
        assert s['yield'] <= s['inflow'] + 1e-9 <= ceiling*CD + 1e-9        # yield <= inflow <= rain x area x CD
        assert d['yield_'] <= d['gross'] and d['gross'] <= ceiling*1.5       # daily: yield <= gross rain
        assert s['indoor'] <= s['indoor_dem'] + 1e-9
    assert abs(demands()[0].sum()*1000/365/(PEOPLE) - (TOILET_LPD+LAUNDRY_LPD)) < 1e-9   # L/person/day round-trip
    print("all assertions passed; ceiling (rain x area) =", ceiling, "m3; max yield seen =", round(max(monthly(c)['yield'] for c in (2,5,10,15,20,30)),1))
```

Output:

```text
Gross rain on roof 72.0 m3 | x CD 64.8 | after first flush 58.2 m3/yr
Wet-season inflow 47.5, dry-season 10.7 m3
Indoor demand 58.4 m3/yr (160 L/d); garden 15.3 m3/yr (dry season)

MONTHLY balance (steady state)
  2 m3: yield  36.9 = indoor 36.9/58.4 (63%) + garden 0.0/15.3; spill  21.4; dS +0.00; closure +0.0e+00; mains-months 7
  5 m3: yield  39.9 = indoor 38.1/58.4 (65%) + garden 1.7/15.3; spill  18.4; dS +0.00; closure +0.0e+00; mains-months 6
 10 m3: yield  44.9 = indoor 41.6/58.4 (71%) + garden 3.3/15.3; spill  13.4; dS +0.00; closure +1.1e-14; mains-months 5
 15 m3: yield  49.9 = indoor 45.0/58.4 (77%) + garden 4.9/15.3; spill   8.4; dS +0.00; closure +1.1e-14; mains-months 4
 20 m3: yield  54.9 = indoor 48.3/58.4 (83%) + garden 6.6/15.3; spill   3.4; dS +0.00; closure +7.1e-15; mains-months 3
 30 m3: yield  58.2 = indoor 49.5/58.4 (85%) + garden 8.7/15.3; spill   0.0; dS +0.00; closure +7.1e-15; mains-months 3

DAILY stochastic (40 yr, CV 0.25)
  2 m3: yield  34.3; indoor 30.8 (53%, P10 47%); garden 23%; spill 28.3; empty after 7.0 dry-season days (n=40/40); closure 9.2e-14
  5 m3: yield  39.0; indoor 34.5 (59%, P10 55%); garden 29%; spill 23.6; empty after 24.0 dry-season days (n=40/40); closure 2.0e-13
 10 m3: yield  43.8; indoor 37.8 (65%, P10 59%); garden 39%; spill 18.8; empty after 50.0 dry-season days (n=40/40); closure 2.8e-13
 15 m3: yield  48.4; indoor 40.9 (70%, P10 63%); garden 49%; spill 14.2; empty after 78.5 dry-season days (n=40/40); closure 3.5e-13
 20 m3: yield  52.4; indoor 43.7 (75%, P10 63%); garden 57%; spill 10.2; empty after 107.0 dry-season days (n=40/40); closure 4.8e-13
 30 m3: yield  57.9; indoor 47.4 (81%, P10 63%); garden 69%; spill 4.7; empty after 140.0 dry-season days (n=40/40); closure 7.3e-13

DRAW-DOWN (full tank at start of dry season, days)
5 {'indoor_no_rain': 31, 'indoor_plus_garden_no_rain': 22, 'indoor_with_avg_dry_inflow': 45}
10 {'indoor_no_rain': 62, 'indoor_plus_garden_no_rain': 43, 'indoor_with_avg_dry_inflow': 91}
15 {'indoor_no_rain': 94, 'indoor_plus_garden_no_rain': 65, 'indoor_with_avg_dry_inflow': 136}
20 {'indoor_no_rain': 125, 'indoor_plus_garden_no_rain': 86, 'indoor_with_avg_dry_inflow': 182}

Monthly table for 10 m3 (month, rain mm, net inflow, indoor dem, indoor supplied, garden dem, garden supplied, spill, S_start, S_end)
1 96.0 9.5 4.96 4.96 0.0 0.0 0.0 0.0 4.54
2 96.0 9.5 4.48 4.48 0.0 0.0 0.0 4.54 9.57
3 96.0 9.5 4.96 4.96 0.0 0.0 4.11 9.57 10.0
4 96.0 9.5 4.8 4.8 0.0 0.0 4.7 10.0 10.0
5 96.0 9.5 4.96 4.96 0.0 0.0 4.54 10.0 10.0
6 17.1 1.53 4.8 4.8 2.14 2.14 0.0 10.0 4.58
7 17.1 1.53 4.96 4.96 2.21 1.15 0.0 4.58 0.0
8 17.1 1.53 4.96 1.53 2.21 0.0 0.0 0.0 0.0
9 17.1 1.53 4.8 1.53 2.14 0.0 0.0 0.0 0.0
10 17.1 1.53 4.96 1.53 2.21 0.0 0.0 0.0 0.0
11 17.1 1.53 4.8 1.53 2.14 0.0 0.0 0.0 0.0
12 17.1 1.53 4.96 1.53 2.21 0.0 0.0 0.0 0.0

SENSITIVITY (10 m3 tank; monthly | daily indoor reliability, yield m3/yr)
base          monthly 71% yield 44.9 | daily 65% yield 43.8
FF 2 mm       monthly 68% yield 42.6 | daily 62% yield 41.7
CD 0.80       monthly 70% yield 43.7 | daily 63% yield 42.1
CD 0.95       monthly 72% yield 45.4 | daily 66% yield 44.6
demand -25%   monthly 79% yield 38.8 | daily 71% yield 38.5
demand +25%   monthly 67% yield 50.9 | daily 60% yield 48.7
no garden     monthly 77% yield 44.9 | daily 75% yield 43.8
garden 50 m2  monthly 68% yield 44.9 | daily 58% yield 43.8

COST (USD-equivalent)
5 m3 total 2970.0 - 8280.0
    tank 620.0 - 980.0
    base slab / foundation 300 - 1000
    gutter repair+leaf mesh, downpipe works 300 - 800
    first-flush diverters (2-4 downpipes, AUD66-95 each + fittings) 150 - 500
    pump + filter + mains-backup/changeover 600 - 1500
    non-potable plumbing to toilets+washer, labelling, plumber 1000 - 3500
10 m3 total 3320.0 - 8950.0
    tank 980.0 - 1650.0
    base slab / foundation 300 - 1000
    gutter repair+leaf mesh, downpipe works 300 - 800
    first-flush diverters (2-4 downpipes, AUD66-95 each + fittings) 150 - 500
    pump + filter + mains-backup/changeover 600 - 1500
    non-potable plumbing to toilets+washer, labelling, plumber 1000 - 3500
marginal 5->10 m3: extra yield (daily) 4.8 m3/yr
annual value of 10 m3 yield at $1.5/m3: $66
annual value of 10 m3 yield at $4.0/m3: $175

CHECKS
all assertions passed; ceiling (rain x area) = 72.0 m3; max yield seen = 58.2
```

## Checks performed

Programmatic (all assertions pass, see end of the script output):
- **Mass balance closes.** `inflow − (indoor supplied + garden supplied) − spill − ΔS` is 0 to 1e-14 (monthly) and below 1e-12 (daily) for every tank size. ΔS = 0 at steady state (the tank cycles back to the same level each year).
- **Yield ceiling.** The largest yield seen is 58.2 m³ (30 m³ tank, no spill), which equals the inflow after losses and is below 64.8 m³ (rain × area × CD) and 72 m³ (rain × area). The daily model's mean gross rain on the roof is 72.0 m³, matching 600 mm × 120 m².
- **Units.** 1 mm over 1 m² is 1 L. 600 mm × 120 m² = 72,000 L = 72 m³. Daily demand 4 × 40 L = 160 L/day gives 58.4 m³ over 365 days. Supplied indoor never exceeds indoor demand.
- **Seasonal split.** Wet-season rain is 480.0 mm = 80.0 % of 600 mm (asserted in code). Wet-season inflow 47.5 m³ + dry-season 10.7 m³ = 58.2 m³.
- **Hand check of the 10 m³ table.** Month 1: inflow 96 mm − 8 mm first flush = 88 mm × 120 m² × 0.9 / 1000 = 9.50 m³. Indoor demand 160 L × 31 d = 4.96 m³, so the tank gains 4.54 m³, as in the table.
- **Monotonicity and convergence.** Yield rises and spill falls as the tank grows. A 30 m³ tank spills 0 in the monthly model, so storage is no longer the limit there; supply is.
- **Two independent models agree in direction.** Monthly and daily models give 71 % and 65 % indoor reliability at 10 m³. The daily model is lower, as expected, because rain falls in discrete events and the tank spills within wet months.
- **Sensitivity (10 m³, indoor reliability monthly / daily).** First flush 2 mm: 68 % / 62 %. CD 0.80: 70 % / 63 %. CD 0.95: 72 % / 66 %. Demand −25 %: 79 % / 71 %. Demand +25 %: 67 % / 60 %. No garden: 77 % / 75 %. 50 m² garden: 68 % / 58 %. Yield varies only 41.7–50.9 m³/yr.
- **Requirement check.** The recommendation does not fully meet "cover non-potable use". It meets it to about 65–71 % for indoor use, with mains backup. This is stated above, not hidden.

## Caveats & confidence

- **Confidence.** High that the structure and numbers are internally consistent. Medium that annual yield is 40–50 m³ (±15 %). Low-to-medium on cost, which is dominated by local plumbing and labour rates, and the user's country is unknown.
- **Rainfall distribution is idealised.** Months are uniform within each season, events per month are my assumption, and the stochastic daily model uses a gamma year-to-year multiplier (CV 0.25) that is not fitted to real data. Real records (daily gauge data) should replace this before final design. A late wet-season start or a single dry year (P10 indoor reliability 59 % at 10 m³) would reduce yield.
- **Garden competes with indoor use.** Total yield barely changes with garden size (the tank drains either way), but every litre to the garden is a litre not available for toilets later in the dry season. Indoor reliability falls from 75 % (no garden) to 65 % (20 m²) to 58 % (50 m²) in the daily model. Practical strategy: water the garden from the tank only in the first weeks of the dry season, use mulch and drip, keep the garden area small, and use mains or greywater later. The 25 mm/week garden demand and the zero wet-season demand are crude.
- **Demand is the biggest uncertainty.** A top-loading washer (170 L/load) would roughly triple laundry demand. Low-flush toilets would cut toilet demand. ±25 % demand moves indoor reliability by about ±5–8 points.
- **Cost inputs are partly assumptions.** Tank prices come from an undated Australian guide, converted at an assumed AUD 1 = USD 0.65 and marked up 20–50 % for delivery and regional uplift (my assumption). Base slab, gutter works and the plumbing range are my estimates informed by the US/UK guides. No pump price was found directly; the pump + filter range uses the US guide's average of about USD 600 for pumps and filtration. One US cost guide (the 2026 one) is from a low-credibility domain, so treat it as indicative only. The tariff of USD 1.5–4/m³ is an assumption with no source.
- **Source quality.** Reference figures came from web-search result summaries; I did not fetch and verify the underlying pages (fetches may be blocked). Where a claim could not be mapped confidently to one URL (for example the 10,000 L tank price) I cite the candidate pages and say so.
- **Out of scope.** Roof material leaching (metal roofs can release zinc or coatings; avoid copper/lead flashing or treated surfaces), mosquito-proofing, overflow routing and local regulations/backflow rules (a dual-plumbed non-potable system needs an air-gap or approved backflow protection and labelled pipes in most jurisdictions). This water is not designed for drinking.

## Lessons

- **Method.** Writing the success criteria first (especially "say honestly if the needs cannot be met") surfaced the key finding early: annual supply (58 m³) roughly equals indoor demand (58 m³), but 82 % of the inflow arrives in the 5 wet months, so storage and timing, not roof area, limit performance.
- **A monthly model is optimistic.** The 5–6 point gap between monthly and daily results shows why a daily check is worth the 40 lines of code.
- **There is no sharp knee in the tank-size curve.** Yield rises about 0.9 m³/yr per extra m³ of tank up to about 20 m³, because the 7-month dry season is far longer than any affordable tank can bridge. The choice of 10 m³ is therefore a judgement (about 2 months of storage, one standard tank, modest cost uplift), not an optimum.
- **Do differently.** Get real daily rainfall for the site, ask the user's country and tariff, measure actual toilet/washer volumes, and quote the tank price in local currency. Also optimise garden use explicitly with a reserve rule (keep a minimum indoor reserve) instead of simple priority. Note for tooling: `python -I` blocks importing sibling scripts, which cost one retry.

## Sources

Search-result summaries only; pages were not fetched or independently verified.

Runoff coefficient and first flush
- Harvesting Water from Roofs, Univ. of Idaho: https://webpages.uidaho.edu/larc380/new380/pages/harvestRoof.html (first flush 1 gal/100 ft²; metal coefficients)
- Calculating roof runoff worked example (0.9 for metal, subtracts diverter loss): https://old.urbanfoodgarden.org/main/water-management/calculating--roof-runoff.htm
- Rainwater Harvesting Planning Guide (metal roof efficiency 0.90–0.95): https://www.toolgrit.com/guides/rainwater-harvesting-guide
- Kyoto Univ. paper (CD 0.9, first flush 2 mm): https://repository.kulib.kyoto-u.ac.jp/server/api/core/bitstreams/eef9c955-2191-4310-a272-1eb06eaedde5/content
- MIT thesis, first flush in Rwanda (1 mm): https://dspace.mit.edu/handle/1721.1/44289
- Metals in rooftop harvested rainwater (why first flush and treatment matter): https://www.jsg.utexas.edu/icp-ms/2009/10/metals-in-rooftop-harvested-rainwater/
- Monthly mass-balance / yield-after-spillage method: https://ir.unimas.my/id/eprint/35241/

Demand
- Kāpiti Coast District Council (BRANZ study; toilet about 31 L/p/d, about 6.2 L/flush): https://www.kapiticoast.govt.nz/services/waters/water-supply/water-conservation/
- Water Corporation (WA), efficient appliances (4.5/3 L dual flush; 50–60 L/wash for 4.5-star): https://www.watercorporation.com.au/-/media/WaterCorp/Documents/Education/Water-supply-and-water-conservation/Activity-library/Investigate-water-efficient-appliances-inside-the-home.pdf
- CHOICE (front-loader 42 L vs top-loader 170 L per cycle): https://choice.com.au/home-and-living/laundry-and-cleaning/washing-machines/articles/how-much-can-you-save-by-choosing-a-water-efficient-front-loader
- Finland, 130 L/p/d with 50 L for toilet and laundry: https://www.vesi.fi/teemasivu/the-water-etiquette-at-home/
- Garden 25 mm/week: https://cdn.dal.ca/content/dam/dalhousie/pdf/agriculture/ExtendedLearning/gardenbox/Irrigating_Garden%20Box_Online.pdf and https://extension.usu.edu/yardandgarden/research/water-recommendations-for-vegetables.php

Cost
- Camden Council (NSW), poly tank 2,000 L about AUD 550–750; pumps, first flush, guttering and install extra: https://www.camden.nsw.gov.au/assets/pdfs/Environment/Sustainable-Living/SustainabilityBlitz-Water-harvesting.pdf
- Wollondilly Council guide (the 4,000–5,000 L AUD 800–1,000 and 10,000 L AUD 1,250–1,695 figures appeared in search output attributed to an Australian council guide; this PDF was in the results but I could not confirm it is the exact source): https://www.wollondilly.nsw.gov.au/assets/Uploads/Water-Harvesting.pdf
- First-flush diverter kits AUD 66–95: https://watertankfactory.com.au/water-tanks/water-diverter-diy-kits-first-flush/ and https://www.tankworld.com.au/?p=9843
- UK installed cost GBP 3,000–6,000+ for toilets and laundry: https://www.myjobquote.co.uk/questions/how-much-does-a-rainwater-harvesting-system-cost and https://www.renewableenergyhub.co.uk/main/rainwater-harvesting/cost-of-installing-rainwater-harvesting-system/
- US cost guide, USD 2,000–8,000, pump+filtration about 600, install about 2,000 (low-credibility domain, indicative only): https://new.risingsunartscentre.org/news/rainwater-harvesting-system-cost-and-price-guide-2026.html
