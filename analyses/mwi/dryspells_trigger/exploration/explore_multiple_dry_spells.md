## Explore multiple dry spells in a season

Simple and quick code to look at times where there have been multiple dry spells in a season

```python
import pandas as pd
import os

dd = os.environ["AA_DATA_DIR"]

df = pd.read_csv(os.path.join(dd, 'public', 'processed', 'mwi', 'arc2', 'mwi_arc2_dry_spells_2000_2020.csv'),
                 parse_dates = ['dry_spell_first_date', 'dry_spell_last_date'])
df = df[df.during_rainy_season]
```

Let's first just look at the times when we have multiple dry spells for an ADM2 area in the same season.

```python
df[df.duplicated(subset=['ADM2_PCODE', 'season_approx'], keep=False)]
```

We can see from above that we have 6 instances of this occurring. However, if we look at the second recorded dry spell, only 1 of them occurs within our monitoring window. The rest are observed in March after the monitoring has stopped.

```python
df[df.duplicated(subset=['ADM2_PCODE', 'season_approx'], keep='first') & (df.dry_spell_first_date.dt.month <= 2) & (df.dry_spell_first_date.dt.day <= 22)]
```

However, in this case, the first dry spell detected was at the beginning of November, again falling outside of the monitoring window.

```python
df[df.duplicated(subset=['ADM2_PCODE', 'season_approx'], keep='last') & (df.dry_spell_last_date.dt.month >= 11)]
```

Thus, none of these would actually change the return period for our analysis based on the monitoring window set up for the V1 Malawi trigger. However, it's still something we need to consider when implementing the monitoring.

Given the focus on geographical spread of a dry spell to measure impact rather than duration, I think makes sense not to count a district twice if 2 separate dry spells occur. Given our short time window for monitoring, these dry spells are likely only broken apart by a small number of days of rainfall. Since the impact of a dry spell is already sufficiently high when the 14 days have passed, the marginal impact of a second dry spell is likely less. The same intuition is why we wouldn’t double count a dry spell that last for 30 days, for instance. However, important to raise for discussion and confirming our decision
