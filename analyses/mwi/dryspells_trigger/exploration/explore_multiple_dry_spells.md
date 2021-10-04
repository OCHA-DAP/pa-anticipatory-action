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

We can see from above that we have 6 instances of this occurring. However, how many of the second dry spells occur within our monitoring window?

```python
df[df.duplicated(subset=['ADM2_PCODE', 'season_approx'], keep='first') & (df.dry_spell_first_date.dt.month <= 2) & (df.dry_spell_first_date.dt.day <= 22)]
```

We can see above that only 1 of them occurs within our monitoring window. The rest are observed in March after the monitoring has stopped. As well, we should check if the first observed dry spell occurs in our monitoring window.

```python
df[df.duplicated(subset=['ADM2_PCODE', 'season_approx'], keep='last') & (df.dry_spell_last_date.dt.month >= 11)]
```

In this case, we have one dry spell detected at the beginning of November, falling outside of the monitoring window. In this case actually, this corresponds to the second dry spell in our monitoring window. Thus, none of these sets of dry spells would actually trigger twice in our monitoring window and therefore we see no change in return period for our analysis based on the monitoring window set up for the V1 Malawi trigger. However, it's still something we need to consider when implementing the monitoring.

Given the focus on geographical spread of a dry spell to measure impact rather than duration, I think makes sense not to count a district twice if 2 separate dry spells occur. Given our short time window for monitoring, these dry spells are likely only broken apart by a small number of days of rainfall. Since the impact of a dry spell is already sufficiently high when the 14 days have passed, the marginal impact of a second dry spell is likely less. The same intuition is why we wouldn’t double count a dry spell that last for 30 days, for instance. However, important to raise for discussion and confirming our decision
