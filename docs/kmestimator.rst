Kaplan-Meier Estimator for Predicting Age/Year of Death
=======================================================

The `KaplanMeierEstimator` takes an array of cumulative deaths and returns an
object that will sample from the Kaplan-Meier distribution.

A sample input array of cumulative deaths might look like this:

```
cd[0] = 0   # No deaths before age 0
cd[1] = 687
cd[2] = 733
cd[3] = 767
...
cd[101] = 100_000  # 100,000 deaths by age 101
```

`predict_year_of_death()` takes an array of current ages (in years) and returns
an array of predicted years of death based on the cumulative deaths input array.

*Note:* `predict_year_of_death()` can use non-constant width age bins and will
return predictions *by age bin*. In this case, it is up to the user to convert
the returned bin indices to actual years.

A sample non-constant width age bin input array might look like this:

```
cd[0] = 0   # No deaths before age 0
cd[1] = 340 # 1/2 of first year deaths in the first 3 months
cd[2] = 170 # 1/4 of first year deaths in the next 3 months
cd[3] = 177 # 1/4 of first year deaths in the last 6 months
cd[4] = 733
cd[5] = 767
...
cd[104] = 100_000  # 100,000 deaths by age 101
```

In this example, values returned from `predict_year_of_death()` would need to
be mapped as follows:

0 -> (0, 3] months
1 -> (3, 6] months
2 -> (6, 12] months
3 -> 1 year
4 -> 2 years
...
103 -> 100 years

`predict_age_at_death()` takes an array of current ages (in days) and returns
an array of predicted ages (in days) at death. The implementation assumes that
the cumulative deaths input array to the estimator represents one year age bins.
If you are using non-constant width age bins, you should manually convert bin
indices returned from `predict_year_of_death()` to ages.
