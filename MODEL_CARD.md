# Model card: spare-parts reorder policy

## Model details
- **Forecast:** global XGBoost regressor (Tweedie loss) predicting next month's demand per part from its own lags, rolling means, demand frequency, months since last demand and calendar month. Parts with under 13 months of history use TSB.
- **Policy:** monthly review, 1-month lead time, order up to the 98% quantile of 2-month demand (negative binomial, variance from recent forecast errors).
- **Version:** 0.1.0, trained on January 1998 – March 2002.

## Intended use
Setting reorder levels and order quantities for slow-moving, intermittent spare parts. The output is advice for planners, who should override it for critical parts, known phase-outs and supplier constraints.

## Evaluation (April 2001 – March 2002, 2,509 parts)

| Metric | Value |
|---|---|
| Fill rate | 93.7% (rule of thumb 90.6%) |
| Average stock per part | 4.10 units (rule of thumb 4.46) |
| Cost against rule of thumb | −21.5% (95% CI −24.6% to −18.7%) |
| Stock for a 90% fill rate | 3.15 units (rule of thumb 4.28) |

## Known limitations and risks
- **Assumed costs:** the service target (98%) was chosen for an assumed holding cost of 25% a year and a backorder cost of one part value per unit-month; other costs give other targets.
- **Lumpy parts get large safety stocks:** a part with occasional large orders can be told to hold many times its average monthly demand. Review these by hand.
- **Obsolescence:** forecasts decline as parts stop selling, but a formal end-of-life signal (model discontinued) should override the forecast.
- **Single test year** of a single dataset.
