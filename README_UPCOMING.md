# Upcoming Matches Block

1. Copy this scaffold into the project root.
2. Promote production models under `modeling/artifacts/production/no_market`.
3. Run `create_and_run_upcoming_block.bat setup`.
4. Populate `data/upcoming/provider_matches.csv` from a licensed or authorized fixture source.
5. Implement the historical feature store in `modeling/build_upcoming_features.py` before production use.
6. Run `create_and_run_upcoming_block.bat all`.

The connector intentionally does not bypass anti-bot controls or scrape restricted sites. The displayed market difference is informational and is not a wagering recommendation.
