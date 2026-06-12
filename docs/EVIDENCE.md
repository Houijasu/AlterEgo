# Feature evidence log

Solo A/B protocol: the in-process side carries `ALTEREGO_ENABLE=<flag>` (plus
optional `ALTEREGO_TUNE`), the opponent is `bin\dev` compiled defaults, nets
matched on both sides. 192 games at 500ms resolve roughly +-20 Elo (1 sigma);
promote only clear positives, one feature at a time (M9 lesson).

| date | commit | flag | tune | games | W-D-L | score | elo | LLR | bench(flag-on) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-12 | eed97e4 | iir |  | 192 | +57=101-34 | 56.0% | +42 | +0.65 | 168588 |
| 2026-06-13 | eed97e4 | histlmr |  | 192 | +50=110-32 | 54.7% | +33 | +0.56 | 266380 |
| 2026-06-13 | eed97e4 | nmp2 |  | 192 | +58=100-34 | 56.2% | +44 | +0.68 | 183323 |
