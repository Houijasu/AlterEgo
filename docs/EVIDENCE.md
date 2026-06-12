# Feature evidence log

Solo A/B protocol: the in-process side carries `ALTEREGO_ENABLE=<flag>` (plus
optional `ALTEREGO_TUNE`), the opponent is `bin\dev` compiled defaults, nets
matched on both sides. 192 games at 500ms resolve roughly +-20 Elo (1 sigma);
promote only clear positives, one feature at a time (M9 lesson).

| date | commit | flag | tune | games | W-D-L | score | elo | LLR | bench(flag-on) |
|---|---|---|---|---|---|---|---|---|---|
