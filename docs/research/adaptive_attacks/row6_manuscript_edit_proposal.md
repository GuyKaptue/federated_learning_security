# Row 6 manuscript edit: resolved proposal v2 (not applied)

Status 2026-11-16: **proposed, not applied**. `latex/manuscript_ieee.tex` is unchanged (hash `3a7bf249…`). Branch B, no re-run. Table XVIII is `t:s2u`, Table XX is `t:s2d`. Every `old` string below was checked to occur exactly once in the current `.tex`.

Choice kept from v1: cells are marked in place (`--$^\dagger$`, `$^\ddagger$`), not removed.

Changes from v1 (2026-11-16): (a) Section X no longer says "the backdoor results are not affected" (not shown for Tables XIX and XXIV); (b) the unverified ALIE and MIN-MAX round counts (2 of 4, 3 of 4) are replaced by "in some rounds"; (c) the closing claim of line 287 is scoped (edit 4b); (d) the corrected-variant sentence names `allfeat_corrected` and says it was not used for the results.

Open choices for you: **A** (the corrected-variant sentence) and **B** (unverified figures); see the response of 2026-11-16 for the recommendations.

## Edit 1a Table XVIII row: withdraw AGR-tailored, mark ALIE and MIN-MAX
old:
```latex
FoolsGold (new) & 0.764$\pm$0.013 & 0.434$\pm$0.064 & 0.763$\pm$0.008 & 0.748$\pm$0.012 & 0.392$\pm$0.015 \\
```
new:
```latex
FoolsGold (new) & 0.764$\pm$0.013 & 0.434$\pm$0.064 & 0.763$\pm$0.008$^\ddagger$ & 0.748$\pm$0.012$^\ddagger$ & --$^\dagger$ \\
```

## Edit 1b Table XX row: withdraw AGR-tailored, mark ALIE and MIN-MAX
old:
```latex
FoolsGold & 0.05/0.04 & 0.35/0.02 & 0.34/0.03 & 0.00/0.06 & 1.00/0.03 & 1.00/0.03 & 1.00/0.03 \\
```
new:
```latex
FoolsGold & 0.05/0.04 & 0.35/0.02$^\ddagger$ & 0.34/0.03$^\ddagger$ & --$^\dagger$ & 1.00/0.03 & 1.00/0.03 & 1.00/0.03 \\
```

## Edit 2a Table XVIII caption footnotes
old:
```latex
New rules are those re-implemented in this stage.}\label{t:s2u}
```
new:
```latex
New rules are those re-implemented in this stage. $^\dagger$ Withdrawn: the AGR-tailored attacker triggered a defect in our FoolsGold implementation (Section~\ref{sec:fgdefect}). $^\ddagger$ Computed with the defective implementation (Section~\ref{sec:fgdefect}); kept for the record, not repaired.}\label{t:s2u}
```

## Edit 2b Table XX caption footnotes
old:
```latex
Untargeted attacks first, then backdoors.}\label{t:s2d}
```
new:
```latex
Untargeted attacks first, then backdoors. $^\dagger$ and $^\ddagger$ as in Table~\ref{t:s2u}.}\label{t:s2d}
```

## Edit 3 Line 287: qualify the 8 of 35
old:
```latex
not a family-wise result over all 35.
```
new:
```latex
not a family-wise result over all 35. Three of the 35 contrasts are FoolsGold under ALIE, MIN-MAX and the AGR-tailored attacker, whose cells are affected by the defect described in Section~\ref{sec:fgdefect}; none is among the 8, and recomputing the Holm correction within those three scenarios without FoolsGold leaves the eight unchanged.
```

## Edit 4a Line 287: the clause that cites the withdrawn number
old:
```latex
RLR (0.407), FoolsGold (0.392) collapse towards the mean (0.384);
```
new:
```latex
RLR (0.407) collapses towards the mean (0.384), and the FoolsGold cell for this attacker is withdrawn (Section~\ref{sec:fgdefect});
```

## Edit 4b Line 287: scope the closing claim to the evidence that remains
old:
```latex
are not expected to stop independent Byzantine updates, which is what we see.
```
new:
```latex
are not expected to stop independent Byzantine updates, which is what we see for both under sign flipping and for RLR under the rule-simulating attacker.
```

## Edit 5 Section X: new paragraph after line 287
old:
```latex
not their general quality.
\begin{table}[t]\centering\caption{Test accuracy (mean$\pm$95\% half-width over 10 seeds), $K=20$, $f=4$ colluding attackers
```
new:
```latex
not their general quality.
\paragraph{A defect in our FoolsGold implementation.}\label{sec:fgdefect} After this stage was run we found that our FoolsGold implementation gives full weight to clients whose accumulated update histories have cosine similarity exactly 1: the logit rescaling then evaluates $\ln 0=-\infty$, and the code replaces non-finite values by 1 instead of 0. Colluders that submit bit-identical updates therefore escape the defence. In a one-seed check (i.i.d., seed 0, 20 rounds) the update chosen by the AGR-tailored attacker triggered this in 20 of 20 rounds (on average 61\% of the 34 candidates it considers do). ALIE and MIN-MAX trigger it in some rounds. Sign flipping did not trigger it in the rounds we checked, and the static, norm-matched and AGR-backdoor attackers did not trigger it in Table~\ref{t:s2d} (rejection 1.00); we did not check the backdoor tables of the later stages for it separately. The AGR-tailored FoolsGold cells are therefore withdrawn and the ALIE and MIN-MAX cells are marked in Tables~\ref{t:s2u} and~\ref{t:s2d}. We did not repeat the affected runs. A corrected variant (\texttt{allfeat\_corrected}) is included in the accompanying archive but was not used for the results above.
\begin{table}[t]\centering\caption{Test accuracy (mean$\pm$95\% half-width over 10 seeds), $K=20$, $f=4$ colluding attackers
```

## Edit 6 Limitations: pointer sentence
old:
```latex
cannot be attributed to the search strategy.

```
new:
```latex
cannot be attributed to the search strategy. Our FoolsGold implementation had a defect that affects some untargeted cells (Section~\ref{sec:fgdefect}); we did not repeat those runs.

```
