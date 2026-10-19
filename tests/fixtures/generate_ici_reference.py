"""Regenerate ICI reference data with R's survival package (no R needed by pytest)."""

import json
import subprocess
from pathlib import Path

import numpy as np

# RCS is the truncated-power basis from Hmisc::rcspline.eval, with explicit
# percentile knots. Supplying knots avoids Hmisc's small-sample/tie heuristics.
R_REFERENCE = r"""
library(survival)
d <- read.table(text="INPUT", col.names=c("time", "status", "pred"))
nk <- KNOT_COUNT
x <- log(-log1p(-d$pred))
outer <- if (nk == 3) .1 else .05
k <- as.numeric(quantile(x, seq(outer, 1-outer, length.out=nk)))
rcs <- function(x) {
    a <- k[nk-1]
    b <- k[nk]
    nonlinear <- sapply(seq_len(nk-2), function(j) {
        (pmax(x-k[j], 0)^3 +
         ((a-k[j])*pmax(x-b, 0)^3 - (b-k[j])*pmax(x-a, 0)^3)/(b-a))/(b-k[1])^2
    })
    ans <- data.frame(cbind(x, nonlinear))
    names(ans) <- paste0("z", seq_len(nk-1))
    ans
}
z <- rcs(x)
d <- cbind(d, z)
terms <- paste(names(z), collapse=" + ")
if (COMPETING) {
    d$state <- factor(d$status, levels=0:3, labels=c("censor", "target", "comp2", "comp3"))
    fg <- finegray(as.formula(paste("Surv(time, state) ~", terms)),
                   data=d, etype="target", timefix=FALSE)
    fit <- coxph(as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~", terms)),
                 data=fg, weights=fgwt, ties="efron", control=coxph.control(eps=1e-11))
} else {
    fit <- coxph(as.formula(paste("Surv(time, status == 1) ~", terms)),
                 data=d, ties="efron", control=coxph.control(eps=1e-11))
}
grid <- seq(quantile(d$pred, .01), quantile(d$pred, .99), length.out=100)
baseline <- basehaz(fit, centered=FALSE)
h <- tail(baseline$hazard[baseline$time <= 3], 1)
observed <- -expm1(-h * exp(as.matrix(z) %*% coef(fit)))
curve <- -expm1(-h * exp(as.matrix(rcs(log(-log1p(-grid)))) %*% coef(fit)))
error <- abs(d$pred - observed)
metrics <- c(mean(error), median(error), quantile(error, .9), max(error))
write.table(c(observed, curve, metrics), row.names=FALSE, col.names=FALSE)
"""


def generate() -> None:
    """
    Save deterministic inputs and independently calculated R reference outputs.

    Returns
    -------
    None
        Write ici_reference.json next to this script.
    """
    rng = np.random.default_rng(4021)
    preds = rng.uniform(0.02, 0.9, 180)
    times = rng.exponential(5 * np.exp(-1.5 * preds))
    times[:3] = 0
    labels = rng.choice([0, 1, 2, 3], times.size, p=[0.25, 0.4, 0.25, 0.1])
    version = subprocess.run(
        ["Rscript", "-e", 'cat(as.character(packageVersion("survival")))'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result = {
        "source": "R survival::finegray / coxph, Efron ties and baseline",
        "survival_version": version,
        "datasets": [],
    }
    for competing in (False, True):
        for tied in (False, True):
            event_time = np.floor(times) if tied else times
            events = labels if competing else (labels > 0).astype(int)
            dataset = {
                "competing": competing,
                "tied": tied,
                "target_time": 3.0,
                "preds": preds.tolist(),
                "event_time": event_time.tolist(),
                "event_indicator": events.tolist(),
                "fits": [],
            }
            rows = "\n".join(
                " ".join(map(str, row))
                for row in np.column_stack((event_time, events, preds))
            )
            for knots in (3, 4, 5):
                code = (
                    R_REFERENCE.replace("INPUT", rows)
                    .replace("KNOT_COUNT", str(knots))
                    .replace("COMPETING", str(competing).upper())
                )
                proc = subprocess.run(
                    ["Rscript", "-"],
                    input=code,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                output = np.fromstring(proc.stdout, sep=" ")
                assert output.size == preds.size + 104, proc.stdout
                dataset["fits"].append(
                    {
                        "knots": knots,
                        "calibrated": output[: preds.size].tolist(),
                        "curve": output[preds.size : -4].tolist(),
                        "metrics": output[-4:].tolist(),
                    }
                )
            result["datasets"].append(dataset)
    Path(__file__).with_name("ici_reference.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    generate()
