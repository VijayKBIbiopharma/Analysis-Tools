# Analysis Tools

Statistics, design of experiments and machine learning for process and product
development. Names the right technique, explains the result in plain language,
and says when the data cannot support an answer.

**Status: stage 1 of 8.** The application runs, reads your data and tells you
what is in it. The analysis modules are listed below with what is built and what
is not — nothing is hidden behind a page that turns out to be empty.

## Running it

**On your own machine** — your data never leaves it.

| | |
|---|---|
| Windows | double-click **`run.bat`** |
| Mac / Linux | **`./run.sh`** |

Needs Python 3.10+ installed once. First run sets up and takes about a minute;
every run after that is immediate. Your browser opens at `http://localhost:8501`.

**In a container**

```
docker build -t analysis-tools .
docker run -p 8501:8501 analysis-tools
```

## What is built

| Module | What it answers | Status |
|---|---|---|
| **Explore** | What is in this file: which columns are measurements, which are groups, which are labels, what is missing, and where the reading is uncertain | **built** |
| Compare groups | Which test applies to these data types, and run it | not built |
| Capability | Is the process capable, and is it stable | not built |
| Measurement systems | Gage R&R, and whether the method can see what you are chasing | not built |
| Relationships | Regression, model reduction | not built |
| Design of experiments | Design, analyse, reduce, optimise | not built |
| Cell line development | Vector selection, clone screening | not built |
| Learn | How each method works, and when it does not | not built |

## The idea

A practitioner arrives with a specific question and a specific file. The thing
that decides which statistical test applies is not the question's wording — it
is the *type* of each column. A continuous input against a continuous response
is regression. A two-level group against a continuous response is a t-test.
Three or more levels is one-way ANOVA, or Kruskal-Wallis where the assumptions
fail.

That routing is well established and is usually delivered as a wall chart or a
menu path someone follows by hand. This tool works it out from the data and then
runs the right thing — which is why **Explore** comes first: every later module
depends on each column being correctly classified.

### It asks rather than guessing

A numeric column holding a handful of distinct values is genuinely ambiguous.
Temperature recorded at 30, 35 and 40 is a *variable* if you want a slope
through it and a *three-level attribute* if you want to compare three group
means. Both are defensible; they lead to different tests and different answers.

So the tool gives its reading, says a second reading exists, explains what
changes, and lets you choose. Silently picking one is choosing the user's
analysis for them without telling them.

## Your data

Held in memory for the session and **never written to disk**. Nothing is sent
anywhere. Run it on your own machine and the data does not leave the machine at
all.

That is also what the platform requires: ECP section A.6 asks that applications
not rely on local storage, and section A.5 that logging goes to the console with
no file handler. Both are enforced in code — `analysis_tools/io.py` takes bytes
and never a path it opens for itself, and there is a test that fails if that
changes.

## Dependencies

Six, all mainstream, all pinned exactly: numpy, pandas, scipy, statsmodels,
scikit-learn, streamlit. `requirements.txt` carries a one-line justification for
each, for the dependency review at intake step 4.

Using scipy and statsmodels rather than hand-rolling the statistics is
deliberate. They are the reference implementations that results get checked
against, and a reviewer who knows them does not have to take anyone's word for
the arithmetic.

## Tests

```
pip install -r requirements-dev.txt
pytest -q
```

61 tests. They are not about tidy output — they are about whether the right test
would be chosen. Two examples of what they pin:

- A titer measured to three decimals is unique on every row of a real dataset.
  Uniqueness is what precision looks like, not what an identifier looks like —
  an earlier version classified every continuous response as a label, which
  would have left the tool unable to analyse the one column anyone cared about.
- `Titer, Titer, Titer` becomes `Titer, Titer (2), Titer (3)` and says so.
  pandas renames the duplicates to `Titer.1` on its own, which looks like a
  decimal and hides that the duplication happened — and two columns with the
  same name usually means a join went wrong upstream.

CI additionally starts the application and waits for it to serve, because a
broken entry point passes every unit test and fails on the first page load.

## Contact

Boulder OpEx — vramasubramanian@kbibiopharma.com
