"""The session briefing: a few lines of orientation built from fixed queries.
It lists no candidates, and the meta agent cannot change it."""
from __future__ import annotations

from .. import config, db, runconfig


def build(agent_key: str, iteration: int, n_agents: int,
          problem: str | None = None,
          rel_iteration: int = 0, total_iterations: int = 0) -> str:
    own = db.run_read(
        """
        MATCH (:Agent {key: $k})-[:RAN]->(:Session)-[:PRODUCED]->(c:Candidate)
        RETURN max(c.fitness) AS best
        """,
        k=agent_key,
    )
    own_best = own[0]["best"] if own else None
    glob = db.run_read("MATCH (c:Candidate) RETURN max(c.fitness) AS best")
    global_best = glob[0]["best"] if glob else None

    # The raw task metric: normalised fitness hides progress on a near-saturated problem.
    best_row = db.run_read(
        """
        MATCH (c:Candidate) WHERE c.objective IS NOT NULL
        RETURN c.objective AS obj, c.objective_name AS name,
               c.objective_target AS target, c.objective_direction AS direction,
               c.fitness AS fitness
        ORDER BY c.fitness DESC LIMIT 1
        """
    )
    counts = db.run_read(
        """
        CALL () { MATCH (c:Candidate) RETURN count(c) AS candidates }
        CALL () { MATCH (i:Idea) RETURN count(i) AS ideas }
        RETURN candidates, ideas
        """
    )[0]

    def fmt(x) -> str:
        return "none yet" if x is None else f"{x:.4g}"

    objective = ""
    if best_row and best_row[0]["obj"] is not None:
        r = best_row[0]
        name = r["name"] or "objective"
        objective = f"Best {name} in the database:  {r['obj']:.9g}"
        if r["target"]:
            # Say "target", never "record", and phrase passing it as a start: a session
            # told it has arrived stops pushing.
            beat = (r["obj"] < r["target"] if r["direction"] == "min"
                    else r["obj"] > r["target"])
            gap = abs(r["obj"] - r["target"]) / r["target"] * 100.0
            objective += (f"   *** past the target of {r['target']:.9g} — it is a bar "
                          f"to beat clearly, not a finish line; push on ***"
                          if beat else
                          f"   (target to beat clearly: {r['target']:.9g}, {gap:.3f}% away)")
        objective += "\n"

    # Every session gets the same brief and is told nothing about its position in the
    # chain; these arguments stay in the signature for the orchestrator only.
    del rel_iteration, total_iterations

    budget = ""
    if problem is not None:
        lim = config.problem_limits(problem)
        if lim.solve_seconds is not None:
            budget = (f"Your solve() may run for {lim.solve_seconds} s "
                      f"(whole evaluation capped at {lim.wall_seconds} s).\n")
        # The session's deadline: the same figure launch.py enforces.
        wall = runconfig.required_agent_timeout(problem)
        budget += (f"This whole session is capped at {wall} s (~{wall / 3600:.1f} h) — "
                   "budget your candidates against it; a session cut off at the cap "
                   "loses its lesson.\n")

    return (
        f"Best fitness in your branch: {fmt(own_best)}\n"
        f"Best in the database:        {fmt(global_best)}\n"
        f"{objective}"
        f"The database holds {counts['candidates']} candidates and {counts['ideas']} ideas.\n"
        f"{n_agents} search agent(s) are running in parallel.\n"
        f"{budget}"
    )
