// AgentDiscover graph schema. One Neo4j database per problem run, shared by all agents.
//
// Neo4j has no fixed schema: any node may hold any property. So this file is the whole
// contract. It does two things — it declares the keys the database enforces, and it
// documents the structure that the write service and the agents must keep to.
//
// Run once at bootstrap. Every statement is idempotent.


// ============================================================================
// NODES
// ============================================================================
//
// User      name                                     the person who owns the agents
// Agent     key = "<user_name>/<agent_name>"         a persistent identity owning one worktree
//           name, worktree
// Session   id  = "<user_name>/<agent_name>/<iter>"  one fresh agent invocation
//           iteration, model, briefing, thinking_text, jsonl_path, started_at, finished_at,
//           token_hash      (sha256 fingerprint of the session's secret call token; the
//                            service attributes calls by matching this fingerprint, so it
//                            never trusts what an agent claims about itself. Only the
//                            fingerprint is stored — reads are unrestricted, and a raw
//                            token in the graph would let any agent impersonate any other)
//           lesson_learned  (written by the search agent, about its own results)
//           meta_lesson     (written by the meta agent, about how well it searched)
// Candidate genome_hash                              one node per unique program
//           genome            the full solution.py text that was submitted
//           linguistic_prediction, fitness_prediction   what the agent expected, written
//                             before the score was known
//           fitness           the normalised score the search ranks on
//           eval_status       "done" | "failed"
//           created_at        when it was submitted (use this to order a session's
//                             candidates; there is no sequence number)
//           plus one property per evaluator metric (written with SET c += $metrics).
//           Every evaluator on a scored problem publishes:
//             objective            the raw task quantity, at full precision — the
//                                  number to reason about; fitness is derived from it
//             objective_name       what it is called, e.g. "combined_score", "C5 bound"
//             objective_direction  "min" or "max"
//             objective_target     the target from PROBLEM.md
//             stage                "full" | "invalid" | "error" | "timeout"
//             time, solve_time     seconds for the whole evaluation / the candidate's
//                                  own call
//             error                present only when the candidate did not score
//           Beyond those, each problem adds its own — e.g. `merit_factor` (labs),
//           `c5_bound` and `n_points` (erdos_over), `mean_gap_db` (jscc_2to1). To see
//           what THIS problem publishes, read the keys of a scored candidate:
//             MATCH (c:Candidate {eval_status:'done'}) RETURN keys(c) LIMIT 1
// Idea      name                                     an idea, reused across candidates
//           description
// Resource  name                                     a file or other context source
//           path, take_away
// Guidance  id                                       one meta-agent strategy rewrite
//           strategy, diff_summary
// Tool      (name, branch)                           a tool the meta agent wrote for one branch
//           description, code, kind (cypher | python | shell),
//           params (JSON Schema for the tool's inputs, so the service offers it to the
//                   branch's agents exactly like a built-in tool)
//
// Notes
//   - Candidate is keyed on the genome, so a repeated genome never makes a second node.
//   - fitness and the metrics are genome-level facts, written once by the evaluator.
//   - Metric keys vary by outcome (score/solve_time on success, stage/error on failure).
//     That is why they are plain properties and not a fixed table.


// ============================================================================
// RELATIONSHIPS
// ============================================================================
//
// (:User)-[:OWNS]->(:Agent)
// (:Agent)-[:RAN]->(:Session)
// (:Agent)-[:HAS_GUIDANCE]->(:Guidance)
// (:Session)-[:PRODUCED]->(:Candidate)
// (:Session)-[:UNDER_GUIDANCE]->(:Guidance)
// (:Candidate)-[:DERIVED_FROM {kind}]->(:Candidate)
//        kind = mutation | crossover | repair | inspired
// (:Candidate)-[:USES_IDEA]->(:Idea)
// (:Candidate)-[:USED_CONTEXT]->(:Resource)
// (:Idea)-[:RELATED_TO {how}]->(:Idea)      read with no arrow; the link is not directional
// (:Guidance)-[:SUPERSEDES]->(:Guidance)
// (:Agent)-[:HAS_TOOL]->(:Tool)             only this branch's agents are offered it
// (:Session)-[:USED_TOOL]->(:Tool)          written when a session actually calls it
//
// Exposure — facts, written by code from the session log, never by the agent:
// (:Session)-[:RETRIEVED]->(:Candidate)
// (:Session)-[:READ]->(:Resource)
//
// Sessions hold what happened. Candidates hold what the agent claimed. The difference
// between the two is itself a signal.


// ============================================================================
// CONSTRAINTS — the enforced keys
// ============================================================================
// A uniqueness constraint also creates an index, so these keys need no separate index.

CREATE CONSTRAINT user_name       IF NOT EXISTS FOR (n:User)      REQUIRE n.name        IS UNIQUE;
CREATE CONSTRAINT agent_key       IF NOT EXISTS FOR (n:Agent)     REQUIRE n.key         IS UNIQUE;
CREATE CONSTRAINT session_id      IF NOT EXISTS FOR (n:Session)   REQUIRE n.id          IS UNIQUE;
CREATE CONSTRAINT candidate_hash  IF NOT EXISTS FOR (n:Candidate) REQUIRE n.genome_hash IS UNIQUE;
CREATE CONSTRAINT idea_name       IF NOT EXISTS FOR (n:Idea)      REQUIRE n.name        IS UNIQUE;
CREATE CONSTRAINT resource_name   IF NOT EXISTS FOR (n:Resource)  REQUIRE n.name        IS UNIQUE;
CREATE CONSTRAINT guidance_id     IF NOT EXISTS FOR (n:Guidance)  REQUIRE n.id          IS UNIQUE;
CREATE CONSTRAINT tool_key        IF NOT EXISTS FOR (n:Tool)      REQUIRE (n.name, n.branch) IS UNIQUE;
// One RunConfig per problem: the settings the current run was launched with. Written by
// the orchestrator at run start, read by every process (see agentdiscover/runconfig.py) so the
// long-lived MCP service can never act on a stale copy from the shell that started it.
CREATE CONSTRAINT runconfig_problem IF NOT EXISTS FOR (n:RunConfig) REQUIRE n.problem IS UNIQUE;


// ============================================================================
// INDEXES — the properties that queries filter on
// ============================================================================

// Leaderboards and every "beat this score" query.
CREATE INDEX candidate_fitness IF NOT EXISTS FOR (n:Candidate) ON (n.fitness);
