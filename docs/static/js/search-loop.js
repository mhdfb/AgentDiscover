/* The search loop, animated.
   One scene, four beats: start → query → experiment → submit, then back to
   query. Everything drawn is a pure function of the clock, so pausing and
   stepping are exact. The example follows the paper's walkthrough figure:
   place 15 circles in the unit square and maximise the sum of their radii. */
(function () {
  'use strict';

  const root = document.getElementById('search-loop');
  if (!root) return;
  const stage = root.querySelector('.loop-stage');
  const captionEl = root.querySelector('.loop-caption');
  const railWrap = root.querySelector('.loop-rail-wrap');
  const railBtns = Array.from(root.querySelectorAll('.loop-rail button'));
  const backSvg = root.querySelector('.loop-back');
  const playBtn = root.querySelector('.loop-play');
  const prevBtn = root.querySelector('.loop-prev');
  const nextBtn = root.querySelector('.loop-next');

  const NS = 'http://www.w3.org/2000/svg';
  const W = 960, H = 500;
  let svg;

  function el(tag, attrs, parent, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    (parent || svg).appendChild(n);
    return n;
  }
  // SVG collapses runs of spaces; keep column alignment with no-break spaces.
  const keepSpaces = s => s.replace(/ {2,}/g, m => ' '.repeat(m.length));

  const clamp01 = x => Math.max(0, Math.min(1, x));
  const seg = (p, a, b) => clamp01((p - a) / (b - a));
  const ease = x => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  const easeOut = x => 1 - Math.pow(1 - x, 3);
  const lerp = (a, b, q) => a + (b - a) * q;

  // ── timeline ──────────────────────────────────────────────────────────
  const DUR = [3000, 9800, 8800, 13500];
  const START = DUR.map((_, i) => DUR.slice(0, i).reduce((a, b) => a + b, 0));
  const TOTAL = START[3] + DUR[3];
  const CAPTIONS = [
    'A coding agent starts in a worktree. Its CLAUDE.md states the problem and briefs it on what the database already holds.',
    'The agent writes its own queries against its long-term memory, whatever it needs to know, and the database answers them.',
    'It plans what to do next: use its tools or write new ones, run experiments, and propose a new candidate.',
    'It submits the candidate. The server scores it, writes back a steering message, and records the result in the graph. Then the agent goes back to step 1.',
  ];

  svg = el('svg', {
    viewBox: `0 0 ${W} ${H}`, class: 'loop-svg', role: 'img',
    'aria-label': 'A search agent queries a graph database of past ideas and candidates, writes a tool and runs an experiment, submits a candidate, and receives a score with a steering message while the result is recorded in the graph.'
  }, stage);

  // Arrowheads. Markers do not inherit the path's colour, so one per colour.
  const defs = el('defs', {});
  for (const [id, cls] of [['arr-faint', 'mk-faint'], ['arr-mute', 'mk-mute'], ['arr-ink', 'mk-ink'], ['arr-server', 'mk-server']]) {
    const m = el('marker', {
      id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7,
      orient: 'auto-start-reverse', class: cls
    }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z' }, m);
  }

  // ── the two boxes ─────────────────────────────────────────────────────
  function box(x, y, w, h, cls, title, aside) {
    const g = el('g', { class: 'panel ' + cls });
    el('rect', { x, y, width: w, height: h, rx: 14, class: 'panel-bg' }, g);
    el('text', { x: x + 20, y: y + 30, class: 'panel-title' }, g, title);
    el('text', { x: x + w - 20, y: y + 30, class: 'panel-aside', 'text-anchor': 'end' }, g, aside);
    return g;
  }
  const gAgent = box(16, 44, 408, 440, 'p-agent', 'Search agent', 'sandbox');
  const gMem = box(536, 44, 408, 440, 'p-memory', 'Long-term memory', 'MCP server');

  // ── agent: the context log ────────────────────────────────────────────
  // One accumulating log. When it fills, older lines scroll up out of view.
  const PANE = { x: 32, y: 92, w: 376, h: 204 };
  const clip = el('clipPath', { id: 'pane-clip' }, defs);
  el('rect', { x: PANE.x, y: PANE.y, width: PANE.w, height: PANE.h, rx: 8 }, clip);
  el('rect', { x: PANE.x, y: PANE.y, width: PANE.w, height: PANE.h, rx: 8, class: 'pane-bg' }, gAgent);
  const gLog = el('g', { 'clip-path': 'url(#pane-clip)' }, gAgent);
  const LH = 19, CAP = 9, ROW0 = 116;

  // The briefing is a file the agent reads, drawn as a card; the rest is
  // the agent's own transcript.
  const fileCard = el('rect', { x: 38, width: 364, rx: 6, class: 'file-card' }, gLog);
  const fileBar = el('rect', { x: 38, width: 3, class: 'file-bar' }, gLog);

  // line: appears whole; type: typed from `at` to `until`;
  // thought: the thinking indicator from `at` to `until`, then the text.
  const L = (s, at, text, cls) => ({ kind: 'line', s, at, text, cls: cls || '' });
  const F = (text, cls) => ({ kind: 'line', s: 0, at: 0, text, cls: cls || '', file: true });
  const Y = (s, at, until, text, cls) => ({ kind: 'type', s, at, until, text, cls });
  const TH = (s, at, until, text) => ({ kind: 'thought', s, at, until, text, cls: 'think' });
  const GAP = s => ({ kind: 'gap', s, at: 0 });

  const LOG = [
    // start: the whole briefing is there from the first frame
    F('CLAUDE.md', 'file-name'),
    F('Problem   15 circles in the unit square, maximise Σr'),
    F('Briefing  best 1.91 · target 1.98'),
    F('          22 ideas and 25 candidates so far'),
    F('Task      generate 10 candidates'),
    // query
    GAP(1),
    TH(1, 0.00, 0.18, "Let's get the best candidate per idea."),
    Y(1, 0.21, 0.32, 'MATCH (c:Candidate)-[:USES_IDEA]->(i:Idea)', 'code'),
    Y(1, 0.32, 0.38, 'RETURN i, max(c.fitness)', 'code'),
    L(1, 0.80, '← A 1.88 · B 1.91 · C 1.86', 'in'),
    TH(1, 0.84, 0.94, 'Idea C is under-explored; 1.86 can be improved.'),
    // experiment
    GAP(2),
    TH(2, 0.00, 0.15, "Let's write a tool: hex layout, refined with SLSQP."),
    Y(2, 0.18, 0.28, '› write tools/optimize.py', 'h'),
    Y(2, 0.32, 0.42, '$ python3 tools/optimize.py', 'code'),
    L(2, 0.42, 'step 0     Σr 1.86'),
    L(2, 0.56, 'step 100   Σr 1.91'),
    L(2, 0.74, 'step 200   Σr 1.97'),
    TH(2, 0.78, 0.88, 'Circles collide after 200. Submit this one.'),
    // submit
    GAP(3),
    Y(3, 0.00, 0.07, '› submit_candidate  (idea C, parent 1.86)', 'h'),
    L(3, 0.82, '← Σr = 1.970 · new best · candidate 1 of 10', 'in'),
    L(3, 0.86, '← refine, or switch idea if stuck', 'in b'),
    TH(3, 0.88, 0.95, 'On to candidate 2 of 10.'),
  ];
  for (const e of LOG) {
    e.t0 = START[e.s] + e.at * DUR[e.s];
    e.t1 = e.until != null ? START[e.s] + e.until * DUR[e.s] : null;
    if (e.kind === 'gap') continue;
    e.text = keepSpaces(e.text);
    e.node = el('text', { x: e.file ? 52 : 44, y: 0, class: 'ln ' + e.cls }, gLog, e.kind === 'type' ? '' : e.text);
    e.node.style.opacity = 0;
  }

  // The thinking indicator, shared by every thought.
  const gThink = el('g', { class: 'think' }, gLog);
  gThink.style.opacity = 0;
  const spin = el('circle', { r: 5, class: 'spin' }, gThink);
  const thinkT = el('text', { x: 12, y: 4, class: 'think-t' }, gThink, 'Thinking');

  // The typing cursor, shared by whichever line is being typed.
  const cursor = el('rect', { width: 1.6, height: 12, class: 'cursor' }, gLog);
  cursor.style.opacity = 0;

  function renderLog(t, tms) {
    let count = 0, lastT0 = -Infinity, addedLast = 0;
    for (const e of LOG) {
      if (e.t0 > t) continue;
      count++;
      if (e.t0 > lastT0) { lastT0 = e.t0; addedLast = 1; } else if (e.t0 === lastT0) addedLast++;
    }
    const over = Math.max(0, count - CAP);
    const off = Math.max(0, over - addedLast + addedLast * easeOut(seg(t - lastT0, 0, 420))) * LH;

    let typing = null, thinkOn = false, i = 0, fileTop = null, fileBottom = 0;
    for (const e of LOG) {
      const on = e.t0 <= t;
      if (e.kind === 'gap') { if (on) i++; continue; }
      if (!on) { e.node.style.opacity = 0; continue; }
      const y = ROW0 + i * LH - off;
      i++;
      e.node.setAttribute('y', y);
      if (e.file) { if (fileTop == null) fileTop = y; fileBottom = y; }
      if (e.kind === 'type') {
        const q = seg(t, e.t0, e.t1);
        e.node.textContent = e.text.slice(0, Math.round(q * e.text.length));
        if (q < 1) typing = e.node;
        e.node.style.opacity = 1;
      } else if (e.kind === 'thought' && t < e.t1) {
        thinkOn = true;
        gThink.setAttribute('transform', `translate(50 ${y - 4})`);
        e.node.style.opacity = 0;
      } else {
        e.node.style.opacity = 1;
      }
    }
    if (fileTop != null) {
      const top = fileTop - 13, h = fileBottom - fileTop + LH - 1;
      for (const r of [fileCard, fileBar]) { r.setAttribute('y', top); r.setAttribute('height', h); }
    }
    gThink.style.opacity = thinkOn ? 1 : 0;
    if (thinkOn) {
      spin.setAttribute('transform', `rotate(${(tms / 900 * 360) % 360})`);
      thinkT.textContent = 'Thinking' + '.'.repeat(1 + Math.floor(tms / 350) % 3);
    }
    return typing;
  }

  // ── agent: the packing being optimised ────────────────────────────────
  const PK = { x: 32, y: 318, s: 140 };
  const gPack = el('g', { class: 'pack' }, gAgent);
  el('rect', { x: PK.x, y: PK.y, width: PK.s, height: PK.s, rx: 6, class: 'pack-bg' }, gPack);
  // Three packings of 15 circles, progressively tighter (x, y, r in the unit square).
  const PACKS = [
    [[0.107,0.118,0.096],[0.399,0.114,0.111],[0.617,0.106,0.105],[0.888,0.143,0.112],[0.223,0.351,0.16],[0.518,0.375,0.133],[0.783,0.411,0.135],[0.134,0.622,0.124],[0.438,0.628,0.09],[0.616,0.599,0.09],[0.877,0.629,0.102],[0.118,0.863,0.118],[0.361,0.871,0.122],[0.623,0.867,0.111],[0.839,0.881,0.106]],
    [[0.105,0.108,0.105],[0.392,0.115,0.115],[0.62,0.123,0.113],[0.888,0.143,0.112],[0.223,0.351,0.161],[0.518,0.375,0.133],[0.783,0.411,0.135],[0.129,0.621,0.125],[0.405,0.593,0.109],[0.624,0.605,0.11],[0.882,0.652,0.118],[0.118,0.863,0.118],[0.375,0.844,0.137],[0.623,0.867,0.112],[0.841,0.888,0.107]],
    [[0.105,0.108,0.105],[0.39,0.125,0.119],[0.63,0.143,0.123],[0.877,0.14,0.123],[0.223,0.351,0.162],[0.518,0.375,0.135],[0.795,0.393,0.142],[0.129,0.621,0.125],[0.369,0.585,0.114],[0.614,0.624,0.132],[0.874,0.655,0.126],[0.118,0.863,0.118],[0.373,0.838,0.139],[0.626,0.872,0.115],[0.849,0.891,0.109]],
  ];
  const circles = PACKS[0].map(() => el('circle', { class: 'pack-c' }, gPack));
  function drawPack(a, b, q) {
    for (let i = 0; i < circles.length; i++) {
      const A = PACKS[a][i], B = PACKS[b][i];
      circles[i].setAttribute('cx', PK.x + lerp(A[0], B[0], q) * PK.s);
      circles[i].setAttribute('cy', PK.y + (1 - lerp(A[1], B[1], q)) * PK.s);
      circles[i].setAttribute('r', lerp(A[2], B[2], q) * PK.s);
    }
  }
  el('text', { x: 186, y: 334, class: 'ro-label' }, gAgent, 'Σr · radii');
  const sigmaEl = el('text', { x: 184, y: 374, class: 'ro-big' }, gAgent, '—');
  const sigmaSub = el('text', { x: 186, y: 394, class: 'ro-sub' }, gAgent, '');

  // ── agent: the worktree ───────────────────────────────────────────────
  el('text', { x: 290, y: 334, class: 'ro-label' }, gAgent, 'worktree/');
  function fileIcon(g, x, y) {
    el('path', { d: `M${x},${y - 10} h7 l3,3 v9 h-10 z`, class: 'fi' }, g);
    el('path', { d: `M${x + 7},${y - 10} v3 h3`, class: 'fi-fold' }, g);
  }
  function folderIcon(g, x, y) {
    el('rect', { x, y: y - 10.5, width: 6, height: 4, rx: 1, class: 'fo' }, g);
    el('rect', { x, y: y - 8.5, width: 14, height: 10, rx: 1.5, class: 'fo' }, g);
  }
  const gTree = el('g', {}, gAgent);
  fileIcon(gTree, 290, 357);
  el('text', { x: 306, y: 357, class: 'wt' }, gTree, 'CLAUDE.md');
  folderIcon(gTree, 290, 377);
  el('text', { x: 310, y: 377, class: 'wt' }, gTree, 'tools/');
  const gToolRow = el('g', { class: 'wt-row' }, gAgent);
  fileIcon(gToolRow, 302, 397);
  el('text', { x: 318, y: 397, class: 'wt' }, gToolRow, 'optimize.py');
  const toolStatus = el('text', { x: 302, y: 424, class: 'tool-status' }, gAgent, '');

  function renderTree(toolQ, running, tms) {
    gToolRow.style.opacity = toolQ;
    gToolRow.setAttribute('transform', `translate(${-8 * (1 - easeOut(toolQ))} 0)`);
    gToolRow.classList.toggle('running', running);
    toolStatus.textContent = running ? 'running' + '.'.repeat(1 + Math.floor(tms / 350) % 3) : '';
  }

  // ── memory: graph database ────────────────────────────────────────────
  el('rect', { x: 552, y: 92, width: 376, height: 180, rx: 8, class: 'sub-bg' }, gMem);
  el('text', { x: 564, y: 110, class: 'sub-t' }, gMem, 'Graph database');
  const gGraph = el('g', { class: 'graph' }, gMem);

  const IDEAS = { A: [610, 134], B: [740, 134], C: [870, 134] };
  const CANDS = [
    { id: 'a1', x: 590, y: 186, v: '1.62', idea: 'A', lx: -14, ly: 4, end: true },
    { id: 'a2', x: 640, y: 178, v: '1.71', idea: 'A', lx: 13, ly: -5 },
    { id: 'a3', x: 656, y: 214, v: '1.75', idea: 'A', lx: 13, ly: 4 },
    { id: 'a4', x: 612, y: 240, v: '1.88', idea: 'A', lx: -14, ly: 4, end: true },
    { id: 'b1', x: 772, y: 194, v: '1.80', idea: 'B', lx: 13, ly: 4 },
    { id: 'b2', x: 744, y: 242, v: '1.91', idea: 'B', lx: -14, ly: 4, end: true },
    { id: 'c1', x: 870, y: 208, v: '1.86', idea: 'C', lx: 14, ly: 4 },
    { id: 'c2', x: 892, y: 246, v: '1.97', idea: 'C', isNew: true, lx: -14, ly: 4, end: true },
  ];
  const PARENTS = [['a1', 'a2'], ['a2', 'a3'], ['a3', 'a4'], ['a3', 'b2'], ['b1', 'b2'], ['c1', 'c2']];
  const byId = Object.fromEntries(CANDS.map(c => [c.id, c]));
  const edgeEl = {};

  el('line', { x1: 621, y1: 134, x2: 729, y2: 134, class: 'e-ideas' }, gGraph);
  el('line', { x1: 751, y1: 134, x2: 859, y2: 134, class: 'e-ideas' }, gGraph);
  for (const c of CANDS) {
    const [ix, iy] = IDEAS[c.idea];
    edgeEl[c.idea + '>' + c.id] = el('line', {
      x1: ix, y1: iy + 11, x2: c.x, y2: c.y, class: 'e-idea' + (c.isNew ? ' new' : '')
    }, gGraph);
  }
  for (const [p, q] of PARENTS) {
    const a = byId[p], b = byId[q];
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
    edgeEl[p + '>' + q] = el('line', {
      x1: a.x + ux * 11, y1: a.y + uy * 11, x2: b.x - ux * 12, y2: b.y - uy * 12,
      class: 'e-par' + (b.isNew ? ' new' : ''), 'marker-end': 'url(#arr-mute)'
    }, gGraph);
  }
  const ideaEl = {};
  for (const k in IDEAS) {
    const [x, y] = IDEAS[k];
    ideaEl[k] = el('rect', { x: x - 11, y: y - 11, width: 22, height: 22, rx: 4, class: 'idea-b' }, gGraph);
    el('text', { x, y: y + 4.5, class: 'idea-l' }, gGraph, k);
  }
  for (const c of CANDS) {
    c.node = el('circle', { cx: c.x, cy: c.y, r: 9, class: 'cand' + (c.isNew ? ' new' : '') }, gGraph);
    c.label = el('text', {
      x: c.x + c.lx, y: c.y + c.ly, class: 'cand-l' + (c.isNew ? ' new' : ''),
      'text-anchor': c.end ? 'end' : 'start'
    }, gGraph, c.v);
  }
  // legend, and the query plan while a query runs
  el('rect', { x: 564, y: 254, width: 10, height: 10, rx: 2, class: 'idea-b' }, gMem);
  el('text', { x: 580, y: 263, class: 'legend' }, gMem, 'idea');
  el('circle', { cx: 628, cy: 259, r: 5, class: 'cand' }, gMem);
  el('text', { x: 640, y: 263, class: 'legend' }, gMem, 'candidate');
  el('circle', { cx: 716, cy: 259, r: 5, class: 'cand best' }, gMem);
  el('text', { x: 728, y: 263, class: 'legend' }, gMem, 'best per idea');
  const planEl = el('text', { x: 916, y: 263, class: 'plan', 'text-anchor': 'end' }, gMem, '');

  // The new node and its two edges grow in when the server records the result.
  const newEdge = edgeEl['c1>c2'], newIdeaEdge = edgeEl['C>c2'];
  const newEdgeLen = Math.hypot(+newEdge.getAttribute('x2') - +newEdge.getAttribute('x1'),
                                +newEdge.getAttribute('y2') - +newEdge.getAttribute('y1'));
  newEdge.style.strokeDasharray = newEdgeLen;
  function renderNewNode(g) {
    newIdeaEdge.style.opacity = seg(g, 0, 0.3);
    newEdge.style.opacity = g > 0 ? 1 : 0;
    newEdge.style.strokeDashoffset = newEdgeLen * (1 - easeOut(seg(g, 0, 0.45)));
    newEdge.setAttribute('marker-end', g >= 0.45 ? 'url(#arr-mute)' : 'none');
    const r = 9 * easeOut(seg(g, 0.35, 0.8));
    byId.c2.node.setAttribute('r', r);
    byId.c2.node.style.opacity = r > 0 ? 1 : 0;
    byId.c2.label.style.opacity = seg(g, 0.75, 1);
  }

  // The query, executed the way a graph database executes
  //   MATCH (c:Candidate)-[:USES_IDEA]->(i:Idea) RETURN i, max(c.fitness)
  // anchor on the Idea nodes, expand every USES_IDEA edge to its candidate,
  // then aggregate per idea. Parent edges play no part in this query.
  const EXPAND = CANDS.filter(c => !c.isNew).map(c => {
    const [ix, iy] = IDEAS[c.idea];
    const len = Math.hypot(c.x - ix, c.y - iy - 11);
    return { c, from: [ix, iy + 11], to: [c.x, c.y], len, e: edgeEl[c.idea + '>' + c.id],
             dot: el('circle', { r: 4, class: 'tdot' }, gGraph) };
  });
  const maxLen = Math.max(...EXPAND.map(x => x.len));
  const PLAN = ['scan Idea nodes', 'expand USES_IDEA', 'max(fitness) per idea'];

  function renderGraph(s, p) {
    let u = -1, settled = false, best = [], grow = 0;
    if (s === 1) {
      if (p >= 0.48) u = Math.min(1, (p - 0.48) / 0.24);
      settled = p >= 0.80;
      if (p >= 0.72) best = ['a4', 'b2', 'c1'];
    } else if (s >= 2) {
      u = 1; settled = true;
      grow = s === 3 ? seg(p, 0.58, 0.72) : 0;
      best = grow >= 1 ? ['a4', 'b2', 'c2'] : ['a4', 'b2', 'c1'];
    }
    const running = u >= 0 && !settled;
    for (const k in ideaEl) ideaEl[k].classList.toggle('lit', running && u < 0.65);
    for (const x of EXPAND) {
      // every edge is expanded at once; longer edges simply take longer
      const q = running ? seg(u, 0.2, 0.2 + 0.45 * x.len / maxLen) : 0;
      x.e.classList.toggle('lit', running && q > 0);
      x.c.node.classList.toggle('seen', running && q >= 1);
      const travelling = running && q > 0 && q < 1;
      x.dot.style.opacity = travelling ? 1 : 0;
      if (travelling) {
        const k = easeOut(q);
        x.dot.setAttribute('cx', lerp(x.from[0], x.to[0], k));
        x.dot.setAttribute('cy', lerp(x.from[1], x.to[1], k));
      }
    }
    planEl.textContent = !running ? '' : u < 0.2 ? PLAN[0] : u < 0.65 ? PLAN[1] : PLAN[2];
    for (const c of CANDS) {
      const b = best.includes(c.id);
      c.node.classList.toggle('best', b);
      c.label.classList.toggle('best', b);
    }
    renderNewNode(grow);
  }

  // ── memory: evaluator and steering ────────────────────────────────────
  el('rect', { x: 552, y: 326, width: 112, height: 136, rx: 8, class: 'sub-bg' }, gMem);
  const gEval = el('g', { class: 'eval' }, gMem);
  el('circle', { cx: 608, cy: 382, r: 17, class: 'eval-icon' }, gEval);
  el('path', { d: 'M599,382 l6,6 l12,-13', class: 'eval-check' }, gEval);
  el('text', { x: 608, y: 418, class: 'eval-t' }, gMem, 'Evaluator');
  el('text', { x: 608, y: 434, class: 'eval-s' }, gMem, 'scores candidates');
  const evalStatus = el('text', { x: 608, y: 452, class: 'eval-status' }, gMem, '');

  el('rect', { x: 680, y: 326, width: 248, height: 136, rx: 8, class: 'sub-bg' }, gMem);
  el('text', { x: 692, y: 346, class: 'sub-t' }, gMem, 'Steering');
  const STEER = [
    'You submitted candidate 1 of 10.',
    'Score: Σr = 1.970, a new best.',
    'Target 1.98 is 0.5% away.',
    'Refine, or switch idea if stuck.',
    'Go for the next one.',
  ];
  const steerNodes = STEER.map((s, i) => el('text', { x: 692, y: 368 + i * 16, class: 'steer-l' + (i >= 3 ? ' b' : '') }, gMem, ''));

  // Types `texts` into `nodes` in order; returns the node the cursor sits on.
  function typeSeq(nodes, texts, q) {
    const total = texts.reduce((a, s) => a + s.length, 0);
    let left = Math.round(q * total), onNode = null;
    texts.forEach((s, i) => {
      const n = Math.max(0, Math.min(s.length, left));
      nodes[i].textContent = s.slice(0, n);
      nodes[i].style.opacity = n > 0 ? 1 : 0;
      if (n > 0) onNode = nodes[i];
      left -= s.length;
    });
    return q > 0 && q < 1 ? onNode : null;
  }

  // The record arrow: the server writes the scored candidate into the graph.
  const recPath = el('path', { d: 'M804,326 V272', class: 'arrow', 'marker-end': 'url(#arr-faint)' }, gMem);
  el('text', { x: 812, y: 303, class: 'arrow-l', 'text-anchor': 'start' }, gMem, 'record');

  // ── arrows between the boxes ──────────────────────────────────────────
  const ARROWS = {
    q:  { d: 'M424,150 H536', label: 'query',         lx: 480, ly: 141 },
    qr: { d: 'M536,186 H424', label: 'results',       lx: 480, ly: 205 },
    s:  { d: 'M424,380 H536', label: 'submit',        lx: 480, ly: 371 },
    sr: { d: 'M536,416 H424', label: 'score + steer', lx: 480, ly: 435 },
  };
  for (const k in ARROWS) {
    const a = ARROWS[k];
    a.path = el('path', { d: a.d, class: 'arrow', 'marker-end': 'url(#arr-faint)' });
    el('text', { x: a.lx, y: a.ly, class: 'arrow-l' }, svg, a.label);
  }
  ARROWS.rec = { path: recPath };
  function arrowOn(key) {
    for (const k in ARROWS) {
      const on = k === key;
      ARROWS[k].path.classList.toggle('on', on);
      ARROWS[k].path.setAttribute('marker-end', on ? 'url(#arr-ink)' : 'url(#arr-faint)');
    }
  }
  const dot = el('circle', { r: 5.5, class: 'dot' });
  dot.style.opacity = 0;

  function moveDot(key, a, b, p, cls) {
    if (p < a || p > b) return false;
    const path = ARROWS[key].path;
    const pt = path.getPointAtLength(path.getTotalLength() * ease(seg(p, a, b)));
    dot.setAttribute('cx', pt.x);
    dot.setAttribute('cy', pt.y);
    dot.setAttribute('class', 'dot ' + cls);
    dot.style.opacity = 1;
    return true;
  }

  // ── the arrow in the step rail, from 3 back to 1 ──────────────────────
  // Hidden until the loop closes; then it draws itself and a dot runs it.
  const backPath = el('path', { class: 'back-arrow' }, backSvg);
  const backLabel = el('text', { class: 'back-l', 'text-anchor': 'middle' }, backSvg, 'next candidate');
  const backDot = el('circle', { r: 4, class: 'back-dot' }, backSvg);
  let backLen = 0;
  function layoutBack() {
    const wr = railWrap.getBoundingClientRect();
    const b1 = railBtns[1].getBoundingClientRect(), b3 = railBtns[3].getBoundingClientRect();
    if (Math.abs(b1.top - b3.top) > 4 || wr.width === 0) { backSvg.style.display = 'none'; return; }
    backSvg.style.display = '';
    const x1 = b1.left + b1.width / 2 - wr.left, x3 = b3.left + b3.width / 2 - wr.left;
    const y = b1.top - wr.top - 3;
    backSvg.setAttribute('viewBox', `0 0 ${wr.width} ${y + 4}`);
    backSvg.setAttribute('width', wr.width);
    backSvg.setAttribute('height', y + 4);
    backPath.setAttribute('d', `M${x3},${y} C${x3},6 ${x1},6 ${x1},${y}`);
    backLen = backPath.getTotalLength();
    backPath.style.strokeDasharray = backLen;
    backLabel.setAttribute('x', (x1 + x3) / 2);
    backLabel.setAttribute('y', y / 4 + 9.5);
  }
  function renderBack(q) {
    if (q <= 0 || backSvg.style.display === 'none') {
      backPath.style.opacity = 0; backLabel.style.opacity = 0; backDot.style.opacity = 0;
      return;
    }
    const draw = easeOut(seg(q, 0, 0.55));
    backPath.style.opacity = 1;
    backPath.style.strokeDashoffset = backLen * (1 - draw);
    backPath.setAttribute('marker-end', draw >= 0.98 ? 'url(#arr-ink)' : 'none');
    backLabel.style.opacity = seg(q, 0.4, 0.6);
    if (q >= 0.6 && q < 1) {
      const pt = backPath.getPointAtLength(backLen * ease(seg(q, 0.6, 1)));
      backDot.setAttribute('cx', pt.x);
      backDot.setAttribute('cy', pt.y);
      backDot.style.opacity = 1;
    } else {
      backDot.style.opacity = 0;
    }
  }

  function placeCursor(node, tms) {
    if (!node) { cursor.style.opacity = 0; return; }
    cursor.setAttribute('x', +node.getAttribute('x') + node.getComputedTextLength() + 2);
    cursor.setAttribute('y', +node.getAttribute('y') - 10);
    cursor.style.opacity = Math.floor(tms / 450) % 2 ? 0.25 : 1;
  }

  let curStep = -1;

  function render(t) {
    let s = 0;
    while (s < 3 && t >= START[s + 1]) s++;
    const p = (t - START[s]) / DUR[s];

    if (s !== curStep) {
      curStep = s;
      stage.dataset.step = s;
      captionEl.textContent = CAPTIONS[s];
      railBtns.forEach(b => b.setAttribute('aria-selected', String(+b.dataset.step === s)));
    }

    const typing = renderLog(t, t);
    renderGraph(s, p);

    let dotShown = false, arrow = null, busy = false, dim = true, steerQ = 0, backQ = 0;
    let sigma = '—', sub = 'nothing tried yet', toolQ = 0, running = false;

    if (s === 0) {
      drawPack(0, 0, 0);
    } else if (s === 1) {
      drawPack(0, 0, 0);
      arrow = p >= 0.40 && p < 0.72 ? 'q' : p >= 0.72 && p < 0.86 ? 'qr' : null;
      dotShown = moveDot('q', 0.40, 0.48, p, 'd-agent') || moveDot('qr', 0.72, 0.80, p, 'd-server');
    } else if (s === 2) {
      dim = false;
      toolQ = seg(p, 0.20, 0.30);
      running = p >= 0.42 && p < 0.74;
      const a = 0.42, b = 0.56, c = 0.74;
      if (p < a) { drawPack(0, 0, 0); sigma = '1.86'; sub = 'start · idea C'; }
      else if (p < b) { const k = ease(seg(p, a, b)); drawPack(0, 1, k); sigma = lerp(1.86, 1.91, k).toFixed(2); sub = 'step 100'; }
      else if (p < c) { const k = ease(seg(p, b, c)); drawPack(1, 2, k); sigma = lerp(1.91, 1.97, k).toFixed(2); sub = 'step 200'; }
      else { drawPack(2, 2, 0); sigma = '1.97'; sub = 'collides after 200'; }
    } else {
      dim = false;
      toolQ = 1;
      drawPack(2, 2, 0);
      sigma = p < 0.82 ? '1.97' : '1.970';
      sub = p < 0.82 ? 'predicted' : 'scored · new best';
      arrow = p >= 0.07 && p < 0.30 ? 's'
            : p >= 0.52 && p < 0.72 ? 'rec'
            : p >= 0.74 && p < 0.88 ? 'sr' : null;
      dotShown = moveDot('s', 0.07, 0.15, p, 'd-agent')
              || moveDot('rec', 0.52, 0.58, p, 'd-server')
              || moveDot('sr', 0.74, 0.82, p, 'd-server');
      busy = p >= 0.15 && p < 0.30;
      steerQ = seg(p, 0.30, 0.52);
      backQ = seg(p, 0.86, 0.995);
    }

    const steerTyping = typeSeq(steerNodes, STEER, steerQ);
    placeCursor(typing || steerTyping, t);
    renderTree(toolQ, running, t);
    renderBack(backQ);

    if (!dotShown) dot.style.opacity = 0;
    arrowOn(arrow);
    gEval.classList.toggle('busy', busy);
    evalStatus.textContent = busy ? 'scoring' + '.'.repeat(1 + Math.floor(t / 350) % 3) : '';
    gPack.classList.toggle('dim', dim);
    sigmaEl.textContent = sigma;
    sigmaSub.textContent = sub;
  }

  // ── clock and controls ────────────────────────────────────────────────
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let t = 0, playing = false, userPaused = reduceMotion, last = null, raf = 0, inView = true;

  function tick(now) {
    if (last == null) last = now;
    t += Math.min(now - last, 100);
    last = now;
    // After the first pass the loop closes on step 1, as the arrow says.
    if (t >= TOTAL) t = START[1] + (t - TOTAL);
    render(t);
    raf = requestAnimationFrame(tick);
  }
  function play() {
    if (playing) return;
    playing = true; last = null;
    raf = requestAnimationFrame(tick);
    playBtn.textContent = 'Pause';
    playBtn.setAttribute('aria-label', 'Pause');
  }
  function pause() {
    playing = false;
    cancelAnimationFrame(raf);
    playBtn.textContent = 'Play';
    playBtn.setAttribute('aria-label', 'Play');
  }
  function jump(step) {
    t = START[(step + 4) % 4];
    last = null;
    render(t);
  }

  playBtn.addEventListener('click', () => {
    userPaused = playing;
    if (playing) pause(); else play();
  });
  prevBtn.addEventListener('click', () => jump(curStep - 1));
  nextBtn.addEventListener('click', () => jump(curStep + 1));
  railBtns.forEach(b => b.addEventListener('click', () => jump(+b.dataset.step)));

  // Save work when the scene is scrolled away or the tab is hidden.
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting;
      if (!inView) pause(); else if (!userPaused) play();
    }, { threshold: 0.15 }).observe(root);
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pause(); else if (inView && !userPaused) play();
  });

  layoutBack();
  window.addEventListener('resize', layoutBack);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(layoutBack);

  if (reduceMotion) {
    // Rest on the richest frame; the controls still step through.
    t = START[3] + DUR[3] * 0.87;
    render(t);
    pause();
  } else {
    render(0);
    play();
  }
})();
