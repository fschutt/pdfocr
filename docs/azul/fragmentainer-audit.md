# azul / printpdf: can text flow through a chain of non-rectangular frames (InDesign threading)?

Audit of azul `9dc2eb66142e84ffbe4d8e2edaabdd3fd8df5899` (extracted to /tmp/azul-audit) and printpdf
`153335dbfdab91357019fcc73a10828165fe5aa0` (/tmp/printpdf-audit), plus html2pdf
(/Users/fschutt/Development/pdfocr/html2pdf/src). Read-only; nothing was built or run.

Path abbreviations: `T/` = `layout/src/text3/cache/`, `S/` = `layout/src/solver3/`, `FC/` = `layout/src/solver3/fc/`,
`PP/` = `/tmp/printpdf-audit/src/`, `H2P/` = `pdfocr/html2pdf/src/`.
**[V]** = verified by reading the code; **[I]** = inferred from the code, not run.

## Short answer

**No.** azul cannot do this today, and the missing parts are more than wiring.

1. The text engine (text3) can already flow **one paragraph** through an ordered chain of rectangular
   fragments with different widths and heights: `TextShapingCache::layout_flow(.., flow_chain: &[LayoutFragment], ..)`
   (`T/shaping_cache.rs:987-1212`). The line state carries over correctly from one fragment to the next.
   But nothing in the layout engine (solver3) calls it with more than one fragment
   (`FC/ifc.rs:867-870`, which reads only `"main"` at `:926`). Nobody reads the leftover content either
   (`FC/text_constraints.rs:1040-1047`).
2. `flow-into` / `flow-from` are parsed and have getters, and nothing else touches them. There is no
   named-flow code or region-chain code to wire them to.
3. Non-rectangular frames do not work. The `shape-inside` branch of `get_line_constraints` is an empty
   stub (`T/line_layout.rs:2169-2171`), so a box with `shape-inside` lays out **no text at all**.
4. Holes in the middle of a line: the per-line span math exists (even-odd, several spans per line,
   `T/exclusions.rs`). It is used only for exclusions, though, and the line breaker fits words against the
   **sum** of the segment widths. Words then get torn across a hole.
5. Pagination is "lay out one tall column, then slice it". It works for a single column of uniform width,
   which is the path printpdf uses. html2pdf sends every page as its own HTML document, so even that does
   not reach pdfocr.

---

## 1. Capability matrix

| Capability | Parsed? | Laid out? | Painted? | Tested? | Known bugs |
|---|---|---|---|---|---|
| **Rectangular region chain** (CSS Regions `flow-into`/`flow-from`) | Yes [V]: `css/src/props/layout/flow.rs:15-48`, `css/src/props/property.rs:364-365,3955-3956`; getters `core/src/prop_cache.rs:4579-4581` | **No** in solver3 [V]: no caller of `get_flow_into`/`get_flow_from` anywhere (repo-wide grep). The text3 engine exists: `layout_flow` loops `for fragment in flow_chain` (`T/shaping_cache.rs:1159-1185`) with one `BreakCursor`. solver3 passes only `"main"` (`FC/ifc.rs:867-870`) | No | text3 only: `layout/tests/text3/{two,three,four}.rs` `test_flow_across_fragments` (live; two equal-width fragments; checks that frag1 has 1 line and frag2 the rest) | §3 B1, B4-B8 |
| **Non-rectangular frame** (`shape-inside: polygon()/path()/circle()`) | Yes, with gaps [V]: `css/src/shape_parser.rs`; `%` kept as raw px (`:307-313`), only px/unitless; fill-rule dropped (`:154-170`); `path(evenodd, "...")` rejected (`:285`) | **No** [V]: resolved into `ShapeBoundary` (`FC/text_constraints.rs:714-720`, `T/shapes.rs:626-760`), but `get_line_constraints` leaves `available_segments` empty when `shape_boundaries` is non-empty (`T/line_layout.rs:2146-2171`). Every line is skipped until the failsafe (`:505-577`), so **no text is placed** | n/a | None (no test sets `shape_boundaries`/`shape-inside`) | B2 |
| **Holes / cutouts inside a frame** | Only via `path()` with several subpaths (even-odd); `polygon()` is one ring [V] | Partly [V]. Span math: `path_segments_line_intersection` (`T/exclusions.rs:100-156`, even-odd) and `polygon_line_intersection` (`:244-292`, even-odd pairing, although the comment at `:252` says "winding"). Several segments per line exist for **exclusions** (`T/line_layout.rs:2179-2232`). Breaking uses `total_available` (sum, `:1060-1065`), then clusters are spread greedily over the segments (`:1569-1615`) | Yes (exclusions) | Span-level unit tests: `T/autotest_generated.rs:2446-2610`, `T/shape_outside_and_ruby_tests.rs:48-113` (hole via even-odd). Line level: `four.rs:384` only checks the first x < 100 and the last x >= 200 | B3, B9 |
| **Exclusions: floats** | Yes | Yes, as **rectangles** [V]: `FC/bfc.rs:2155-2182` (made relative to the IFC) and `FC/text_constraints.rs:650-680` | Yes | `float_integration.rs`, `float_and_scrollbar.rs` | Not shifted per text3 column (B10). Mid-line holes only from the IFC's own shape-outside or initial-letter |
| **Exclusions: shape-outside / shape-margin / exclusion-margin** | Yes | Wrong [V]. `shape-outside` is taken from the **IFC root's own style** and cuts its own text (`FC/text_constraints.rs:723-727`); a float's shape-outside is never read (only reader: `text_constraints.rs:535-541`). Margins are stored (`T/constraints.rs:319`) and never applied (`ShapeBoundary::inflate` has no caller in layout and is a no-op for polygon/path, `T/shapes.rs:556`) | (rect only) | No line-level test | B11, B12 |
| **Exclusions: CSS Exclusions `wrap-flow`** | **No** (no such property) [V] | No | No | No | — |
| **Multicol** | count/width/gap/fill/span/rule (`property.rs:357-363,197`); no `balance-all` (`css/src/props/layout/column.rs:105-109`) | Yes, two mechanisms [V]. **A**: columns inside one IFC (`FC/text_constraints.rs:934-942`, `T/line_layout.rs:198-781`). **B**: block multicol (`FC/bfc.rs:480-496,2329-2340` → `S/multicol.rs:472-553` → `FC/columns.rs:324-390`). B splits an IFC child at line indices (`ColumnFlow`, `T/constraints.rs:358-370`; `T/line_layout.rs:362-400,465-473,720-724`). `column-span`, `column-rule`: getters only, never used | Text yes; **column rules not painted** | `a_multicol_block_flows_its_children_through_its_columns.rs` (9 tests), `S/multicol.rs:569-763`; `text3/two.rs:172` live, `three.rs:177`/`four.rs:175` `#[ignore]` | B13-B16 |
| **Multicol across pages** | — | **No** [V/I]. The slicer cuts a tall multicol across all columns (reading order broken) [I]. In a token pass block columns are switched off (`FC/columns.rs:114-121`). No mention of columns in `break_token.rs`/`page_breaks.rs`/`paged_layout.rs`/`pagination.rs` | — | No test | `break-before: column/avoid*` forces a **page** break in the token pass (`FC/bfc.rs:745-758`) |
| **Flow across pages** | `break-*` yes, `widows`/`orphans` yes, `box-decoration-break` yes (getter `S/getters.rs:4839` never called), `@page` **no** (`S/paged_layout.rs:10-11`, `PP/lib.rs:461`) | One column of uniform width only [V]. printpdf: `layout_document_paged_v2` (`S/paged_layout.rs:201`) → `_impl` (`:356-759`): one layout at the viewport width, one master display list, cut between lines (`S/page_breaks.rs:194`, widows/orphans `:539-660`, enabled by printpdf at `PP/html/mod.rs:449`). Per-page **heights** work through `PageSequence`; per-page **widths** do not ("Re-wrapping text per fragmentainer is not implemented yet", `S/pagination.rs:847-856,935-960`). `compute_sectioned_pagination` (`S/paged_layout.rs:1238-1262`) re-wraps per width section, but only by whole blocks. Token path (`layout_document_tokenized`, `:2708`, used only by tests): IFCs are never split (`FC/bfc.rs:1684-1690`); `InlineBreakToken` (`S/break_token.rs:95-141`) is never produced; `FragmentainerSpace` has block extents only (`FC/types.rs:151-165`) | Yes; glyphs clipped per page band (`S/display_list.rs:12002`) | `a_block_taller_than_a_page_is_split_across_pages.rs`, `break_token_pages.rs`, `token_vs_slicer_differential.rs`, `streaming_pages.rs` | **pdfocr never reaches this**: html2pdf renders each page alone (`H2P/main.rs:141-193,297-322`), keeps PDF page 1 and appends it (`:183-184`) |
| **InDesign threading** (several frames per page, several pages, any order) | — | **No** [V]: no named flow, no region chain, no per-fragmentainer inline size or shape anywhere in solver3 | No | No | — |

### State carried across fragments in `layout_flow` (text3, verified by reading)

| State | Carried? | Evidence |
|---|---|---|
| Break position | Yes. The same `BreakCursor` (`next_item_index` + `partial_remainder`) is used for every fragment | `T/shaping_cache.rs:1145-1147,1159-1185`; `T/break_opportunities.rs:311-323` |
| Hyphenation across a fragment end | Yes. The remainder of the word goes to `partial_remainder` | `T/line_layout.rs:1070-1086` |
| "Don't hyphenate the last line of a frame" | **Missing** (no `hyphenate-limit-*` anywhere) | grep |
| Justification of a fragment's last line | Correct. Only the paragraph's last line is treated as last (`is_last_line = cursor.is_done() && !was_hyphenated`) | `T/line_layout.rs:681` |
| `text-indent` only on the first line | Correct, greedy path and KP path | `T/line_layout.rs:279-284`, `:157-161` |
| `text-indent: each-line` after a forced break at the end of a fragment | **Lost** (`is_after_forced_break` reset to false) | `T/line_layout.rs:402` |
| Bidi base direction (`unicode-bidi: plaintext`) | **Detected again per fragment** from the remaining text | `T/line_layout.rs:241-262` |
| Bidi run order | Reordered once over the whole content before line breaking (stage 2); L2 per line | `T/shaping_cache.rs:1107`, `T/line_layout.rs:660` |
| Inline box edges (a span split at a fragment boundary) | **Reset**: `before_chain = None` at each fragment/column start, so the span's start edge is drawn again [I] | `T/line_layout.rs:421`, used at `:1655-1666` |
| `line_index` / `line-clamp` | **Restart at 0** in each fragment and each text3 column | `T/line_layout.rs:396-400,449` |
| Writing mode / orientation / hyphens / word-break | Taken from the **first** fragment only (documented) | `T/shaping_cache.rs:1055-1058,1139,1145-1148` |
| Knuth-Plass (`TextWrap::Balance`) | **Not fragment-aware**: takes all remaining content into the first fragment | `T/line_layout.rs:150-182` |

---

## 2. Gaps: what can be reused, what is missing, how to build it

### G1. `shape-inside` frames (real feature, small) — **S**
- **Reuse:** `get_shape_horizontal_spans` (`T/exclusions.rs:162-217`) and the even-odd path code (`:100-156`); `ShapeBoundary::from_css_shape` (`T/shapes.rs:626`). The exclusion subtraction after it (`T/line_layout.rs:2179-2232`) stays as it is.
- **Missing:** the boundary branch of `get_line_constraints` (`T/line_layout.rs:2169-2171`).
- **Sketch:**
  1. For each boundary, compute the spans of the shape that cover the **whole** line band. Intersect the spans at the band's top and bottom edges and at every vertex y inside the band, or clip the polygon to the band.
  2. Intersect those spans with `[0, available_width]`.
  3. Push one `LineSegment` per span, then subtract exclusions as today.
  4. Let `balanced_lines_per_column` (`:286-345`) run with shapes too.
  5. Fix the parser: resolve `%` against the reference box (keep it as a percentage in `css/src/shape.rs` and resolve it in `from_css_shape`); add `pt`; accept `path(evenodd|nonzero, "…")` (`css/src/shape_parser.rs:276-294`); implement nonzero.
- **Files:** `T/line_layout.rs`, `T/exclusions.rs`, `T/shapes.rs`, `css/src/shape_parser.rs`, `css/src/shape.rs`.

### G2. Holes: several segments per line, filled properly (real feature) — **M**
- **Reuse:** `LineConstraints.segments`, `break_one_line` (`T/line_layout.rs:950-1212`), per-segment justification and alignment in `position_one_line_between` (`:1569-1760`), and the `before`/`after` edge chain that already treats a span as continuing across segments (`:1655-1666`).
- **Missing:**
  - Breaking is done per line against the summed width (`:1060-1065`).
  - Spreading over the segments is greedy, cluster by cluster, and forces "at least one item" into any segment, however narrow (`:1597-1610`).
- **Sketch:**
  1. Break **per segment**: for each segment in inline order, call `break_one_line` with a `LineConstraints` holding only that segment, so words and hyphenation never cross a hole.
  2. Skip segments narrower than a threshold (for example the next unit's width, or 1-2 em).
  3. Pass `Vec<Vec<ShapedItem>>` (items per segment) to the positioner instead of distributing there.
  4. Sample shapes over the band (G1) for exclusions too: take the union of spans over the band, not the centre line (B9).
  5. Knuth-Plass: use the greedy path when shapes or exclusions are present.

### G3. Exclusions — mixed
- **Wiring:**
  - Apply `exclusion_margin` in `get_line_constraints` by calling `inflate` (rect and circle only). **S**
  - Offset exclusions by `-column_start_x` and by the column's y origin in text3 columns. **S** (B10)
- **Feature:**
  - Polygon and path offsetting for `shape-margin`. **M**
  - Make shape-outside follow CSS Shapes 1: keep the float node's `shape-outside` in `FloatingContext` (`FC/floats.rs`, `FC/bfc.rs:2155-2182`) and emit that `ShapeBoundary` in `FC/text_constraints.rs:650-680`; drop the IFC-root push at `:723-727`. **M**
  - `wrap-flow` on positioned elements: a new property plus collecting abs-pos boxes as BFC exclusions. **L** (pdfocr does not need it if cutouts are holes in the frame shape.)

### G4. Named flows and the region chain (CSS Regions) — real feature, not wiring — **L (2-3 weeks)**
- **Reuse:**
  - `layout_flow` / `perform_fragment_layout` (`T/shaping_cache.rs:987`, `T/line_layout.rs:127`) with a `BreakCursor` that keeps going.
  - `translate_to_text3_constraints` (`FC/text_constraints.rs:637`) to get per-paragraph `UnifiedConstraints`.
  - The `ColumnFlow` trick: one `UnifiedLayout` per IFC, with later lines moved to another box (`T/constraints.rs:358-370`, `T/line_layout.rs:388-400,720-724`). This is how a split paragraph can be painted by the existing display-list code and the per-page glyph clipping (`S/display_list.rs:12002`) without changing them.
  - `fragment_fit` (`S/break_token.rs:176`) for whole blocks.
  - `InlineBreakToken::{from_cursor,resume}` (`S/break_token.rs:107-141`).
- **Missing:**
  - Taking `flow-into` content out of normal flow.
  - Collecting the `flow-from` region chain.
  - A pass that flows a **sequence of blocks** (many paragraphs) through regions. `layout_flow` takes the `InlineContent` of a single IFC (`T/inline_items.rs:49-74`), with one set of constraints per fragment, not per paragraph.
  - Storing results per region.
  - Reporting overflow.
- **Sketch:**
  1. `S/layout_tree.rs`, `generate_layout_tree`: leave elements with `flow-into: X` (and their subtrees) out of their parent's box tree, and build one anonymous BFC root per flow name holding them in document order. Region boxes (`flow-from: X`) lay out as normal boxes; their own DOM children are ignored (CSS Regions). **M**
  2. New `S/regions.rs`, `layout_named_flows(ctx, tree, positions)`, run after the main pass has absolute positions for every region box:
     - regions = `flow-from` boxes in document order (optionally an `-azul-region-order` key);
     - for each block child of the flow root:
       - **IFC:** build constraints from the region's content width, `shape_boundaries` = the region's `shape-inside` shifted by `-pen_y`, exclusions local to the region, `available_height = region_h - pen_y`. Then call a new resumable text3 entry point, e.g. `TextShapingCache::layout_flow_resume(&mut FlowState, &LayoutFragment)`, that keeps the shaped items and `BreakCursor` between calls. Go to the next region when the cursor is not done.
       - **Block child** (image, table): `fragment_fit` → place it here or push it to the next region.
     - Margins collapse inside a region and are truncated at a region start (css-break-3 §5.2).
     - Move the items into the IFC node's coordinate space (region origin minus node origin) and store **one** `CachedInlineLayout`, as `ColumnFlow` does.
  3. Inline caches are keyed by width and content (`FC/ifc.rs:692-707`, `S/layout_tree.rs:270-290`). For IFCs split across regions, include the region geometry in the key or skip the cache. **M**
  4. Overflow: put the first unplaced node and byte into the paged result and warn. **S**
  5. text3 fixes B1, B4-B8. **S**

### G5. One document spans several pages of different sizes (printpdf / html2pdf) — API work — **M**
Pages are explicit boxes, so automatic pagination is not needed. Each `.page` box is exactly one PDF page.
- **printpdf:**
  1. Per-page sizes in `GeneratePdfOptions` (`PP/lib.rs:109`) and `XmlRenderOptions` (`PP/html/mod.rs:119`).
  2. Fill `FakePageConfig.page_sequence` (`PageSequence` overrides, heights already work) in `build_page_config` (`PP/html/mod.rs:428-462`).
  3. Viewport width = the widest page (`:577-586`).
  4. Call `PdfPage::new` with each page's own size (`:730`), and use that size for the y-flip (`PP/html/bridge.rs:383`).
  5. Stop dropping "empty" pages, or return an index map (`PP/html/mod.rs:662-676`).
- **html2pdf:**
  1. Write one HTML with N `.page` boxes and make one `from_html_with_cache` call (replaces `H2P/main.rs:139-193`).
  2. `flow::page_html` (`H2P/flow.rs:126-146`) should emit only a page fragment; frames in pt rather than % (`:405`).
  3. Drop the line estimator (`H2P/flow.rs:356-400`) and the `FLOW_SCALES` retries (`H2P/main.rs:24,146-173`).
  4. Make the layout report cover every page (`H2P/measure.rs:138`).
- **A cheaper but worse option (M):** keep one HTML per page and continue the story across documents. azul can already name the text node and byte at a y position (`spine_line_start_at_y`, `S/paged_layout.rs:2410`), and can cut a DOM there (`document_edit::split_dom_at_path`, as used by `compute_sectioned_pagination`). html2pdf would cut the story at the first line that did not fit and start the next page with the tail. Costs: the story is restyled for every page, and a hyphenated word must be split again at the cut.

### G6. Multicol across pages / column fragmentation — **L** (not needed for pdfocr)
Columns would have to become fragmentainers in the token path. Today block columns are switched off there (`FC/columns.rs:114`), and nested blocks never split (`FC/columns.rs:275-288`).

### G7. IFC split in the token path (`InlineBreakToken`) — **L** (only for automatic pagination of flowing text that re-wraps)
Produce `InlineBreakToken::from_cursor` in `layout_ifc` when the lines exceed `FragmentainerSpace.remaining_block_extent`. Resume with `InlineBreakToken::resume`. Add `inline_size` and `shape` to `FragmentainerSpace` (`FC/types.rs:151`) so each page can re-wrap. Store per-page inline layouts.

---

## 3. Bugs and suspicious code

| # | Where | What | Status |
|---|---|---|---|
| B1 | `T/line_layout.rs:436-446` | Fit check tests the line's **top** against `available_height` (`line_top_y >= max_height`). The last line of a fragment or column can hang below it by almost a whole line | [V] |
| B2 | `T/line_layout.rs:2146-2171` | `shape-inside` branch is a stub (`// ... complex boundary logic ...`): zero segments, so no text at all | [V] |
| B3 | `T/line_layout.rs:1060-1065`, `:1569-1615` | Lines are broken against the summed width of all segments, then clusters (graphemes, `T/shaped_items.rs:373-393`) are spread greedily over the segments. Words get torn across a hole with no hyphen, and the last segment overflows | [V] logic, [I] visible result |
| B4 | `T/line_layout.rs:150-182`, `T/knuth_plass.rs:697-706` | Knuth-Plass path calls `cursor.drain_remaining()`: it ignores `available_height`, later fragments, columns, exclusions and shapes (a single full-width segment). solver3 never selects it (`FC/text_constraints.rs:946-952`), so only direct text3 users hit this | [V] |
| B5 | `T/line_layout.rs:241-262` | `unicode-bidi: plaintext` base direction is detected again per fragment from the remaining text, not once per paragraph | [V] |
| B6 | `T/line_layout.rs:421` | `before_chain` is reset per fragment and per column, so an inline box split there gets its start edge (padding, border) drawn again | [I] |
| B7 | `T/line_layout.rs:396-402,449` | `line_index`, `line-clamp` and the forced-break state for `text-indent: each-line` restart per fragment and per text3 column | [V] |
| B8 | `T/shaping_cache.rs:1182` | `fragment_layouts` is a `HashMap` keyed by fragment id; a repeated id silently overwrites the earlier fragment | [V] |
| B9 | `T/exclusions.rs:105,179,189,249` vs `:163-176` | Polygon, path, circle and ellipse spans are sampled only at the line's **centre**; rectangles use the whole band. Text overlaps slanted or round exclusions near their top and bottom | [V] |
| B10 | `T/line_layout.rs:476-494,723` | text3 columns: exclusions stay in IFC coordinates, but each column computes its lines from x=0 and y=0. A float masks every column's left edge, and one inside column 2 masks nothing | [V] logic |
| B11 | `FC/text_constraints.rs:723-727`, `:650-680` | `shape-outside` acts on the IFC root's **own** text; a float's `shape-outside` is ignored (wrap is always the float's rectangle) | [V] |
| B12 | `T/constraints.rs:319`, `T/shapes.rs:541-560` | `exclusion-margin` and `shape-margin` are parsed and stored, never applied; `inflate` is a no-op for polygon and path | [V] |
| B13 | `T/line_layout.rs:436-446` + `FC/text_constraints.rs:1040-1047` | text3 columns with a definite height: lines past the last column are **silently dropped** (no overflow columns) | [V] logic, [I] visible result |
| B14 | `T/line_layout.rs:340` vs `:728-731` | Column-balancing probe assumes every line has the same height; real lines advance by `band_height` | [V] |
| B15 | `FC/columns.rs:191-195` | Any float in a block multicol switches off line splitting for **every** paragraph | [V] |
| B16 | `FC/bfc.rs:745-758` | Token pass: any `break-before` other than `auto` (including `column`, `avoid`, `avoid-column`) forces a page break | [V] |
| B17 | `css/src/shape_parser.rs:307-313`, `:154-170`, `:285` | Shape `%` read as px; only px/unitless; fill-rule ignored; `path(evenodd, "...")` rejected | [V] |
| B18 | `FC/text_constraints.rs:988-1003` | `initial-letter` exclusion always at x=0 (wrong for RTL) | [V] |
| B19 | `PP/html/mod.rs:662-676` | printpdf drops pages that hold only background fills, so output page numbers can drift from input page numbers | [V] (sub-audit) |
| B20 | `T/exclusions.rs:252` | Comment says "winding number", but the code pairs sorted crossings (even-odd) | [V] cosmetic |

### Minimal HTML repros (for html2pdf; render with the default layout, not `--layout flow`)

Each repro is a zip of `page_001.html` plus `metadata.json`:
```json
{"engine":"x","page_count":1,"pages":[{"page_num":0,"html":"page_001.html","width_pt":600,"height_pt":450}]}
```
Outcomes are predicted from the code, not observed (no binaries built).

**R1. `shape-inside` drops all text (B2)**
```html
<!DOCTYPE html><html><head><style>
body{margin:0}
.f{position:absolute;left:40px;top:40px;width:300px;height:240px;font:16px/20px serif;
   shape-inside:polygon(0px 0px, 300px 0px, 150px 240px)}
</style></head><body><div class="f">Lorem ipsum dolor sit amet, consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</div></body></html>
```
- **Expected:** text in a downward triangle.
- **Predicted:** an empty page. Every line has zero segments; the loop gives up once y passes 340 with more than 50 empty lines (`T/line_layout.rs:535-577`).

**R2. Words torn across a hole in the middle of the line (B3)**
```html
<!DOCTYPE html><html><head><style>
body{margin:0}
p{position:absolute;left:20px;top:20px;width:400px;height:100px;margin:0;
  font:16px/20px "Courier New",monospace; shape-outside:inset(0px 150px 40px 150px)}
</style></head><body><p>aaaaaaaaaa bbbbbbbbbb cccccccccc dddddddddd eeeeeeeeee ffffffffff
gggggggggg hhhhhhhhhh iiiiiiiiii jjjjjjjjjj</p></body></html>
```
- The hole is x 150..250, y 0..60.
- **Expected:** whole words on each side of the hole.
- **Predicted** (glyph width about 9.6px): line 0 is broken against 300px total, so it takes 2 words plus part of the third. It is then spread as `aaaaaaaaaa bbbb` | `bbbbbb …`: "bbbbbbbbbb" is torn in two with no hyphen.

**R3. Exclusion polygon sampled at the line centre (B9) and float shape ignored (B11)**
```html
<!DOCTYPE html><html><head><style>
body{margin:0;font:16px/20px serif}
.a,.b{position:absolute;left:20px;width:400px;height:80px;margin:0}
.a{top:20px; shape-outside:polygon(100px 10px,300px 10px,300px 25px,100px 25px)}
.b{top:140px;shape-outside:inset(10px 100px 55px 100px)}  /* same rectangle */
.fl{float:left;width:120px;height:120px;shape-outside:circle(60px at 60px 60px);background:#ddd;border-radius:60px}
.c{position:absolute;left:20px;top:260px;width:400px}
</style></head><body>
<p class="a">Text text text text text text text text text text text text text text text text text text.</p>
<p class="b">Text text text text text text text text text text text text text text text text text text.</p>
<div class="c"><div class="fl"></div>Text text text text text text text text text text text text text
text text text text text text text text text text text text text text text text text text text.</div>
</body></html>
```
- **Expected:** `.a` and `.b` look the same (lines 0 and 1 both avoid x 100..300), and `.c` wraps along the circle.
- **Predicted:**
  - `.a` runs line 1 (y 20..40) through the rectangle, because its centre (30) misses y 10..25.
  - `.b` is correct.
  - `.c` wraps along the 120×120 square.

**R4. Lines lost in a fixed-height text multicol (B13) and hanging below it (B1)**
```html
<!DOCTYPE html><html><head><style>
body{margin:0}
.m{position:absolute;left:20px;top:20px;width:400px;height:50px;column-count:2;column-gap:20px;
   font:16px/20px serif;outline:1px solid red}
</style></head><body><div class="m">One two three four five six seven eight nine ten eleven twelve
thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three
twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine thirty.</div></body></html>
```
- **Expected (CSS):** overflow columns to the right, or visible overflow.
- **Predicted:**
  - 3 lines per column, the third ending at y=60, 10px below the red box;
  - every later line missing from the PDF.

**R5. `exclusion-margin` / `shape-margin` have no effect (B12)**

Take R3's `.b` and add `exclusion-margin:20px; shape-margin:20px`. The output is identical to plain `.b`.

**Rust-only (text3) repros for B1 and B4** (in the style of `layout/tests/text3/four.rs:270`):
```rust
let frag = |id: &str| LayoutFragment { id: id.into(), constraints: UnifiedConstraints {
    available_width: AvailableSpace::Definite(100.0), available_height: Some(30.0),
    line_height: LineHeight::Px(20.0), ..Default::default() } };
let r = layout_flow_compat(&mut cache, &long_text, &[], &[frag("a"), frag("b")], &mgr).unwrap();
// B1: expected "a" = 1 line (a 2nd would end at 40 > 30); predicted 2 lines, the 2nd below the frame.
// B4: the same with `text_wrap: TextWrap::Balance` -> all lines in "a", no "b" (cursor drained).
```

---

## 4. Recommended API for pdfocr

**Surface: CSS Regions plus CSS Shapes, in ONE HTML document holding the pages of a run.**

```html
<style>
 .page  { position:relative; width:600pt; height:450pt; overflow:hidden; break-after:page }
 .frame { position:absolute; flow-from: art7 }
 .story { flow-into: art7 }
 .story p { margin:0; text-indent:1em; text-align:justify; hyphens:auto }
</style>
<div class="story" lang="en"><p>…</p><p>…</p></div>          <!-- one flow per article -->
<div class="page">                                            <!-- page 1 -->
  <div class="frame" style="left:54pt;top:60pt;width:230pt;height:330pt;
       shape-inside:path(evenodd,'M0 0H230V330H0Z M0 0H44V40H0Z')"></div>  <!-- hole: drop capital -->
  <img src="cap.png" style="position:absolute;left:54pt;top:60pt;width:44pt">
  <div class="frame" style="left:316pt;top:60pt;width:230pt;height:330pt"></div>
</div>
<div class="page">…frames go on with art7…</div>
```

**Semantics:**
- The order of `flow-from` boxes in the document is the thread order (CSS Regions). That fits book reading order.
- An azul extension `-azul-region-order:<int>` is only needed for threads that run against document order.
- A frame is the region's content box, cut by an optional `shape-inside`: `polygon()`, or `path()` with even-odd holes for margin notes, figures and drop capitals.
- For pdfocr, holes in the frame shape are simpler and more robust than floats or `wrap-flow`, because the geometry comes straight from the scan. Floats inside the story still work as exclusions.
- Content left after the last region is **reported**: the first unplaced node and byte go into the paged result, and the page is flagged. This follows the project rule against hiding engine faults.

**Is CSS Regions the right surface on top of FlowLayout?** Yes. It maps one to one: named flow = content, `flow-from` boxes in order = `flow_chain`, region shape = `LayoutFragment.constraints.shape_boundaries`. `FlowLayout` alone is not enough, though, because it is one IFC: one paragraph, constraints per fragment rather than per paragraph, inline content only. An article needs the block-level driver of G4, which carries a `BreakCursor` per paragraph across regions and collapses or truncates margins at region starts.

**Build order (smallest useful steps):**
1. G1 + B1 + B9: frames with shapes and holes, centre-sampling fix, lines kept inside the fragment. **S-M**
2. G2: segment-aware breaking. **M**
3. G4: named flow and region chain in solver3 (this is the bulk). **L**
4. G5: per-page sizes in printpdf; single-document html2pdf. **M**

**What must change so one flow spans several pages:**
- **azul:** nothing in the pagination machinery. With explicit `.page` boxes of exact height and `break-after: page`, the existing slicer path (`layout_document_paged_v2`) cuts at page-box edges, and `PageSequence` already supports per-page heights. The work is the region pass (G4) running on the single tall canvas before the display list is built.
- **printpdf:** per-page geometry (`GeneratePdfOptions`/`XmlRenderOptions` → `PageSequence` overrides → `PdfPage::new` per page), viewport = widest page, keep blank pages (G5).
- **html2pdf:** one document per run of pages instead of `render_page` per page and `append_document`; remove `flow.rs`'s line estimator and the size-retry loop.
