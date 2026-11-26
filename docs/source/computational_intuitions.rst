Computational intuitions
========================

This page explains why clone2vec is built the way it is, using a toy system where the right answer is known: **rings
and crosses**. Each design choice is introduced as a fix for a concrete failure that you can reproduce in the
interactive walkthrough at the bottom of the page. Everything runs in your browser on simulated data; nothing is sent
to a server.

For the biological side of the same ideas, see :doc:`biological_intuitions`. For the code, see :doc:`tutorial`.

.. image:: _static/images/pipeline.png
   :alt: clone2vec pipeline
   :width: 800
   :align: center

The toy system
--------------

* **Expression space** is a 2D disc: a stand-in for a UMAP of real cells, with 10,000 unbarcoded cells in the
  background.
* **Every clone** is a set of barcoded cells drawn from one of ten hidden **shapes**: a core, three rings of
  increasing radius, and six crosses rotated in 15° steps.
* **The task** is to place clones of the same shape together without ever being told the shapes.

The shapes are chosen to be hard in a realistic way. Rings differ by *where* their cells are. Crosses all pass
through the centre and differ only by angle, so clones of different crosses share the central cells, while two clones
of the same cross may not have a single cell in the same place. The crosses also form a continuum that wraps around:
75° is next to 0°, just as real differentiation paths share early states and diverge later.

Six ready-made examples range from easy to hard by changing the number and size of clones:

* **Toy:** 10 clones, enough to follow every number by hand.
* **Very easy / Easy:** 200 clones of 100 or 30 cells.
* **Hard / Very hard:** realistic, heavy-tailed clone sizes, with many clones of only a few cells.
* **Extreme:** half the clones have 2–4 cells, the other half 60–120.

You can also edit the dataset yourself: the number of clones per shape, the clone-size distribution, and the fraction
of cells captured.

Lesson 1: comparing clones directly needs dense sampling
--------------------------------------------------------

The most direct way to compare two clones is to compare their cell distributions, for example with the energy
distance: ``E(A,B) = 2·mean‖a−b‖ − mean‖a−a′‖ − mean‖b−b′‖``. It is zero when both clones come from the same
distribution.

This works when clones are large, because each clone then fills in its shape. With small clones, each distribution is
estimated from a few cells, so the distance mostly measures which cells happened to be sampled. Two small clones of
the same cross look far apart unless their few cells land near each other.

**Try it:** in Part 2, compare the E-distance heatmap and t-SNE for *Very easy* and *Very hard*. With large clones the
shapes separate; with small clones they blur together.

Lesson 2: describe a clone by its neighbours
--------------------------------------------

clone2vec does not compare clone shapes directly. Instead, it asks every barcoded cell which clones its *k* nearest
neighbours belong to, and adds these counts up per clone. The result is a clone × clone matrix: row *i* says how
often the cells of clone *i* sit next to cells of every clone.

This changes what "similar" means. Two clones of the same cross never need to share a position: it is enough that
both sit next to cells of *other* clones of that cross. A three-cell clone gives only three points for its shape, but
3 × *k* neighbour observations, all comparable with those of a large clone. The neighbourhood size *k* is the scale of
the comparison: a small *k* reads fine structure but gives noisy counts, a large *k* averages over wider regions and
blurs neighbouring shapes together.

**Try it:** in Part 3, open step 1 with the *Toy* example and click cells of the matrix: the left panel draws exactly
the neighbour edges behind each number. With *Easy*, the matrix shows blocks along the diagonal, one per shape.

Lesson 3: rows have very different totals
-----------------------------------------

The rows of the neighbour-count matrix are not on the same scale: a clone with 300 cells collects 100 times more
neighbours than a clone with 3 cells. The usual single-cell recipe fixes this by scaling every row to the same total
and taking ``log1p`` before PCA.

That scaling hides how much evidence each row has. After normalization, a row built from 15 neighbours counts as
much as a row built from 1,500. A tiny clone's row is mostly zeros, and mostly-zero rows look alike regardless of
the shape they came from, so PCA pulls the tiny clones of every shape towards the same place.

**Try it:** load *Extreme*, go to step 3 and switch between *PCA* and *GLM-PCA*. With PCA, the tiny clones of all
shapes gather in a central blob; with GLM-PCA, most of them sit with their shape.

Lesson 4: model the counts instead of normalizing them
------------------------------------------------------

clone2vec keeps the raw counts and fits a model of them. Each row is treated as a **multinomial sample**: the
clone's neighbours are split between clones with probabilities ``softmax(C·W + b)``. ``C`` holds one low-dimensional
vector per clone (the clonal embedding), ``W`` describes each neighbour clone, and ``b`` captures how common each clone
is overall. This is a multinomial GLM-PCA, and it is the same objective as the Skip-Gram model of word2vec: predict
the clones of a cell's neighbours from the cell's own clone.

Three consequences follow:

* **Each clone is weighted by its evidence.** A row with 1,500 neighbours contributes 1,500 observations to the
  likelihood, a row with 15 only 15. Small clones are kept, but they can't distort the embedding.
* **Zeros need no special treatment.** A missing neighbour is simply an observed count of zero; there is no
  pseudo-count to choose.
* **Overall abundance is factored out.** Because ``b`` absorbs how common each clone is, ``C`` describes how a clone's
  neighbourhood *deviates* from the average one. Distances in the embedding follow the pointwise mutual information
  between clones: whether two clones meet more or less often than chance would predict.

**Try it:** in step 2, watch the GLM-PCA fit converge and click a clone to compare its observed and reconstructed
neighbour profile.

Lesson 5: length is evidence, direction is behaviour
----------------------------------------------------

A clone with little data gives the model little reason to move it away from the average neighbourhood, so its
embedding vector stays short, close to the origin. Its *direction* still points toward the shape it belongs to. Large
clones have long vectors. The walkthrough therefore compares clones by direction (cosine distance) before running
t-SNE, so that tiny clones are not mistaken for a separate group just because they have little data.

The same geometry makes mixtures easy to read: a clone that splits its cells equally between two behaviours ends up
between the clones that show each behaviour alone.

What the toy does and doesn't show
----------------------------------

* **When clones are large, the methods agree.** With *Very easy* or *Easy*, PCA and GLM-PCA give nearly the same
  picture. The difference shows when clone sizes are small and uneven, as in *Extreme*, which is the typical regime of
  real lineage tracing data.
* **No method recovers information that isn't there.** In *Very hard*, many clones have two or three cells, and all
  approaches lose part of the structure. Describing clones by their neighbours still keeps much more of it than
  comparing their cells directly, but no model can make a two-cell clone informative.
* **The toy is two-dimensional.** Real expression spaces are high-dimensional and the kNN graph is built in PCA or
  a batch-corrected space, but the logic of every lesson above carries over unchanged.

Interactive walkthrough
-----------------------

.. only:: html

   .. raw:: html

      <iframe id="c2v-walkthrough" src="_static/walkthrough/index.html" title="Interactive clone2vec walkthrough"
              style="display:block;width:100%;height:2400px;border:0;overflow:hidden" scrolling="no"></iframe>
      <script>
      (function () {
        // Sizes the iframe to its content and passes the docs theme (light/dark, colors, fonts) into it.
        var frame = document.getElementById("c2v-walkthrough");
        var root = document.documentElement;
        var MAP = {
          "--page": "--pst-color-background",
          "--panel": "--pst-color-on-background",
          "--fg": "--pst-color-text-base",
          "--muted": "--pst-color-text-muted",
          "--line": "--pst-color-border",
          "--empty": "--pst-color-surface",
          "--accent": "--pst-color-primary",
          "--f-sans": "--pst-font-family-base",
          "--f-mono": "--pst-font-family-monospace"
        };
        function sendTheme() {
          if (!frame.contentWindow) return;
          var cs = getComputedStyle(root), vars = {};
          for (var k in MAP) { var v = cs.getPropertyValue(MAP[k]).trim(); if (v) vars[k] = v; }
          var theme = root.dataset.theme ||
            (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
          frame.contentWindow.postMessage({ type: "c2v-walkthrough:theme", theme: theme, vars: vars }, "*");
        }
        window.addEventListener("message", function (e) {
          if (e.source !== frame.contentWindow || !e.data) return;
          if (e.data.type === "c2v-walkthrough:height") frame.style.height = e.data.height + "px";
          if (e.data.type === "c2v-walkthrough:ready") sendTheme();
        });
        frame.addEventListener("load", sendTheme);
        new MutationObserver(sendTheme).observe(root, { attributes: true, attributeFilter: ["data-theme", "data-mode"] });
      })();
      </script>

.. only:: latex

   The interactive walkthrough is available in the online documentation at
   https://clone2vec.readthedocs.io/computational_intuitions.html.

From the toy to your data
-------------------------

Every knob in the walkthrough has a counterpart in the package:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Walkthrough
     - Package
   * - *k* in step 0
     - ``c2v.tl.clonal_nn(adata, clones, k=15)``
   * - Dropping tiny clones
     - ``c2v.pp.clones_adata(adata, min_size=...)``
   * - Number of GLM-PCA factors
     - ``c2v.tl.clone2vec(clones, z_dim=10)``
   * - The neighbour-count matrix
     - ``clones.obsp["gex_adjacency"]``
   * - The clonal embedding ``C``
     - ``clones.obsm["clone2vec"]``

See :doc:`tutorial` for the full workflow.