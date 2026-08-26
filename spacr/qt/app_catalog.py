"""What every self-registering app IS, without importing what it DOES.

A registry row is a key, a name, a sentence, a section and a factory. None of
that needs the screen's code — but the only way to read a constant out of a
module is to execute it, so asking every screen for its own name imported
every screen, and pandas, scipy, sklearn and numpy behind them. The launch
walked more than a thousand modules before the window drew a pixel, to learn
strings that were already written down.

So they are written down HERE, once, and the screen module reads them back:
:data:`DECLARED_APPS` is the definition site for the row, and
``spacr.qt.screens.trellis.APP_NAME`` is a lookup into it rather than a second
copy. There is no drift to police because there is nothing to drift from.

The factory is the one field that really is code, and it stays code: the row
names the callable rather than holding it, and :class:`LazyScreenFactory`
imports the module the first time somebody actually builds the screen. A user
who never opens Dose–Response never pays for scipy.

Nothing here imports Qt, numpy or pandas, and nothing here may start to: this
module is read while the splash screen is up, and its whole purpose is to be
the cheap half of a registration.

See :func:`spacr.qt.app.register_app` for what each field does once it is
handed over, and :data:`spacr.qt.SELF_REGISTERING_MODULES` for the ordering
constraint that decides *when* these rows are registered.
"""
from __future__ import annotations

import importlib

__all__ = [
    "DeclaredApp",
    "LazyScreenFactory",
    "DECLARED_APPS",
    "declared_for",
    "declared_app",
    "register_declared",
]

#: Fields a row may carry beyond the four positional ones, in the order
#: :func:`spacr.qt.app.register_app` declares them. Used to build the keyword
#: arguments for that call, so a field added there is added in one place here.
_OPTIONAL_FIELDS = (
    "stage", "title", "intro", "cli_note", "api_module", "entry",
    "defaults_module", "translations",
)


class DeclaredApp:
    """One registry row, declared as data.

    Deliberately a hand-written class rather than a dataclass or a
    NamedTuple: both of those import modules this file exists to avoid
    importing, and the saving is measured in the module count the splash
    screen is judged by.

    :param module: dotted name of the module that owns the screen. The
        identity of the row — :func:`declared_for` is keyed on it, and it is
        what :class:`LazyScreenFactory` imports.
    :param key: the app key, unique across the registry.
    :param name: display name; the sidebar row, the tile and the menu entry.
    :param desc: one-line summary; the tooltip and status tip.
    :param section: one of :data:`spacr.qt.app.SECTION_ORDER`, spelled out
        rather than referenced. Importing ``app`` from here would be a cycle:
        ``app`` reads this table while it is itself being imported. A
        misspelling is not a silent one — :func:`spacr.qt.app.register_app`
        raises on a section it does not know.
    :param factory: ATTRIBUTE NAME of the screen factory on ``module``, not
        the callable. ``None`` for an app that takes the generic settings
        screen.
    :param stage: one of :data:`spacr.qt.app.STAGES`; likewise spelled out.
    :param title: header at the top of the app's own screen, when it wants
        the longer form. Defaults to ``name``.
    :param intro: the paragraph beside that header. Defaults to ``desc``.
    :param cli_note: for a GUI-only app, the sentence ``spacr-run <key>``
        prints instead of "unknown module".
    :param api_module: module path under the generated API docs, for the
        info link beside the settings.
    :param entry: ``"module:function"`` the Run button runs.
    :param defaults_module: the module whose import registers this key's
        settings defaults.
    :param translations: the display name in the nine non-English UI
        languages, in :data:`spacr.qt.i18n.LANGUAGES` order.
    """

    __slots__ = ("module", "key", "name", "desc", "section", "factory",
                 "stage", "title", "intro", "cli_note", "api_module",
                 "entry", "defaults_module", "translations")

    def __init__(self, *, module: str, key: str, name: str, desc: str,
                 section: str, factory=None, stage=None, title: str = "",
                 intro: str = "", cli_note: str = "", api_module: str = "",
                 entry: str = "", defaults_module: str = "",
                 translations=()):
        self.module = module
        self.key = key
        self.name = name
        self.desc = desc
        self.section = section
        self.factory = factory
        self.stage = stage
        self.title = title
        self.intro = intro
        self.cli_note = cli_note
        self.api_module = api_module
        self.entry = entry
        self.defaults_module = defaults_module
        self.translations = tuple(translations)

    def __repr__(self) -> str:
        return f"DeclaredApp({self.key!r} from {self.module!r})"

    def register_kwargs(self) -> dict:
        """The keyword arguments :func:`spacr.qt.app.register_app` wants.

        Empty fields are dropped rather than passed as ``""``: ``register_app``
        treats a falsy optional as "not given" and falls back — ``title`` to
        ``name``, ``intro`` to ``desc`` — and passing the empty string would
        get the same answer by a longer route. The ``factory`` is a
        :class:`LazyScreenFactory`, so building these costs no import at all.
        """
        kwargs = {}
        if self.factory:
            kwargs["factory"] = LazyScreenFactory(self.module, self.factory)
        for field in _OPTIONAL_FIELDS:
            value = getattr(self, field)
            if value:
                kwargs[field] = value
        return kwargs


class LazyScreenFactory:
    """A registered screen factory that has not imported its screen yet.

    Registration hands one of these to :func:`spacr.qt.app.register_app` in
    place of the real callable. It imports the owning module and looks the
    callable up the first time anything asks for it, which is when a user
    opens the app — so a launch pays for the screens it draws and not for the
    ones it merely lists.

    Two things make it safe to stand in for the real function.

    :func:`spacr.qt.app.registered_factory` RESOLVES one before returning it,
    so the object that reaches :func:`spacr.qt.app._call_screen_factory` — and
    the object a test compares with ``is`` — is the module's own callable. The
    signature inspection that decides whether to pass ``app_key`` and ``host``
    therefore reads the real signature, not this class's ``**kwargs``.

    And calling one directly still works, for the caller that reaches into
    :data:`spacr.qt.app.APP_FACTORIES` itself: :meth:`__call__` applies the
    same "take what you need" rule rather than forwarding arguments the real
    factory never declared.
    """

    __slots__ = ("module", "attribute", "_resolved")

    def __init__(self, module: str, attribute: str):
        self.module = module
        self.attribute = attribute
        self._resolved = None

    def __repr__(self) -> str:
        state = "resolved" if self._resolved is not None else "not imported"
        return f"<lazy screen factory {self.module}:{self.attribute} ({state})>"

    def resolve(self):
        """Import the module and return the real factory. Cached.

        :raises ImportError: if the module cannot be imported.
        :raises AttributeError: if it has no such attribute — a declared row
            naming a factory that does not exist, which the catalog test
            catches long before a user clicks the tile.
        """
        if self._resolved is None:
            self._resolved = getattr(
                importlib.import_module(self.module), self.attribute)
        return self._resolved

    def __call__(self, **kwargs):
        # `inspect` is imported here rather than at the top of the file: it
        # costs a dozen modules of its own, and this module is read while the
        # splash screen is up to avoid exactly that kind of bill.
        import inspect

        factory = self.resolve()
        try:
            params = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            params = {}
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD
                        for p in params.values())
        if takes_any:
            return factory(**kwargs)
        return factory(**{name: value for name, value in kwargs.items()
                          if name in params})


def declared_for(module: str):
    """The row declared for ``module``, or ``None`` if it declares none."""
    return _BY_MODULE.get(module)


def declared_app(key: str) -> DeclaredApp:
    """The row declared for app ``key``.

    :raises KeyError: if no declared app has that key.
    """
    return _BY_KEY[key]


def register_declared(module: str, *, key=None, section=None, stage=None):
    """Register the app declared for ``module`` without importing it.

    The whole point of the table: :func:`spacr.qt.app.register_app` is called
    with strings that were already written down, and the screen's own module
    stays unimported until somebody opens it.

    Idempotent, because it has to be. The same registration is reached from
    three directions — ``app.py``'s own table, the launch walk over
    :data:`spacr.qt.SELF_REGISTERING_MODULES`, and a module's ``register()``
    called directly by a test — and a duplicate key raises.

    :param module: the dotted module name the row is declared under.
    :param key: register under this key instead of the declared one. Two of
        these registrars expose it so a second copy of a screen can be given
        its own row; the row's own key is the default and the normal case.
    :param section: override the declared section. For
        :func:`spacr.qt.layer_viewer.register_layer_viewer_app`, which lets a
        caller place the app elsewhere.
    :param stage: override the declared maturity stage, likewise.
    :returns: the registry row that was appended, or ``None`` when the key was
        already registered or no row is declared for ``module``.
    """
    row = _BY_MODULE.get(module)
    if row is None:
        return None
    # Imported here rather than at the top: `app` reads this table while it is
    # itself being imported, so a module-level import would be a cycle. By the
    # time anything CALLS this, `register_app` is defined — that is what the
    # ordering note in `spacr.qt` is about.
    from .app import APPS, register_app

    key = row.key if key is None else str(key)
    if any(existing[0] == key for existing in APPS):
        return None
    kwargs = row.register_kwargs()
    if stage is not None:
        kwargs["stage"] = stage
    return register_app(key, row.name, row.desc,
                        row.section if section is None else section, **kwargs)


#: Every app whose registration is metadata rather than code.
#:
#: A module belongs here when its ``register()`` was one ``register_app``
#: call and nothing else. A module that does real work at registration —
#: ``chaining`` and ``prerun`` wrap other screens' factories, ``maturity``
#: reassesses stages, ``resource_cleanup`` installs a run hook — cannot be
#: declared and is still imported by
#: :func:`spacr.qt.register_self_registering_modules`.
DECLARED_APPS = (
    DeclaredApp(
        module='spacr.qt.screens.data_manager',
        key='data_manager',
        name='Data Manager',
        desc=(
            'See what a project costs in disk, and reclaim it without '
            'touching the originals'
        ),
        section='Data',
        factory='make_data_manager_screen',
        stage='alpha',
        api_module='data_manager',
    ),
    DeclaredApp(
        module='spacr.qt.screens.pipeline_graph',
        key='pipeline_graph',
        name='Pipeline Graph',
        desc=(
            'The DAG of what produced what, with everything stale or missing '
            'marked'
        ),
        section='Explore',
        factory='make_pipeline_graph_screen',
        stage='alpha',
        title='Pipeline Graph',
        intro=(
            'Every registered output of a project, drawn as the graph of '
            'what was made from what. Each box carries the run that produced '
            'it, the settings digest and the spaCR version; the colour is '
            "the artifact registry's verdict on whether it still follows "
            'from its inputs. Amber is stale — an input moved on or a '
            'material setting changed after this was written — and red is '
            'missing from disk. Click a box for the reasons and for what '
            're-running it would invalidate.'
        ),
        cli_note=(
            "Pipeline Graph is an interactive view of one project's "
            'provenance DAG; headless, call '
            'spacr.pipeline_graph.build_graph(project) and '
            'format_graph(graph) for the same content as text, or '
            'to_dot(graph) for a Graphviz figure.'
        ),
        api_module='qt/screens/pipeline_graph',
        translations=(
            'Pipelinediagram',
            'Pipeline-Graph',
            'Grafo de la tubería',
            '流程图',
            'Grafo do pipeline',
            'पाइपलाइन ग्राफ़',
            '파이프라인 그래프',
            'Vinnsluferilsrit',
            'Graphe du pipeline',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.profiler',
        key='profiler',
        name='Prediction Profiler',
        desc='Move one input of a fitted model and watch the prediction move',
        section='Explore',
        factory='make_profiler_screen',
        stage='alpha',
        title='Prediction Profiler',
        intro=(
            'Interrogate a fitted regression: sweep one input across its '
            'range, hold every other input wherever you choose, and see what '
            'the model predicts. The inputs are ranked by how far each one '
            'actually moves the prediction, so a design with thousands of '
            'gRNA terms still tells you which one to look at first. Nothing '
            'is re-fitted — the coefficients a run already wrote are the '
            'model — and the axis always says which scale it is on, because '
            'a probability, a rate and a hinge margin are not the same '
            'curve.'
        ),
        cli_note=(
            'The Prediction Profiler is an interactive sweep of one model '
            'input; headless, call spacr.profiler.profile(model, design, '
            'variable) for the same curve and '
            'spacr.profiler.sensitivity(model, design) for the same ranking.'
        ),
        api_module='qt/screens/profiler',
        translations=(
            'Prediktionsprofilerare',
            'Vorhersage-Profiler',
            'Perfilador de predicciones',
            '预测剖析器',
            'Analisador de previsões',
            'पूर्वानुमान प्रोफ़ाइलर',
            '예측 프로파일러',
            'Spágreinir',
            'Profileur de prédiction',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.qc_dashboard',
        key='qc_dashboard',
        name='QC Dashboard',
        desc=(
            'Segmentation, units, leakage, plate effects and annotator '
            'agreement in one place, with the verdict they add up to.'
        ),
        section='Explore',
        factory='make_qc_dashboard_screen',
        stage='alpha',
        title='QC Dashboard',
        intro=(
            'Every verdict here was written by the run that produced it -- '
            'this screen reads them, it does not score anything, so opening '
            'it costs a directory listing rather than minutes of mask '
            'loading. A card whose inputs are newer than it is says OUT OF '
            'DATE rather than pretending to describe them. A card that says '
            "'missing' means the check has not been run, which is not the "
            'same as clean.'
        ),
        cli_note=(
            'The QC Dashboard is a GUI screen: it aggregates verdicts other '
            'runs wrote so they can be read together. Headless, call '
            'spacr.qt.widgets.qc_summary.read_dashboard(src) and '
            'format_dashboard() instead -- that is the same code this screen '
            'runs.'
        ),
        api_module='qt/screens/qc_dashboard',
        translations=(
            'QC-panel',
            'QC-Übersicht',
            'Panel de control de QC',
            '质控面板',
            'Painel de QC',
            'QC डैशबोर्ड',
            'QC 대시보드',
            'Gæðayfirlit',
            'Tableau de bord QC',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.lineage',
        key='lineage',
        name='Lineage',
        desc='What is inside what: cell → nucleus → pathogen',
        section='Explore',
        factory='make_lineage_screen',
        stage='alpha',
        intro=(
            'Every cell with the nuclei and pathogens it contains, read off '
            'the cell_id links Measure has always written. Selecting a node '
            "highlights the same object in every other open view; 'Select "
            "with contents' highlights the whole family. Children whose "
            'cell_id names no cell get their own list — that is the two '
            'masks disagreeing, and it is a finding rather than noise.'
        ),
        cli_note=(
            'Lineage is an interactive tree; run it in the GUI (spacr-qt). '
            'Headless, spacr.lineage.build_forest gives the same tree as '
            'data.'
        ),
        api_module='qt/screens/lineage',
        translations=(
            'Härstamning',
            'Abstammung',
            'Linaje',
            '谱系',
            'Linhagem',
            'वंशावली',
            '계보',
            'Ætterni',
            'Lignée',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.experiment_design',
        key='experiment_design',
        name='Experiment Design',
        desc=(
            'Lay out conditions, controls and replicates on a plate, check '
            'the layout, and export it for the pipeline to read later.'
        ),
        section='Design',
        factory='make_experiment_design_screen',
        stage='alpha',
        title='Experiment Design',
        intro=(
            'Everything on this screen is a decision that cannot be undone '
            'after acquisition. Where the controls sit, whether a condition '
            'is confounded with a row, whether the plate edge is used at all '
            '-- no analysis repairs any of them, and all of them are free to '
            'change today. Export writes a plate_map.csv keyed the way spaCR '
            'keys measurements, so the layout is typed once instead of '
            'twice.'
        ),
        cli_note=(
            'Experiment Design is a GUI screen: it exists to draw a plate '
            'and warn about its layout before acquisition. For a headless '
            'design, build a spacr.qt.widgets.plate_layout.PlateDesign and '
            'call write_design() instead -- that is the same code this '
            'screen runs.'
        ),
        api_module='qt/screens/experiment_design',
        translations=(
            'Experimentdesign',
            'Experimentdesign',
            'Diseño de experimento',
            '实验设计',
            'Desenho do experimento',
            'प्रयोग डिज़ाइन',
            '실험 설계',
            'Tilraunahönnun',
            "Conception d'expérience",
        ),
    ),
    DeclaredApp(
        module='spacr.qt.layer_viewer',
        key='layer_viewer',
        name='Layer Viewer',
        desc='Images, masks, points and ROIs as separate layers in one world',
        section='Explore',
        factory='make_layer_viewer_screen',
        stage='alpha',
        intro=(
            'One world, many layers: an image channel, the label mask over '
            'it, the points and the shapes, each with its own colormap, '
            'opacity, blending and visibility, reordered by dragging. '
            'Picking an object here selects the same object in every other '
            'open view, and vice versa.'
        ),
        cli_note=(
            'Layer Viewer is an interactive image viewer — the layer stack, '
            'the blending and the picking are the whole feature; run it in '
            'the GUI (spacr-qt). Headless, build a spacr.layers stack from '
            'Python instead.'
        ),
        api_module='qt/layer_viewer',
        translations=(
            'Lagervisare',
            'Ebenenansicht',
            'Visor de capas',
            '图层查看器',
            'Visualizador de camadas',
            'लेयर व्यूअर',
            '레이어 뷰어',
            'Lagaskoðari',
            'Visionneuse de calques',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.graph_builder',
        key='graph_builder',
        name='Graph Builder',
        desc=(
            'Drag columns onto x / y / colour / size / facet and get a chart'
        ),
        section='Explore',
        factory='make_graph_builder_screen',
        stage='alpha',
        intro=(
            'Drop a column on X or Y and the chart appears; the plot type '
            'follows the column types. Facet down and across for small '
            'multiples on shared axes, and brush a region to highlight the '
            'same objects in every other open view.'
        ),
        cli_note=(
            'Graph Builder is interactive chart building — the drop zones '
            'and the brush are the whole feature; run it in the GUI '
            '(spacr-qt). Headless, call spacr.plot from Python and pick the '
            'columns yourself.'
        ),
        api_module='qt/screens/graph_builder',
        translations=(
            'Diagrambyggare',
            'Diagramm-Baukasten',
            'Constructor de gráficos',
            '图表构建器',
            'Construtor de gráficos',
            'ग्राफ़ बिल्डर',
            '그래프 빌더',
            'Grafasmiður',
            'Générateur de graphiques',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.power',
        key='power',
        name='Power / Design',
        desc=(
            'How many cells per well and how many wells to detect an effect '
            'of a given size'
        ),
        section='Design',
        factory='make_power_screen',
        stage='alpha',
        title='Power / Design',
        intro=(
            'Before a pooled screen runs, the only honest way to know '
            'whether it can find its hits is to simulate screens you know '
            'the truth for and fit the model you would really use. Describe '
            'the library, the plates, the classifier and the effect you '
            'expect; this sweeps cells-per-well and wells, and reports the '
            'fraction of simulated screens in which the model recovered the '
            'planted hits. The departures from the R package it is ported '
            'from — including that the R version overstates power — are '
            'shown next to the number, not in a footnote.'
        ),
        cli_note=(
            'Interactive design exploration; '
            'spacr.power_model.scan_parameters() is the headless equivalent '
            'and takes the same parameters.'
        ),
        api_module='qt/screens/power',
        defaults_module='spacr.qt.screens.power',
        translations=(
            'Statistisk styrka / design',
            'Teststärke / Design',
            'Potencia / diseño',
            '检验效能 / 设计',
            'Potência / delineamento',
            'सांख्यिकीय शक्ति / डिज़ाइन',
            '검정력 / 설계',
            'Tölfræðilegt afl / hönnun',
            'Puissance / plan',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.run_compare',
        key='run_compare',
        name='Run Compare',
        desc='Put two runs side by side: settings, counts and hit-list diffs',
        section='Results & QC',
        factory='make_run_compare_screen',
        stage='alpha',
        title='Run Compare',
        intro=(
            'Two runs of the same project, side by side. What you changed '
            '(the settings diff, grouped the way the settings panel groups '
            'them, showing only what moved), what came out (objects, wells '
            'and fields per plate) and which hits moved — appeared, '
            'vanished, or just changed rank. Runs that are not comparable '
            'are not diffed: the banner says why, and you can override it.'
        ),
        cli_note=(
            'Run Compare is an interactive side-by-side of two runs; '
            'headless, call spacr.run_compare.runs_in(project) to list them '
            'and spacr.run_compare.compare_runs(a, b) for the same three '
            'tables.'
        ),
        api_module='qt/screens/run_compare',
        translations=(
            'Jämför körningar',
            'Läufe vergleichen',
            'Comparar ejecuciones',
            '运行对比',
            'Comparar execuções',
            'रन तुलना',
            '실행 비교',
            'Bera saman keyrslur',
            'Comparer les exécutions',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.tabulate',
        key='tabulate',
        name='Tabulate',
        desc=(
            'Pivot the measurement table — rows, columns, aggregations, and '
            'the n behind each one'
        ),
        section='Explore',
        factory='make_tabulate_screen',
        stage='alpha',
        intro=(
            'Drag columns onto Rows and Columns to group by them, a '
            'measurement onto Values to summarise it, and tick the '
            'statistics you want. plateID / rowID / columnID down the rows '
            'is a plate summary. Every cell prints its n, because a mean '
            'over four objects and a mean over four thousand look the same '
            'otherwise, and a combination with no objects is blank rather '
            'than zero. Export the table, or hand it to the Graph Builder '
            'below.'
        ),
        cli_note=(
            'Interactive pivot table; spacr.qt.widgets.pivot_spec.pivot() is '
            'the headless equivalent.'
        ),
        api_module='qt/screens/tabulate',
        translations=(
            'Tabellera',
            'Tabellieren',
            'Tabular',
            '汇总表',
            'Tabular',
            'सारणीबद्ध',
            '표 만들기',
            'Taflugerð',
            'Tabuler',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.investigate_hit',
        key='investigate_hit',
        name='Investigate Hit',
        desc=(
            'Return one exact regression hit to cross-fitted candidate cells '
            'and well-level quantitative evidence'
        ),
        section='Results & QC',
        factory='_make_screen',
        stage='alpha',
        title='Investigate Hit',
        intro=(
            'Carry the exact regression run, gene, phenotype direction, FDR '
            'and guide support back to measured cells. The first output is '
            'an honest score-based review ranking. An optional hierarchical '
            'mixture then assigns cross-fitted hit-like probabilities '
            'without forcing sequencing fraction to equal cell prevalence. '
            'Comparisons use wells as the independent unit; stored calls are '
            'versioned and never overwrite hand annotations.'
        ),
        api_module='hit_investigation',
        entry='spacr.hit_investigation:investigate_hit',
        defaults_module='spacr.hit_investigation',
        translations=(
            'Undersök träff',
            'Treffer untersuchen',
            'Investigar acierto',
            '调查命中',
            'Investigar acerto',
            'हिट की जाँच करें',
            '히트 조사',
            'Rannsaka niðurstöðu',
            'Examiner le résultat',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.trellis',
        key='trellis',
        name='Small Multiples',
        desc='One chart per group, in a grid, on axes that really are shared',
        section='Explore',
        factory='make_trellis_screen',
        stage='alpha',
        intro=(
            'Drop a column on X or Y to say what each panel shows, then a '
            'grouping column on Facet ↓ or Facet → to repeat it once per '
            'level. Axes are shared by default, so a shift between panels is '
            'a shift in the data; free, per-row and per-column scales are '
            'available and the grid says so when they are on. Every panel '
            'prints its n.'
        ),
        cli_note=(
            'Small Multiples is interactive: the drop zones, the scale '
            'options and the brush are the feature. Run it in the GUI '
            '(spacr-qt). Headless, spacr.qt.widgets.trellis_spec.trellis() '
            'computes the same grid — panels, scales and per-panel n — with '
            'no Qt involved.'
        ),
        api_module='qt/screens/trellis',
        translations=(
            'Smådiagram',
            'Kleine Vielfache',
            'Múltiplos pequeños',
            '小型多组图',
            'Pequenos múltiplos',
            'स्मॉल मल्टीपल्स',
            '스몰 멀티플',
            'Smámyndaröð',
            'Petits multiples',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.gate_editor',
        key='gate_editor',
        name='Gate Editor',
        desc='Draw a threshold or a region on a plot; it becomes a filter',
        section='Explore',
        factory='make_gate_editor_screen',
        stage='alpha',
        intro=(
            'The flow-cytometry gesture on measurement tables. Drag a '
            'threshold across a histogram or click a polygon round a cloud '
            'on a two-parameter scatter, name it, and the shape becomes a '
            'filter every open view honours. Gates nest — gate on gate on '
            'gate — and each one shows its n, its percentage of its parent '
            'and its percentage of the whole table. Save the strategy and '
            're-apply it to the next plate.'
        ),
        cli_note=(
            'The Gate Editor is drawing on a plot; run it in the GUI '
            '(spacr-qt). Headless, spacr.qt.widgets.gate_spec.GateSet.load() '
            'reads a saved strategy and .population(frame, name) applies it '
            '— no Qt involved, so a gate drawn once can gate a whole '
            'campaign from a script.'
        ),
        api_module='qt/screens/gate_editor',
        translations=(
            'Gate-redigerare',
            'Gate-Editor',
            'Editor de compuertas',
            '门控编辑器',
            'Editor de gates',
            'गेट संपादक',
            '게이트 편집기',
            'Gate-ritill',
            'Éditeur de gates',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.feature_explorer',
        key='feature_explorer',
        name='Feature Explorer',
        desc='Every feature ranked by how well it separates the classes',
        section='Explore',
        factory='make_feature_explorer_screen',
        stage='alpha',
        intro=(
            'spaCR measures hundreds of features per object, so the ranking '
            'is the feature and the plotting is the easy half. Pick the '
            'column that says which class each object is in and every '
            'continuous column is scored and sorted by separation — AUC by '
            'default, because it is rank-based, unit-free and assumes '
            'nothing about the distributions. What the chosen statistic '
            'cannot see is printed next to it, a feature whose classes '
            'differ in spread rather than level is flagged, and the shuffle '
            'test says what the best of your features reaches by chance.'
        ),
        cli_note=(
            'The Feature Explorer is a ranked table you scroll; run it in '
            'the GUI (spacr-qt). Headless, '
            'spacr.qt.widgets.feature_rank.rank_features(frame, spec) '
            'returns the same ranking with every statistic per feature and '
            'no Qt involved.'
        ),
        api_module='qt/screens/feature_explorer',
        translations=(
            'Egenskapsutforskaren',
            'Merkmals-Explorer',
            'Explorador de características',
            '特征浏览器',
            'Explorador de características',
            'फ़ीचर एक्सप्लोरर',
            '특징 탐색기',
            'Eiginleikakönnuður',
            'Explorateur de caractéristiques',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.outliers',
        key='outliers',
        name='Outliers',
        desc=(
            'Robust per-object and per-well outlier detection — MAD, Tukey '
            'and MCD Mahalanobis'
        ),
        section='Explore',
        factory='make_outliers_screen',
        stage='alpha',
        intro=(
            'Finds the objects that are wrong and, separately, the wells '
            'that are wrong — which is usually the one that matters, and '
            'which per-object flags are nearly blind to: a well shifted as a '
            'whole flags almost none of its individual cells. Nothing is '
            'estimated from a mean or an SD, because the outliers would move '
            'both. Pick features, pick a rule — a modified z against the '
            "median, Tukey's fence, or a robust multivariate distance whose "
            'threshold is a stated false-positive rate — and the flags '
            'arrive as added columns. No row is ever dropped.'
        ),
        cli_note=(
            'Outliers is an interactive QC surface: the feature list, the '
            'method and the threshold are the feature; run it in the GUI '
            '(spacr-qt). Headless, '
            'spacr.qt.widgets.outlier_model.detect_outliers() computes '
            'exactly the same object flags, well scores and report with no '
            'Qt involved.'
        ),
        api_module='qt/screens/outliers',
        translations=(
            'Avvikare',
            'Ausreißer',
            'Valores atípicos',
            '离群值',
            'Valores atípicos',
            'आउटलायर',
            '이상치',
            'Frávik',
            'Valeurs aberrantes',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.dose_response',
        key='dose_response',
        name='Dose–Response',
        desc='4PL curves and EC50s, with an interval that can say no',
        section='Design',
        factory='make_dose_response_screen',
        stage='alpha',
        intro=(
            'Point it at a concentration column and a response column and it '
            'fits a four-parameter logistic per gene or compound, in '
            'log10(EC50) so the interval is multiplicative and never reaches '
            'below zero. The interval is a profile likelihood by default, '
            'because the usual asymptotic one is finite even for a series '
            'that never reached a plateau: when the midpoint is outside the '
            "doses you tested, this reports 'EC50 > 30 µM' and no point "
            'estimate rather than a confident wrong number. Bell-shaped '
            'series — cytotoxicity at the top dose — are refused with the '
            'concentrations where they turn. R² is shown with the warning '
            'that it means almost nothing on a sigmoid, next to the '
            'lack-of-fit test against pure error that does.'
        ),
        cli_note=(
            'Dose–Response is interactive: choosing the columns and reading '
            'the refusals is the feature. Run it in the GUI (spacr-qt). '
            'Headless, spacr.qt.widgets.dose_response.fit_frame() computes '
            'the same curves, intervals, bounds and lack-of-fit tests with '
            'no Qt involved.'
        ),
        api_module='qt/screens/dose_response',
        translations=(
            'Dos–respons',
            'Dosis-Wirkung',
            'Dosis–respuesta',
            '剂量反应',
            'Dose–resposta',
            'खुराक–अनुक्रिया',
            '용량–반응',
            'Skammtasvörun',
            'Dose–réponse',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.control_chart',
        key='control_chart',
        name='Control Charts',
        desc=(
            'Track a control plate by plate and see drift before it ruins a '
            'screen'
        ),
        section='Results & QC',
        factory='make_control_chart_screen',
        stage='alpha',
        intro=(
            "A campaign's controls are supposed to be the same thing every "
            'time, and when they stop being the same, hit calling and '
            'normalisation are already wrong. Pick the plate column, the run '
            'order and the control, and the chart puts limits round it: an '
            'individuals / moving-range chart when a plate has one control '
            'well, X-bar/S when it has several, and a robust variant when '
            'one bad plate would drag the classical limits out. Sigma comes '
            'from short-term variation, never from the spread of the whole '
            'series — that one is inflated by exactly the drift you are '
            'looking for. Limits are estimated from a stated baseline and '
            'applied forward, and every Nelson rule that fires is marked on '
            'the plate and named in words, along with how many false alarms '
            'the rule set you chose is worth over a campaign this long.'
        ),
        cli_note=(
            'Control Charts is a picture you read: the zones, the marked '
            'plates and the rule list are the feature. Run it in the GUI '
            '(spacr-qt). Headless, '
            'spacr.qt.widgets.control_chart.control_chart(frame, spec) '
            'returns the same limits, the same violations and the same '
            'report text with no Qt involved, so a QC gate in a script can '
            'refuse a campaign on it.'
        ),
        api_module='qt/screens/control_chart',
        translations=(
            'Styrdiagram',
            'Regelkarten',
            'Gráficos de control',
            '控制图',
            'Cartas de controlo',
            'कंट्रोल चार्ट',
            '관리도',
            'Stýririt',
            'Cartes de contrôle',
        ),
    ),
    DeclaredApp(
        module='spacr.qt.screens.project_browser',
        key='project_browser',
        name='Project Browser',
        desc='Every project on disk: stage, size, last run and what is stale',
        section='Data',
        factory='make_project_browser_screen',
        stage='alpha',
        intro=(
            'Point it at the folder your experiments live in and it lists '
            'every spaCR project under it: how far each one got, what it '
            'costs on disk, when it last produced anything, and which of its '
            'results no longer match the data underneath them. A project '
            'spaCR has never recorded — a plate folder copied from a '
            'colleague this morning — is listed too, with everything the '
            'filesystem can answer; what it will not do is call it current, '
            'because with no run record there is nothing to compare it '
            'against. Nothing here is computed twice: the stage is which '
            "declared outputs exist, the size is the Data Manager's own "
            "walk, the staleness is the artifact registry's verdict, and the "
            "next step is the offer that module's own screen makes."
        ),
        cli_note=(
            'Project Browser is a table you read and sort. Run it in the GUI '
            '(spacr-qt). Headless, spacr.projects.browse([root]) returns the '
            'same summaries and spacr.projects.format_projects prints the '
            'same table, so a nightly job can mail you which projects went '
            'stale.'
        ),
        api_module='qt/screens/project_browser',
        translations=(
            'Projektbläddrare',
            'Projektbrowser',
            'Explorador de proyectos',
            '项目浏览器',
            'Navegador de projetos',
            'प्रोजेक्ट ब्राउज़र',
            '프로젝트 브라우저',
            'Verkefnavafri',
            'Navigateur de projets',
        ),
    ),
)

_BY_MODULE = {row.module: row for row in DECLARED_APPS}
_BY_KEY = {row.key: row for row in DECLARED_APPS}
