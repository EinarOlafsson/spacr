"""Application metadata and lazy factories for the Qt interface.

:data:`DECLARED_APPS` is the canonical source for application names,
descriptions, navigation sections, documentation links, and factory paths.
Screen modules read their metadata from this catalog instead of duplicating
the same strings.

:class:`LazyScreenFactory` imports a screen module only when the application
is opened. Keeping this module free of Qt and scientific-computing imports
reduces startup work while preserving the ordinary screen-factory interface.

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
    """Store the registration metadata for one Qt application.

    This dependency-light class avoids importing the additional modules used
    by dataclasses or named tuples during application startup.

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
        """Record one module's declaration.

        :param module: the Python module that declares it.
        :param key: the registry key everything else dispatches on.
        :param name: the short name shown in the dock and on Home.
        :param desc: the one-line blurb.
        :param section: which Home section it belongs to.
        :param factory: builds the screen; usually a :class:`LazyScreenFactory`.
        :param stage: the release stage, used to gate visibility.
        :param title: the masthead title, falling back to ``name``.
        :param intro: the longer blurb shown on the screen itself.
        :param cli_note: how to reach the same thing from the command line.
        :param api_module: the module the API help link points at.
        :param entry: the callable a run enters through.
        :param defaults_module: where its settings defaults live.
        :param translations: extra translation catalogues to load with it.
        """
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
        """Return the key and the module that declared it."""
        return f"DeclaredApp({self.key!r} from {self.module!r})"

    def register_kwargs(self) -> dict:
        """Return populated keyword arguments for ``register_app``.

        Empty optional fields are omitted so ``register_app`` can apply its
        documented fallbacks. Factory paths are represented by
        :class:`LazyScreenFactory` instances and do not import screen modules.
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
    """Resolve a screen factory when it is first requested.

    :func:`spacr.qt.app.registered_factory` resolves this proxy before normal
    screen construction, allowing signature inspection to use the underlying
    callable. Direct calls are also supported; only keyword arguments accepted
    by the resolved factory are forwarded unless it accepts ``**kwargs``.

    :param module: dotted path of the module holding the factory. NOT
        imported here -- that is the whole point: naming a screen must not
        pull its imports into application startup.
    :param attribute: the factory's name inside that module.
    """

    __slots__ = ("module", "attribute", "_resolved")

    def __init__(self, module: str, attribute: str):
        """Record where a screen class lives, without importing it.

        :param module: the module to import on first use.
        :param attribute: the name to take out of it.
        """
        self.module = module
        self.attribute = attribute
        self._resolved = None

    def __repr__(self) -> str:
        """Return the target and whether it has been imported yet."""
        state = "resolved" if self._resolved is not None else "not imported"
        return f"<lazy screen factory {self.module}:{self.attribute} ({state})>"

    def resolve(self):
        """Import and cache the underlying screen factory.

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
        """Import the screen class if needed and build one.

        Keyword arguments the factory does not accept are dropped rather than
        raising, so a caller can offer ``threaded=`` or ``link=`` to every
        screen and let each take what it understands. ``inspect`` is imported
        here rather than at module scope: it costs a dozen modules of its own,
        and this module is read while the splash screen is up to avoid exactly
        that bill.

        :param kwargs: passed to the factory, filtered to what it accepts unless
            it takes ``**kwargs``.
        :returns: the constructed screen.
        """
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
    """Register the catalog entry for ``module`` without loading its screen.

    Registration is idempotent because startup and direct module registration
    can reach the same catalog entry. If its key is already registered, this
    function leaves the registry unchanged.

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
            'Inspect project disk usage and remove derived data while '
            'preserving source images'
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
            'Visualize data-product dependencies and identify missing or '
            'stale outputs'
        ),
        section='Data',
        factory='make_pipeline_graph_screen',
        stage='alpha',
        title='Pipeline Graph',
        intro=(
            'Displays registered project outputs as a dependency graph. Each '
            'node reports the producing run, settings digest, spaCR version, '
            'and current status. A stale output no longer matches its inputs '
            'or material settings; a missing output is absent from disk. '
            'Select a node to review the status evidence and the downstream '
            'outputs that would be invalidated by re-running it.'
        ),
        cli_note=(
            'Pipeline Graph is interactive. For headless use, call '
            'spacr.pipeline_graph.build_graph(project), then format_graph() '
            'for text or to_dot() for Graphviz output.'
        ),
        api_module='qt/screens/pipeline_graph',
        translations=(
            'Pipelinediagram',
            'Pipeline-Graph',
            'Grafo del flujo de trabajo',
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
        desc=(
            "Evaluate how a fitted model's prediction changes across one "
            'input variable'
        ),
        section='Core',
        factory='make_profiler_screen',
        stage='alpha',
        title='Prediction Profiler',
        intro=(
            'Varies one predictor across its range while holding the other '
            'predictors at selected values, then evaluates the stored fitted '
            'model. Predictors are ranked by the resulting change in the '
            'prediction. The model is not re-fitted, and the response axis '
            'identifies whether values are probabilities, rates, margins, '
            'or another model-specific scale.'
        ),
        cli_note=(
            'For headless use, call spacr.profiler.profile(model, design, '
            'variable) for a response curve and '
            'spacr.profiler.sensitivity(model, design) to rank predictors.'
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
        # "QC", NOT "QC Dashboard". Asked for on 2026-08-31 as part of
        # making ONE QC module: Layer Viewer and Control Charts folded in
        # as buttons, and the module that hosts them is now just QC. The
        # key is unchanged -- it is in saved sessions, run records and
        # settings files, and renaming a display name must not rename
        # anything that has been written to disk.
        name='QC',
        desc=(
            'Review stored checks for segmentation, units, leakage, plate '
            'effects, and annotation agreement'
        ),
        section='Data',
        factory='make_qc_dashboard_screen',
        stage='alpha',
        title='QC',
        intro=(
            'Summarizes quality-control results stored by completed runs; '
            'opening this screen does not recompute masks or measurements. '
            'An out-of-date result was produced from older inputs. A missing '
            'result indicates that the corresponding check has not been run '
            'and must not be interpreted as a passing result.'
        ),
        cli_note=(
            'For headless use, call '
            'spacr.qt.widgets.qc_summary.read_dashboard(src) and '
            'format_dashboard() to read and format the same stored results.'
        ),
        api_module='qt/screens/qc_dashboard',
        # ALL NINE IDENTICAL, and that is a rule rather than laziness.
        # "QC" is declared in `tools/build_i18n_catalogs.py::_IDENTITY_TEXT`
        # alongside PNG and RGB -- text that must stay byte-identical in
        # every language because it is an identifier, not a word.
        # Translating it to 质控 and Gæðaeftirlit, which is what the old
        # "QC Dashboard" names did, breaks that rule; the test
        # `test_standalone_technical_identity_values_remain_exact_in_every_language`
        # is what caught it.
        translations=('QC',) * 9,
    ),
    DeclaredApp(
        module='spacr.qt.screens.lineage',
        key='lineage',
        name='Lineage',
        desc=(
            'Inspect cell-containment relationships for nuclei, pathogens, '
            'and organelles'
        ),
        section='Data',
        factory='make_lineage_screen',
        stage='alpha',
        intro=(
            'Builds a containment tree from the cell_id links written during '
            'measurement. Selecting a node highlights the corresponding '
            'object in other open views; selecting it with its contents also '
            'highlights all descendants. Child objects whose cell_id does '
            'not identify a measured cell are listed separately as potential '
            'segmentation or linking discrepancies.'
        ),
        cli_note=(
            'For headless use, call spacr.lineage.build_forest() to return '
            'the same containment structure as data.'
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
            'Design plate layouts for conditions, controls, and replicates '
            'and export a pipeline-ready map'
        ),
        section='Data',
        factory='make_experiment_design_screen',
        stage='alpha',
        title='Experiment Design',
        intro=(
            'Assign conditions, controls, and replicates before acquisition. '
            'Layout checks identify row, column, and plate-edge confounding. '
            'Export writes plate_map.csv using the same identifiers as spaCR '
            'measurements, allowing the design to be reused by the analysis '
            'pipeline without manual re-entry.'
        ),
        cli_note=(
            'For headless use, create a '
            'spacr.qt.widgets.plate_layout.PlateDesign and call '
            'write_design() to validate and export the layout.'
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
        desc=(
            'Visualize images, masks, points, and regions of interest as '
            'synchronized layers'
        ),
        section='Data',
        factory='make_layer_viewer_screen',
        stage='alpha',
        intro=(
            'Displays image channels, label masks, points, and regions of '
            'interest in a shared coordinate system. Configure each layer\'s '
            'colour map, opacity, blending, visibility, and order. Object '
            'selection is synchronized with other open views.'
        ),
        cli_note=(
            'Layer visualization and interactive object selection require '
            'the GUI. For headless processing, construct a spacr.layers '
            'stack from Python.'
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
            'Create plots by assigning variables to axes, colour, size, and '
            'facets'
        ),
        section='Tools',
        factory='make_graph_builder_screen',
        stage='alpha',
        intro=(
            'Assign columns to the x and y axes; the available plot type is '
            'selected from their data types. Assign row and column facets to '
            'create small multiples with shared axes. Brushing a region '
            'highlights the corresponding objects in other open views.'
        ),
        cli_note=(
            'Graph Builder requires the GUI for interactive variable '
            'assignment and brushing. For headless plotting, call spacr.plot '
            'with explicit column selections.'
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
            'Estimate the cells and wells required to detect a specified '
            'effect size'
        ),
        section='Data',
        factory='make_power_screen',
        stage='alpha',
        title='Power / Design',
        intro=(
            'Simulates experiments under the specified library, plate, '
            'classifier, and effect-size assumptions, then fits the selected '
            'analysis model. The parameter scan varies cells per well and '
            'the number of wells and reports the fraction of simulations in '
            'which the planted effects are recovered.'
        ),
        cli_note=(
            'For headless use, call spacr.power_model.scan_parameters() with '
            'the same simulation and design parameters.'
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
        desc='Compare settings, object counts, and ranked hits between runs',
        section='Data',
        factory='make_run_compare_screen',
        stage='alpha',
        title='Run Compare',
        intro=(
            'Compares two runs from the same project. The report groups '
            'changed settings by their settings-panel category, compares '
            'object, well, and field counts per plate, and identifies hits '
            'that were added, removed, or re-ranked. Incompatible runs are '
            'not compared unless the compatibility warning is overridden.'
        ),
        cli_note=(
            'For headless use, call spacr.run_compare.runs_in(project) to '
            'list runs and spacr.run_compare.compare_runs(a, b) to produce '
            'the comparison tables.'
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
            'Aggregate measurement data into pivot tables with groupwise '
            'sample sizes'
        ),
        section='Data',
        factory='make_tabulate_screen',
        stage='alpha',
        intro=(
            'Assign grouping variables to rows and columns, select a '
            'measurement as the value, and choose one or more summary '
            'statistics. Each table cell reports its sample size; empty '
            'groups remain blank rather than being represented as zero. '
            'Results can be exported or passed to Graph Builder.'
        ),
        cli_note=(
            'For headless use, call spacr.qt.widgets.pivot_spec.pivot() with '
            'an equivalent pivot specification.'
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
            'Link a regression hit to cross-fitted candidate cells and '
            'well-level quantitative evidence'
        ),
        section='Core',
        factory='_make_screen',
        stage='alpha',
        title='Investigate Hit',
        intro=(
            'Links a selected regression run, gene, phenotype direction, '
            'false-discovery rate, and guide support to measured cells. The '
            'initial output ranks candidates by their scores. An optional '
            'hierarchical mixture model estimates cross-fitted hit-like '
            'probabilities without equating sequencing fraction with cell '
            'prevalence. Comparisons treat wells as independent units, and '
            'stored calls are versioned without replacing manual annotations.'
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
        desc='Create one plot per group with shared or independent axes',
        section='Tools',
        factory='make_trellis_screen',
        stage='alpha',
        intro=(
            'Assign x and y variables and one or more grouping variables to '
            'create a grid of plots. Axes are shared by default so values can '
            'be compared directly between panels; independent, row-specific, '
            'and column-specific scales are also available and are labelled '
            'when active. Each panel reports its sample size.'
        ),
        cli_note=(
            'For headless use, call '
            'spacr.qt.widgets.trellis_spec.trellis() to compute the panel '
            'layout, scales, and per-panel sample sizes.'
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
        desc='Define threshold and polygon gates as reusable data filters',
        section='Tools',
        factory='make_gate_editor_screen',
        stage='alpha',
        intro=(
            'Define a threshold on a histogram or a polygon on a '
            'two-variable scatter plot, then save the region as a named '
            'filter shared by open views. Gates can be nested. Each gate '
            'reports its sample size and its percentage of both the parent '
            'population and the complete table. Saved gating strategies can '
            'be applied to subsequent plates.'
        ),
        cli_note=(
            'Creating gates requires the GUI. For headless use, load a saved '
            'strategy with spacr.qt.widgets.gate_spec.GateSet.load() and '
            'apply it with population(frame, name).'
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
        desc='Rank measured features by their ability to distinguish classes',
        section='Core',
        factory='make_feature_explorer_screen',
        stage='alpha',
        intro=(
            'Select the column that defines object classes, then score and '
            'rank continuous features by class separation. The default AUC '
            'metric is rank-based and independent of measurement units. The '
            'results describe limitations of the selected statistic, flag '
            'features whose classes differ primarily in spread, and use a '
            'permutation test to estimate the best separation expected by '
            'chance across the feature set.'
        ),
        cli_note=(
            'For headless use, call '
            'spacr.qt.widgets.feature_rank.rank_features(frame, spec) to '
            'return the same feature-level statistics and ranking.'
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
            'Detect robust univariate and multivariate outliers at object '
            'and well levels'
        ),
        section='Data',
        factory='make_outliers_screen',
        stage='alpha',
        intro=(
            'Evaluates object-level and well-level deviations separately, '
            'allowing detection of wells whose distributions shift without '
            'producing extreme individual objects. Available methods include '
            'modified z scores based on the median absolute deviation, Tukey '
            'fences, and robust minimum-covariance-determinant Mahalanobis '
            'distance with a specified false-positive threshold. Results are '
            'added as columns; input rows are retained.'
        ),
        cli_note=(
            'For headless use, call '
            'spacr.qt.widgets.outlier_model.detect_outliers() to compute the '
            'same object flags, well scores, and report.'
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
        desc=(
            'Fit four-parameter logistic curves and estimate EC50 with '
            'profile-likelihood intervals'
        ),
        section='Data',
        factory='make_dose_response_screen',
        stage='alpha',
        intro=(
            'Fits a four-parameter logistic model for each gene or compound '
            'using selected concentration and response columns. EC50 is '
            'estimated on a log10 scale, and profile-likelihood intervals are '
            'used by default. When the midpoint lies outside the tested dose '
            'range, the result is reported as a bound instead of an '
            'unsupported point estimate. Non-monotonic series are identified '
            'with their turning concentrations. Results include R² with an '
            'interpretive warning and a lack-of-fit test against pure error.'
        ),
        cli_note=(
            'For headless use, call '
            'spacr.qt.widgets.dose_response.fit_frame() to compute curves, '
            'intervals, bounds, and lack-of-fit tests.'
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
            'Monitor control measurements across plates and detect process '
            'drift'
        ),
        section='Data',
        factory='make_control_chart_screen',
        stage='alpha',
        intro=(
            'Select the plate identifier, acquisition order, and control '
            'population. The application uses an individuals/moving-range '
            'chart for one control well per plate, an X-bar/S chart for '
            'multiple wells, or a robust alternative when requested. Limits '
            'are estimated from a defined baseline using short-term '
            'variation and then applied to subsequent plates. Nelson-rule '
            'violations are labelled, together with the expected false-alarm '
            'count for the selected rule set and campaign length.'
        ),
        cli_note=(
            'For headless use, call '
            'spacr.qt.widgets.control_chart.control_chart(frame, spec) to '
            'return the same limits, violations, and report text.'
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
        desc='Review project stage, disk usage, last run, and stale outputs',
        section='Data',
        factory='make_project_browser_screen',
        stage='alpha',
        intro=(
            'Scans a selected root folder for spaCR projects and reports '
            'their completed stage, disk usage, most recent output, and stale '
            'results. Unregistered project folders are included using the '
            'metadata available from the file system, but are not labelled '
            'current because they have no recorded run for comparison. Stage, '
            'size, staleness, and suggested next steps use the same project '
            'and artifact metadata as their corresponding applications.'
        ),
        cli_note=(
            'For headless use, call spacr.projects.browse([root]) to return '
            'project summaries and spacr.projects.format_projects() to '
            'format the table.'
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
