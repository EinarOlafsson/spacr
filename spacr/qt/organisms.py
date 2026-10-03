"""Sourced organism introductions and image-analysis module proposals.

The assay keys are existing pipeline identifiers. A proposal served by a
general module opens that module through ``workflows``; any other proposal
has no route, so the organism pages cannot run an unimplemented analysis.
Artwork provenance, licences and checksums ship in organism_sources.json.
"""
from __future__ import annotations


ORGANISMS = {
    "toxoplasma": {
        "name": "Toxoplasma gondii",
        "description": (
            "Toxoplasma gondii is an intracellular protozoan parasite that causes "
            "toxoplasmosis in humans and other warm-blooded animals. Its rapidly "
            "growing tachyzoites and persistent tissue cysts represent different "
            "stages of infection. Imaging can follow entry into host cells, "
            "replication within vacuoles, and damage to a cell monolayer."
        ),
        "source": "https://www.cdc.gov/dpdx/toxoplasmosis/index.html",
        "diagram": "organism_apicomplexa.svg",
        "diagram_note": (
            "Explore the apicomplexan cell using Starplast's Toxoplasma hyperLOPIT "
            "labels. Hover for a location description, check several compartments "
            "to keep them highlighted, or use Clear components to reset them. "
            "Several classes share a shape; dense granules use the generic "
            "cytoplasmic-granule shape. Ribosomes, proteasomes, apical classes "
            "and mixed endomembrane vesicles have no matching shape here. "
            "The illustration shows compartment names, not protein measurements."
        ),
        "sections": (
            ("Entry and the intracellular niche", (
                "Invasion Assay separates parasites attached to a host cell from "
                "those that have entered it. Recruitment measures host-protein "
                "enrichment around the parasite-containing vacuole. Host–Pathogen "
                "combines vacuole-level recruitment, parasite counts and host "
                "infection denominators in one analysis. Individual parasite "
                "counts require suitable parasite masks or a validated estimator; "
                "whole-vacuole masks alone cannot supply those counts. Used together, "
                "these modules help separate a change in entry from a change in "
                "the host response after entry. Choose image channels and marker "
                "definitions that make those two populations distinguishable."
            ), ("invasion", "recruitment", "host_pathogen")),
            ("Replication and the complete lytic cycle", (
                "Replication Assay counts parasites per vacuole, providing a "
                "readout of intracellular growth at the sampled time. Plaque "
                "Assay measures cleared areas across a host-cell monolayer and "
                "therefore integrates multiple rounds of infection and growth. "
                "A smaller plaque can reflect more than one defect: compare "
                "the plaque result with invasion and replication measurements "
                "before assigning it to a particular step."
            ), ("replication", "analyze_plaques")),
            ("Movement, exit and persistent stages", (
                "Gliding motility opens the Motility Assay to measure the "
                "speed and straightness of extracellular parasite tracks. "
                "The planned Egress module will focus on vacuole rupture and "
                "parasite exit. Bradyzoite conversion will address "
                "stage-marker and cyst-wall readouts, while Host cell damage "
                "will describe monolayer loss. These three are proposed "
                "workflows, so their tiles are marked Coming soon. Fixed images, time "
                "series and stage-specific markers answer different questions "
                "and should be selected for the intended readout."
            ), ("gliding",)),
            ("From a phenotype to candidate proteins", (
                "The compartment diagram connects imaging phenotypes to "
                "subcellular vocabulary used in Starplast. ToxoDB provides "
                "gene and genome context, while UniProt supplies protein "
                "annotations. The hyperLOPIT atlas provides experimental "
                "localization evidence for Toxoplasma proteins; a highlighted "
                "organelle here is a guide to that vocabulary, not a prediction "
                "for a selected gene or proof of the cause of a phenotype."
            ), ()),
        ),
        "links": (
            ("ToxoDB", "https://toxodb.org/toxo/"),
            ("Starplast", "https://github.com/EinarOlafsson/starplast"),
            ("Toxoplasma hyperLOPIT atlas", "https://pubmed.ncbi.nlm.nih.gov/33053376/"),
            ("UniProt", "https://www.uniprot.org/uniprotkb?query=taxonomy_id%3A5811"),
            ("NCBI Taxonomy", "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=5811"),
            ("BEI Resources", "https://www.beiresources.org/PathogensLinks.aspx"),
        ),
        "modules": (
            ("starplast", "Starplast", "Explore the Toxoplasma knowledge map in a separate alpha application.", "replication"),
            ("analyze_plaques", "Plaque Assay", "Quantify plaque number and size.", "analyze_plaques"),
            ("recruitment", "Recruitment", "Measure host-protein enrichment at the vacuole.", "recruitment"),
            ('host_pathogen', 'Host–Pathogen', 'Combine vacuole marker recruitment, parasite counts and host infection denominators.', 'host_pathogen'),
            ("invasion", "Invasion Assay", "Distinguish attached and invaded parasites.", "invasion"),
            ("replication", "Replication Assay", "Count parasites per vacuole.", "replication"),
            (None, "Egress", "Follow vacuole rupture and parasite exit over time.", "egress"),
            (None, "Gliding motility", "Measure extracellular parasite trails and speed.", "gliding"),
            (None, "Bradyzoite conversion", "Quantify cyst-wall staining and stage conversion.", "cyst"),
            (None, "Host cell damage", "Measure monolayer integrity and host-cell loss.", "damage"),
        ),
        "workflows": {
            "gliding": ("motility", {"infection_intensity_qc_scope": "none"}, (
                "Opens the Motility Assay with infection QC off, because "
                "extracellular parasites have no host cell. Segment the "
                "parasites as the tracked cell objects; it reports track "
                "speed and straightness per well.")),
        },
    },
    "plasmodium": {
        "name": "Plasmodium spp.",
        "description": (
            "Plasmodium parasites cause malaria. Their life cycle includes "
            "mosquito and vertebrate hosts, with liver and blood stages in "
            "humans. Microscopy distinguishes parasite forms in infected red "
            "blood cells. Image-based studies can measure infection frequency, "
            "stage progression, motility and responses to compounds."
        ),
        "source": "https://www.cdc.gov/dpdx/malaria/",
        "diagram": "organism_apicomplexa.svg",
        "diagram_note": (
            "A shared apicomplexan cell plan, using the SwissBioPics artwork "
            "also used in Starplast. Hover for a UniProt location description "
            "and check several compartments to keep them highlighted. "
            "Parasite shape and organelle organization vary by species and "
            "life-cycle stage; this is not a blood-stage reconstruction. "
            "Toxoplasma hyperLOPIT assignments are not transferred to Plasmodium."
        ),
        "sections": (
            ("Infection frequency and blood-stage development", (
                "Parasitaemia opens Host–Pathogen Analysis, which reports "
                "infected red blood cells relative to all measured red "
                "cells. "
                "Blood-stage staging will separate rings, trophozoites and "
                "schizonts so a shift in stage composition can be examined "
                "alongside infection frequency. Merozoite invasion will focus "
                "on entry into red blood cells. Together these readouts would "
                "help distinguish fewer newly infected cells from altered "
                "development of parasites already inside them."
            ), ("parasitaemia",)),
            ("Liver infection and parasite movement", (
                "Liver-stage growth is planned to measure parasite number "
                "and size within hepatocytes. Sporozoite motility opens the "
                "Motility Assay for gliding tracks, while Cell traversal "
                "will address host-cell wounding along parasite paths. "
                "Traversal, productive invasion and subsequent growth are "
                "different outcomes: their image markers, observation times "
                "and denominators need to be defined separately."
            ), ("sporozoite",)),
            ("Transmission stages and compound responses", (
                "Gametocyte maturity is a proposed workflow for stage and "
                "sex-associated image features when suitable markers and "
                "reference annotations are available. Drug response imaging "
                "opens Dose–Response to fit an image readout against "
                "concentration. Record "
                "the species, starting stage and exposure duration with "
                "these measurements: changes in the mix of stages can "
                "otherwise be mistaken for changes in parasite number."
            ), ("drug",)),
            ("Building an image-analysis workflow", (
                "Parasitaemia, Sporozoite motility and Drug response imaging "
                "open existing spaCR modules; the other five tiles are "
                "planned and marked Coming soon. Their descriptions define "
                "intended readouts rather than available analysis "
                "pipelines. PlasmoDB and UniProt provide "
                "genome and protein context for choosing markers and "
                "interpreting results. The apicomplexan illustration is "
                "useful for discussing compartments shared with Toxoplasma, "
                "but it does not establish that a protein has the same "
                "location in both organisms."
            ), ()),
        ),
        "links": (
            ("PlasmoDB", "https://plasmodb.org/plasmo/"),
            ("UniProt", "https://www.uniprot.org/uniprotkb?query=taxonomy_id%3A5820"),
            ("NCBI Taxonomy", "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=5820"),
            ("BEI Resources / MR4", "https://www.beiresources.org/ProgramInformation.aspx"),
        ),
        "modules": (
            (None, "Parasitaemia", "Count infected red blood cells per field.", "parasitaemia"),
            (None, "Blood-stage staging", "Classify rings, trophozoites and schizonts.", "staging"),
            (None, "Merozoite invasion", "Follow parasite entry into red blood cells.", "merozoite"),
            (None, "Liver-stage growth", "Measure parasite number and size in hepatocytes.", "liver"),
            (None, "Sporozoite motility", "Measure gliding trails and speed.", "sporozoite"),
            (None, "Cell traversal", "Measure wounded host cells along parasite paths.", "traversal"),
            (None, "Gametocyte maturity", "Classify gametocyte stage and sex from images.", "gametocyte"),
            (None, "Drug response imaging", "Measure growth inhibition across compound concentrations.", "drug"),
        ),
        "workflows": {
            "parasitaemia": ("host_pathogen", {}, (
                "Opens Host–Pathogen Analysis. Measure red blood cells as "
                "host cells, including uninfected ones, and parasites as "
                "pathogen objects; its infection fraction per well is the "
                "parasitaemia.")),
            "sporozoite": ("motility", {"infection_intensity_qc_scope": "none"}, (
                "Opens the Motility Assay with infection QC off, because "
                "gliding sporozoites have no host cell. Segment the "
                "sporozoites as the tracked cell objects; it reports track "
                "speed and straightness per well.")),
            "drug": ("dose_response", {}, (
                "Opens Dose–Response. Fit a four-parameter logistic curve "
                "and EC50 to a per-well image readout, such as "
                "parasitaemia, against compound concentration.")),
        },
    },
    "candida": {
        "name": "Candida spp.",
        "description": (
            "Candida yeasts can live on the skin and mucosal surfaces. Some "
            "species cause superficial or invasive candidiasis. Candida albicans "
            "can form budding yeasts, germ tubes and hyphae; these forms are not "
            "shared by every Candida species. Imaging can quantify morphology, "
            "biofilm growth and interactions with host cells."
        ),
        "source": "https://www.cdc.gov/candidiasis/about/index.html",
        "diagram": "organism_yeast.svg",
        "diagram_note": (
            "A generic budding-yeast cell from SwissBioPics. Hover for a "
            "UniProt location description and check several compartments to "
            "keep them highlighted. This is a cell "
            "schematic, not a Candida species identification or a depiction "
            "of every yeast, pseudohyphal and hyphal form."
        ),
        "sections": (
            ("Morphology and the transition to filaments", (
                "Filamentation is planned to quantify the frequency and "
                "extent of filament growth. Germ tube formation will focus "
                "on early outgrowth from individual cells, and Morphology "
                "will distinguish yeast, pseudohyphal and hyphal forms "
                "where those categories apply. These related modules "
                "address different points in a morphological transition. "
                "Use species-appropriate reference images and record the "
                "growth conditions when comparing their results."
            ), ()),
            ("Surface attachment and community growth", (
                "Adhesion opens the Invasion Assay, whose attached and "
                "invaded counts give the fungal cells bound to a host-cell "
                "monolayer. The planned Biofilm module will focus "
                "on collective growth, including covered area and thickness "
                "when the acquisition contains depth information. A single "
                "two-dimensional field can describe surface coverage but "
                "cannot by itself establish biofilm thickness. Choose "
                "single-plane or volumetric imaging to match the endpoint."
            ), ("adhesion",)),
            ("Interactions with host cells", (
                "Epithelial invasion opens the Invasion Assay to distinguish "
                "extracellular fungi from those internalised by epithelial "
                "cells. Phagocytosis opens Host–Pathogen Analysis to measure "
                "uptake by phagocytes. Both "
                "require a way to resolve contact from internalisation, "
                "such as appropriate differential labelling or spatial "
                "information. Report the host-cell population and fungal "
                "morphology together with the uptake readout."
            ), ("epithelial", "phagocytosis")),
            ("Antifungal response and interpretation", (
                "Antifungal response opens Dose–Response to fit fungal "
                "growth or a morphology score against drug concentration. "
                "Combining it with Morphology or Biofilm would help "
                "describe which image features change during treatment. "
                "Filamentation, Germ tube formation, Biofilm and Morphology "
                "are planned and their tiles are marked Coming soon. The "
                "Candida Genome Database and UniProt "
                "provide gene and protein annotations to support marker "
                "selection and interpretation."
            ), ("antifungal",)),
        ),
        "links": (
            ("Candida Genome Database", "https://www.candidagenome.org/"),
            ("Genome browser", "https://www.candidagenome.org/jbrowse2/"),
            ("UniProt", "https://www.uniprot.org/uniprotkb?query=taxonomy_id%3A5475"),
            ("NCBI Taxonomy", "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=5475"),
            ("BEI Resources", "https://www.beiresources.org/PathogensLinks.aspx"),
        ),
        "modules": (
            (None, "Filamentation", "Measure the frequency and extent of filament growth.", "filamentation"),
            (None, "Germ tube formation", "Quantify early germ-tube emergence per cell.", "germ_tube"),
            (None, "Biofilm", "Measure biofilm area and thickness over time.", "biofilm"),
            (None, "Adhesion", "Count fungal cells bound to a host-cell monolayer.", "adhesion"),
            (None, "Epithelial invasion", "Distinguish internalised and extracellular fungi.", "epithelial"),
            (None, "Phagocytosis", "Quantify uptake by phagocytes.", "phagocytosis"),
            (None, "Morphology", "Classify yeast, pseudohyphal and hyphal forms.", "morphology"),
            (None, "Antifungal response", "Measure growth and morphology across antifungal concentrations.", "antifungal"),
        ),
        "workflows": {
            "adhesion": ("invasion", {}, (
                "Opens the Invasion Assay. With two-colour differential "
                "staining, its attached plus invaded counts per well are "
                "the fungi bound to the monolayer.")),
            "epithelial": ("invasion", {}, (
                "Opens the Invasion Assay. Two-colour differential staining "
                "separates extracellular from internalised fungi, as for "
                "Toxoplasma invasion.")),
            "phagocytosis": ("host_pathogen", {}, (
                "Opens Host–Pathogen Analysis. Measure phagocytes as host "
                "cells and fungi as pathogen objects; it reports the "
                "fraction of phagocytes containing fungi and fungi per "
                "phagocyte. Killing is not measured.")),
            "antifungal": ("dose_response", {}, (
                "Opens Dose–Response. Fit a four-parameter logistic curve "
                "and EC50 to fungal growth or a morphology score per well "
                "against drug concentration.")),
        },
    },
    "trypanosoma": {
        "name": "Trypanosoma spp.",
        "description": (
            "Trypanosoma brucei causes African sleeping sickness and lives "
            "outside host cells in blood and tissue fluids, while Trypanosoma "
            "cruzi causes Chagas disease and replicates inside host cells as "
            "amastigotes. Both are flagellated kinetoplastids whose single "
            "mitochondrion carries a kinetoplast. Imaging can follow motility, "
            "host-cell infection, the cell cycle and drug responses."
        ),
        "source": "https://www.cdc.gov/dpdx/trypanosomiasisafrican/index.html",
        "diagram": "organism_trypanosomatid.svg",
        "diagram_note": (
            "The SwissBioPics trypanosomatid cell. Hover for a UniProt "
            "location description and check several compartments to keep "
            "them highlighted. The labels are the UniProt subcellular "
            "locations annotated for Trypanosoma proteins that this artwork "
            "draws. It shows one trypomastigote-like form; amastigotes and "
            "epimastigotes differ in flagellum length and organelle position."
        ),
        "sections": (
            ("Motility and life-cycle forms", (
                "Flagellar motility opens the Motility Assay with infection "
                "QC off, because swimming trypanosomes have no host cell; it "
                "reports track speed and straightness. The planned Stage "
                "differentiation module will classify bloodstream, stumpy "
                "and procyclic forms of T. brucei, or trypomastigotes, "
                "epimastigotes and amastigotes of T. cruzi. Record the "
                "species and life-cycle form, because flagellum length and "
                "swimming behaviour differ between them."
            ), ("gliding",)),
            ("Host-cell infection by Trypanosoma cruzi", (
                "Host-cell invasion opens the Invasion Assay, which separates "
                "attached from internalised trypomastigotes when two-colour "
                "differential staining is available. Amastigote infection "
                "opens Host–Pathogen Analysis to count infected "
                "host cells and amastigotes per cell. The planned "
                "Trypomastigote egress module will follow parasite release, "
                "and Host cell damage will describe monolayer loss. These "
                "readouts apply to T. cruzi; T. brucei does not invade cells."
            ), ("epithelial", "parasitaemia")),
            ("Cell cycle and compound responses", (
                "Cell-cycle staging is planned to count kinetoplasts and "
                "nuclei per cell, because the kinetoplast divides before the "
                "nucleus and the counts mark cell-cycle position. Drug "
                "response imaging opens Dose–Response to fit a per-well "
                "image readout, such as parasite number, against compound "
                "concentration. Report the exposure time with each curve: "
                "a delayed cell cycle and parasite killing can both lower "
                "the count."
            ), ("drug",)),
            ("Building an image-analysis workflow", (
                "Flagellar motility, Host-cell invasion, Amastigote "
                "infection and Drug response imaging open existing spaCR "
                "modules; the other four tiles are planned and marked Coming "
                "soon. TriTrypDB provides genome and gene context and "
                "UniProt supplies protein annotations. The highlighted "
                "compartments are UniProt vocabulary for choosing markers, "
                "not a measured location for a selected gene."
            ), ()),
        ),
        "links": (
            ("TriTrypDB", "https://tritrypdb.org/tritrypdb/"),
            ("Chagas disease: CDC", "https://www.cdc.gov/dpdx/trypanosomiasisamerican/index.html"),
            ("UniProt", "https://www.uniprot.org/uniprotkb?query=taxonomy_id%3A5690"),
            ("NCBI Taxonomy", "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=5690"),
            ("BEI Resources", "https://www.beiresources.org/PathogensLinks.aspx"),
        ),
        "modules": (
            (None, "Flagellar motility", "Track swimming speed and straightness.", "gliding"),
            (None, "Host-cell invasion", "Separate attached from internalised trypomastigotes.", "epithelial"),
            (None, "Amastigote infection", "Count infected host cells and amastigotes per cell.", "parasitaemia"),
            (None, "Drug response imaging", "Measure parasite growth across compound concentrations.", "drug"),
            (None, "Cell-cycle staging", "Count kinetoplasts and nuclei per cell.", "staging"),
            (None, "Stage differentiation", "Classify life-cycle forms from images.", "morphology"),
            (None, "Trypomastigote egress", "Follow parasite release from host cells.", "egress"),
            (None, "Host cell damage", "Measure monolayer integrity and host-cell loss.", "damage"),
        ),
        "workflows": {
            "gliding": ("motility", {"infection_intensity_qc_scope": "none"}, (
                "Opens the Motility Assay with infection QC off, because "
                "swimming trypanosomes have no host cell. Segment the "
                "parasites as the tracked cell objects; it reports track "
                "speed and straightness per well.")),
            "epithelial": ("invasion", {}, (
                "Opens the Invasion Assay. Two-colour differential staining "
                "separates attached from internalised T. cruzi "
                "trypomastigotes, as for Toxoplasma invasion.")),
            "parasitaemia": ("host_pathogen", {}, (
                "Opens Host–Pathogen Analysis. Measure host cells, including "
                "uninfected ones, and amastigotes as pathogen objects; it "
                "reports the infected fraction and amastigotes per cell.")),
            "drug": ("dose_response", {}, (
                "Opens Dose–Response. Fit a four-parameter logistic curve "
                "and EC50 to a per-well image readout, such as parasite "
                "number, against compound concentration.")),
        },
    },
    "leishmania": {
        "name": "Leishmania spp.",
        "description": (
            "Leishmania parasites cause cutaneous, mucosal and visceral "
            "leishmaniasis and are transmitted by sand flies. Flagellated "
            "promastigotes develop in the insect, and amastigotes replicate "
            "inside macrophages in a parasitophorous vacuole. Imaging can "
            "measure macrophage infection, parasite load, promastigote "
            "motility, stage conversion and responses to compounds."
        ),
        "source": "https://www.cdc.gov/dpdx/leishmaniasis/index.html",
        "diagram": "organism_leishmania.svg",
        "diagram_note": (
            "A schematic Leishmania promastigote, with its free anterior "
            "flagellum leaving the flagellar pocket, the kinetoplast just "
            "behind it and a central nucleus, above the rounded amastigote. "
            "Hover for a UniProt location description and check several "
            "compartments to keep them highlighted. The labels are the "
            "UniProt subcellular locations annotated for Leishmania "
            "proteins."
        ),
        "sections": (
            ("Macrophage infection and parasite load", (
                "Macrophage infection opens Host–Pathogen Analysis to "
                "report the fraction of infected macrophages and the "
                "amastigotes per macrophage, the two usual measures of "
                "parasite load. Macrophage binding opens the Invasion Assay "
                "to separate bound from internalised promastigotes when "
                "differential staining is used. The planned Vacuole size "
                "module will measure parasitophorous vacuole area, which "
                "differs between Leishmania species and grows as amastigotes "
                "multiply inside it."
            ), ("phagocytosis", "adhesion")),
            ("Promastigote development and motility", (
                "Promastigote motility opens the Motility Assay with "
                "infection QC off to measure swimming speed and "
                "straightness. The planned Metacyclogenesis module will "
                "separate procyclic from infective metacyclic promastigotes "
                "by shape, and Amastigote conversion will follow the "
                "transformation into rounded amastigotes. Record the culture "
                "day and medium, because promastigote populations change "
                "composition as cultures age."
            ), ("gliding",)),
            ("Compound responses and host damage", (
                "Drug response imaging opens Dose–Response to fit "
                "amastigote clearance or infected-cell fraction against "
                "compound concentration. The planned Host cell damage "
                "module will measure macrophage loss, so that a compound "
                "that kills host cells is not mistaken for one that clears "
                "parasites. Count host cells in every well alongside the "
                "parasite readout, and include an untreated infected control "
                "on every plate so that the curve has a defined top."
            ), ("drug",)),
            ("Building an image-analysis workflow", (
                "Macrophage infection, Macrophage binding, Promastigote "
                "motility and Drug response imaging open existing spaCR "
                "modules; the other four tiles are planned and marked Coming "
                "soon. TriTrypDB provides genome context and UniProt supplies "
                "protein annotations. The diagram shows UniProt vocabulary, "
                "not the measured location of a Leishmania protein."
            ), ()),
        ),
        "links": (
            ("TriTrypDB", "https://tritrypdb.org/tritrypdb/"),
            ("UniProt", "https://www.uniprot.org/uniprotkb?query=taxonomy_id%3A5658"),
            ("NCBI Taxonomy", "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=5658"),
            ("BEI Resources", "https://www.beiresources.org/PathogensLinks.aspx"),
        ),
        "modules": (
            (None, "Macrophage infection", "Count amastigotes per macrophage and infected cells.", "phagocytosis"),
            (None, "Macrophage binding", "Separate bound from internalised promastigotes.", "adhesion"),
            (None, "Promastigote motility", "Track swimming speed and straightness.", "gliding"),
            (None, "Drug response imaging", "Measure amastigote clearance across compound concentrations.", "drug"),
            (None, "Metacyclogenesis", "Classify procyclic and metacyclic promastigotes.", "morphology"),
            (None, "Amastigote conversion", "Follow promastigote-to-amastigote transformation.", "staging"),
            (None, "Vacuole size", "Measure parasitophorous vacuole area per infected cell.", "cyst"),
            (None, "Host cell damage", "Measure macrophage loss and monolayer integrity.", "damage"),
        ),
        "workflows": {
            "phagocytosis": ("host_pathogen", {}, (
                "Opens Host–Pathogen Analysis. Measure macrophages as host "
                "cells and amastigotes as pathogen objects; it reports the "
                "infected fraction and amastigotes per macrophage.")),
            "adhesion": ("invasion", {}, (
                "Opens the Invasion Assay. Two-colour differential staining "
                "separates bound from internalised promastigotes.")),
            "gliding": ("motility", {"infection_intensity_qc_scope": "none"}, (
                "Opens the Motility Assay with infection QC off, because "
                "promastigotes in culture have no host cell. Segment them "
                "as the tracked cell objects; it reports track speed and "
                "straightness per well.")),
            "drug": ("dose_response", {}, (
                "Opens Dose–Response. Fit a four-parameter logistic curve "
                "and EC50 to the infected fraction or amastigotes per "
                "macrophage against compound concentration.")),
        },
    },
    "giardia": {
        "name": "Giardia duodenalis",
        "description": (
            "Giardia duodenalis, also called G. lamblia or G. intestinalis, "
            "causes giardiasis, a diarrhoeal disease of the small intestine. "
            "Trophozoites swim with four pairs of flagella, carry two nuclei "
            "and attach to the intestinal epithelium with a ventral disc; "
            "infection spreads through environmentally resistant cysts. "
            "Imaging can measure attachment, motility and encystation."
        ),
        "source": "https://www.cdc.gov/dpdx/giardiasis/index.html",
        "diagram": "organism_giardia.svg",
        "diagram_note": (
            "A schematic Giardia trophozoite seen from above: a pear-shaped "
            "cell with two nuclei, the ventral adhesive disc, the median "
            "bodies and four pairs of flagella. Hover for a UniProt "
            "location description and check several compartments to keep "
            "them highlighted. The labels are UniProt locations annotated "
            "for Giardia proteins. Giardia has mitosomes instead of "
            "mitochondria and no stacked Golgi, so those are not offered."
        ),
        "sections": (
            ("Attachment and motility", (
                "Epithelial attachment opens Host–Pathogen Analysis to "
                "count trophozoites on epithelial host cells; a single "
                "plane cannot separate attachment from overlap, so confirm "
                "it with a focal plane at the cell surface. Trophozoite "
                "motility opens the Motility Assay with infection QC off "
                "to measure swimming speed and straightness. The planned "
                "Disc morphology module will measure the ventral disc."
            ), ("adhesion", "gliding")),
            ("Encystation and excystation", (
                "Encystation is planned to quantify cyst-wall staining and "
                "the fraction of trophozoites forming cysts, and "
                "Excystation will follow trophozoites emerging from cysts. "
                "These transitions take hours and depend on bile, pH and "
                "culture conditions, so report the induction protocol and "
                "the time of imaging with the result. Cysts are small and "
                "refractile, so a wall stain and a fixed focal plane help "
                "separate them from debris."
            ), ()),
            ("Division and host damage", (
                "Nuclear division is planned to follow the two nuclei "
                "through mitosis, which a cell-cycle readout must count as "
                "a pair. Barrier damage will measure epithelial monolayer "
                "integrity after exposure to trophozoites. Drug response "
                "imaging opens Dose–Response to fit trophozoite number or "
                "attachment against compound concentration. Detached "
                "trophozoites are lost when wells are washed, so decide "
                "whether attachment or survival is the endpoint."
            ), ("drug",)),
            ("Building an image-analysis workflow", (
                "Epithelial attachment, Trophozoite motility and Drug "
                "response imaging open existing spaCR modules; the other "
                "five tiles are planned and marked Coming soon. GiardiaDB "
                "provides genome context and UniProt supplies protein "
                "annotations. The generic cell shows UniProt vocabulary, not "
                "Giardia anatomy or the measured location of a protein."
            ), ()),
        ),
        "links": (
            ("GiardiaDB", "https://giardiadb.org/giardiadb/"),
            ("UniProt", "https://www.uniprot.org/uniprotkb?query=taxonomy_id%3A5740"),
            ("NCBI Taxonomy", "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=5740"),
            ("BEI Resources", "https://www.beiresources.org/PathogensLinks.aspx"),
        ),
        "modules": (
            (None, "Epithelial attachment", "Count trophozoites on epithelial cells.", "adhesion"),
            (None, "Trophozoite motility", "Track swimming speed and straightness.", "gliding"),
            (None, "Drug response imaging", "Measure trophozoite growth across compound concentrations.", "drug"),
            (None, "Encystation", "Quantify cyst-wall staining and cyst formation.", "cyst"),
            (None, "Excystation", "Follow trophozoite emergence from cysts.", "egress"),
            (None, "Disc morphology", "Measure ventral disc shape and integrity.", "morphology"),
            (None, "Nuclear division", "Follow both nuclei through mitosis.", "staging"),
            (None, "Barrier damage", "Measure epithelial monolayer integrity.", "damage"),
        ),
        "workflows": {
            "adhesion": ("host_pathogen", {}, (
                "Opens Host–Pathogen Analysis. Measure epithelial cells as "
                "host cells and trophozoites as pathogen objects; it reports "
                "the fraction of host cells with trophozoites and "
                "trophozoites per cell.")),
            "gliding": ("motility", {"infection_intensity_qc_scope": "none"}, (
                "Opens the Motility Assay with infection QC off, because "
                "swimming trophozoites have no host cell. Segment them as "
                "the tracked cell objects; it reports track speed and "
                "straightness per well.")),
            "drug": ("dose_response", {}, (
                "Opens Dose–Response. Fit a four-parameter logistic curve "
                "and EC50 to trophozoite number or attachment per well "
                "against compound concentration.")),
        },
    },
    "virus": {
        "name": "Virus infection",
        "description": (
            "Viruses replicate only inside host cells, using host machinery "
            "in compartments that differ between virus families: many DNA "
            "viruses replicate in the nucleus, and many RNA viruses build "
            "replication organelles in the cytoplasm. Imaging can measure "
            "the infected fraction of cells, plaques, the spread of "
            "infection and responses to antiviral compounds."
        ),
        "source": "https://viralzone.expasy.org/",
        "source_label": "Biology source: ViralZone",
        "diagram": "organism_virus.svg",
        "diagram_note": (
            "A schematic enveloped virus with an icosahedral capsid, "
            "tegument and spiked envelope, beside an infected host cell "
            "with nascent capsids in the nucleus, an internalised virion "
            "and budding particles. Hover for a UniProt location "
            "description and check several compartments to keep them "
            "highlighted. The labels are the virion and host-cell locations "
            "UniProt annotates for viral proteins. Virion structure and "
            "replication sites differ between virus families."
        ),
        "sections": (
            ("Infection and replication sites", (
                "Infection rate opens Host–Pathogen Analysis to count "
                "host cells positive for a viral antigen or reporter among "
                "all host cells. Recruitment opens the Recruitment module to "
                "measure host-protein enrichment at viral replication "
                "compartments. Define the positivity threshold from mock-"
                "infected wells, and record the multiplicity of infection "
                "and fixation time with every measurement."
            ), ("parasitaemia", "recruitment")),
            ("Plaques and spread", (
                "Plaque Assay opens the Plaque Assay module to count and "
                "measure plaques in a stained monolayer. The planned Viral "
                "spread module will follow infection foci over time, and "
                "Cytopathic effect will measure host-cell rounding, "
                "detachment and loss. Plaque size integrates several rounds "
                "of replication and spread, so compare it with the infection "
                "rate before assigning a defect to one step."
            ), ("analyze_plaques",)),
            ("Entry, fusion and antivirals", (
                "Virus entry is planned to separate bound from internalised "
                "virions, and Syncytium formation will detect fused, "
                "multinucleated host cells. Antiviral response opens "
                "Dose–Response to fit the infected fraction against "
                "compound concentration. Count host cells in each well too, "
                "so that a cytotoxic compound is not read as an antiviral, "
                "and keep the multiplicity of infection constant across the "
                "dilution series."
            ), ("drug",)),
            ("Building an image-analysis workflow", (
                "Infection rate, Plaque Assay, Recruitment and Antiviral "
                "response open existing spaCR modules; the other four tiles "
                "are planned and marked Coming soon. ViralZone describes "
                "virus families and their replication cycles, and UniProt "
                "supplies viral and host protein annotations. The diagram "
                "shows UniProt vocabulary, not where a selected protein is "
                "measured."
            ), ()),
        ),
        "links": (
            ("ViralZone", "https://viralzone.expasy.org/"),
            ("UniProt", "https://www.uniprot.org/uniprotkb?query=taxonomy_id%3A10239"),
            ("NCBI Virus", "https://www.ncbi.nlm.nih.gov/labs/virus/"),
            ("ICTV virus taxonomy", "https://ictv.global/"),
        ),
        "modules": (
            (None, "Infection rate", "Count virus-positive host cells per well.", "parasitaemia"),
            (None, "Plaque Assay", "Quantify plaque number and size.", "analyze_plaques"),
            (None, "Recruitment", "Measure host-protein enrichment at replication sites.", "recruitment"),
            (None, "Antiviral response", "Measure infection across antiviral concentrations.", "drug"),
            (None, "Cytopathic effect", "Measure host-cell rounding, detachment and loss.", "damage"),
            (None, "Viral spread", "Follow infection foci expanding over time.", "traversal"),
            (None, "Syncytium formation", "Detect fused multinucleated host cells.", "morphology"),
            (None, "Virus entry", "Separate bound from internalised virions.", "epithelial"),
        ),
        "workflows": {
            "parasitaemia": ("host_pathogen", {}, (
                "Opens Host–Pathogen Analysis. Measure all host cells and "
                "viral antigen or reporter signal as pathogen objects; its "
                "infected fraction per well is the infection rate.")),
            "analyze_plaques": ("analyze_plaques", {}, (
                "Opens the Plaque Assay. Segment plaques in the stained "
                "monolayer; it reports plaque number and size per well.")),
            "recruitment": ("recruitment", {}, (
                "Opens Recruitment. Use the viral replication compartment "
                "as the pathogen object; it reports host-protein enrichment "
                "around it.")),
            "drug": ("dose_response", {}, (
                "Opens Dose–Response. Fit a four-parameter logistic curve "
                "and EC50 to the infected fraction per well against "
                "antiviral concentration.")),
        },
    },
    "mammalian": {
        "name": "Mammalian cells",
        "description": (
            "Cultured mammalian cells, from primary cells to established "
            "lines, are the hosts in most infection assays and the subject "
            "of many screens on their own. Imaging can measure cell number, "
            "shape, migration, the cell cycle, organelle organization and "
            "responses to compounds, with no pathogen in the well."
        ),
        "source": "https://www.uniprot.org/help/subcellular_location",
        "source_label": "Biology source: UniProt",
        "diagram": "organism_mammalian.svg",
        "diagram_note": (
            "A schematic adherent mammalian cell spread on extracellular "
            "matrix, with its nucleus and nucleoli, endoplasmic reticulum, "
            "Golgi apparatus beside the centrosome and primary cilium, "
            "mitochondria and cytoskeleton. Hover for a UniProt location "
            "description and check several compartments to keep them "
            "highlighted. The labels are UniProt subcellular locations "
            "annotated for mammalian proteins. Cell shape and organelle "
            "arrangement vary between cell types."
        ),
        "sections": (
            ("Segmentation and morphology", (
                "Cell segmentation opens Mask to segment nuclei and cells in "
                "fixed or live images. Morphology profiling opens Measure "
                "to quantify per-cell intensity, texture and shape. The "
                "planned Organelle morphology module will measure "
                "mitochondrial and other organelle shapes. Choose stains "
                "that mark the compartments in the diagram you want to "
                "measure, and keep imaging settings fixed across a plate."
            ), ("mask", "measure")),
            ("Migration and wound closure", (
                "Cell migration opens the Motility Assay with infection QC "
                "off, because no pathogen is present; it reports track speed "
                "and straightness. The planned Wound closure module will "
                "measure gap closure in scratch assays. Record the frame "
                "interval and confluency, because crowding changes how "
                "fast and how straight cells move."
            ), ("gliding",)),
            ("Cell cycle, uptake and cytotoxicity", (
                "Cell-cycle staging is planned to classify G1, S, G2 and "
                "mitotic cells from DNA content and markers. Phagocytosis "
                "opens Host–Pathogen Analysis with beads or particles as the "
                "pathogen objects. Cytotoxicity opens Dose–Response to fit "
                "cell number or a viability readout against compound "
                "concentration. Separate internalised from surface-bound "
                "particles with a quenching or differential stain before "
                "reading uptake, and count cells in each well so that "
                "cytotoxicity and proliferation are not confused."
            ), ("phagocytosis", "drug")),
            ("Building an image-analysis workflow", (
                "Cell segmentation, Morphology profiling, Cell migration, "
                "Phagocytosis and Cytotoxicity open existing spaCR modules; "
                "the other three tiles are planned and marked Coming soon. "
                "UniProt and the Human Protein Atlas give subcellular "
                "locations for choosing markers. The diagram shows UniProt "
                "vocabulary, not where a selected protein is measured."
            ), ()),
        ),
        "links": (
            ("Human Protein Atlas", "https://www.proteinatlas.org/humanproteome/subcellular"),
            ("UniProt", "https://www.uniprot.org/uniprotkb?query=taxonomy_id%3A40674"),
            ("NCBI Taxonomy", "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=40674"),
            ("Cellosaurus", "https://www.cellosaurus.org/"),
        ),
        "modules": (
            (None, "Cell segmentation", "Segment nuclei and cells in fixed or live images.", "mask"),
            (None, "Morphology profiling", "Measure per-cell intensity and shape features.", "measure"),
            (None, "Cell migration", "Track cell speed and straightness over time.", "gliding"),
            (None, "Phagocytosis", "Quantify uptake of beads or particles.", "phagocytosis"),
            (None, "Cytotoxicity", "Measure cell number across compound concentrations.", "drug"),
            (None, "Cell-cycle staging", "Classify G1, S, G2 and mitotic cells.", "staging"),
            (None, "Organelle morphology", "Measure mitochondrial and organelle shape.", "filamentation"),
            (None, "Wound closure", "Measure gap closure in scratch assays.", "damage"),
        ),
        "workflows": {
            "mask": ("mask", {}, (
                "Opens Mask. Choose a Cellpose model for nuclei and cells; "
                "it writes masks for Measure and the other modules.")),
            "measure": ("measure", {}, (
                "Opens Measure. Point it at the images and masks; it "
                "writes per-cell intensity, texture and shape features.")),
            "gliding": ("motility", {"infection_intensity_qc_scope": "none"}, (
                "Opens the Motility Assay with infection QC off, because no "
                "pathogen is present. Segment the cells as the tracked "
                "objects; it reports track speed and straightness per well.")),
            "phagocytosis": ("host_pathogen", {}, (
                "Opens Host–Pathogen Analysis. Measure the cells as host "
                "cells and beads or particles as pathogen objects; it "
                "reports the fraction of cells with uptake and particles "
                "per cell.")),
            "drug": ("dose_response", {}, (
                "Opens Dose–Response. Fit a four-parameter logistic curve "
                "and EC50 to cell number or viability per well against "
                "compound concentration.")),
        },
    },
}
"""Organism guides keyed by ``toxoplasma``, ``plasmodium``, ``candida``,
``trypanosoma``, ``leishmania``, ``giardia``, ``virus`` and ``mammalian``.

Each record supplies a display name, description, biology source URL, bundled
diagram filename and diagram note; ``source_label`` names the biology source
when it is not the CDC. The last five pages are alpha features. ``sections`` contains heading, prose and
linked assay-key triples; ``links`` contains display-label and URL pairs.
``modules`` contains route-key, title, description and icon-key tuples. A
``None`` route denotes an organism-specific assay with no pipeline of its
own. ``workflows`` maps such a tile's icon key to the existing module that
measures its readout, a settings preset applied on opening, and a note on
how to use it; a ``None`` tile absent from ``workflows`` is Coming soon.
``starplast`` launches an external app from the Toxoplasma page; it is not
a spaCR analysis registry key or segmentation backend.
Display prose is translated at use; route keys, URLs and asset names stay fixed.
"""


def workflow(organism_key: str, icon: str):
    """Return the existing module route of one organism tile, if any.

    :param organism_key: an organism page key from :data:`ORGANISMS`.
    :param icon: the tile's icon key, unique within its organism.
    :returns: ``(app key, preset, note)`` or ``None`` for a Coming soon tile.
    """
    return ORGANISMS.get(organism_key, {}).get("workflows", {}).get(icon)
