"""Sourced organism introductions and image-analysis module proposals.

The assay keys are existing pipeline identifiers. A proposal has no route,
so the organism pages cannot accidentally run an unimplemented analysis.
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
                "enrichment around the parasite-containing vacuole. Used together, "
                "these modules help separate a change in entry from a change in "
                "the host response after entry. Choose image channels and marker "
                "definitions that make those two populations distinguishable."
            ), ("invasion", "recruitment")),
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
                "The planned Egress module will focus on vacuole rupture and "
                "parasite exit; Gliding motility will quantify extracellular "
                "trails and movement. Bradyzoite conversion will address "
                "stage-marker and cyst-wall readouts, while Host cell damage "
                "will describe monolayer loss. These are proposed workflows, "
                "so their tiles are marked Coming soon. Fixed images, time "
                "series and stage-specific markers answer different questions "
                "and should be selected for the intended readout."
            ), ()),
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
            ('host_pathogen', 'Host–Pathogen Analysis', 'Combine vacuole marker recruitment, parasite counts and host infection denominators.', 'recruitment'),
            ("invasion", "Invasion Assay", "Distinguish attached and invaded parasites.", "invasion"),
            ("replication", "Replication Assay", "Count parasites per vacuole.", "replication"),
            (None, "Egress", "Follow vacuole rupture and parasite exit over time.", "egress"),
            (None, "Gliding motility", "Measure extracellular parasite trails and speed.", "gliding"),
            (None, "Bradyzoite conversion", "Quantify cyst-wall staining and stage conversion.", "cyst"),
            (None, "Host cell damage", "Measure monolayer integrity and host-cell loss.", "damage"),
        ),
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
                "The proposed Parasitaemia module will count infected red "
                "blood cells relative to the total red-cell population. "
                "Blood-stage staging will separate rings, trophozoites and "
                "schizonts so a shift in stage composition can be examined "
                "alongside infection frequency. Merozoite invasion will focus "
                "on entry into red blood cells. Together these readouts would "
                "help distinguish fewer newly infected cells from altered "
                "development of parasites already inside them."
            ), ()),
            ("Liver infection and parasite movement", (
                "Liver-stage growth is planned to measure parasite number "
                "and size within hepatocytes. Sporozoite motility will focus "
                "on gliding trajectories or trails, while Cell traversal "
                "will address host-cell wounding along parasite paths. "
                "Traversal, productive invasion and subsequent growth are "
                "different outcomes: their image markers, observation times "
                "and denominators need to be defined separately."
            ), ()),
            ("Transmission stages and compound responses", (
                "Gametocyte maturity is a proposed workflow for stage and "
                "sex-associated image features when suitable markers and "
                "reference annotations are available. Drug response imaging "
                "will relate growth and morphology to concentration. Record "
                "the species, starting stage and exposure duration with "
                "these measurements: changes in the mix of stages can "
                "otherwise be mistaken for changes in parasite number."
            ), ()),
            ("Building an image-analysis workflow", (
                "All eight Plasmodium modules on this page are planned. "
                "Their descriptions define intended readouts rather than "
                "available analysis pipelines. PlasmoDB and UniProt provide "
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
                "The proposed Adhesion module will count fungal cells "
                "associated with a host-cell surface. Biofilm will focus "
                "on collective growth, including covered area and thickness "
                "when the acquisition contains depth information. A single "
                "two-dimensional field can describe surface coverage but "
                "cannot by itself establish biofilm thickness. Choose "
                "single-plane or volumetric imaging to match the endpoint."
            ), ()),
            ("Interactions with host cells", (
                "Epithelial invasion is planned to distinguish extracellular "
                "fungi from those internalised by epithelial cells. "
                "Phagocytosis will measure uptake by phagocytes. Both "
                "require a way to resolve contact from internalisation, "
                "such as appropriate differential labelling or spatial "
                "information. Report the host-cell population and fungal "
                "morphology together with the uptake readout."
            ), ()),
            ("Antifungal response and interpretation", (
                "Antifungal response is a proposed concentration-response "
                "workflow for fungal growth and morphology. Combining it "
                "with Morphology or Biofilm would help describe which "
                "image features change during treatment. All eight Candida "
                "modules are currently planned and their tiles are marked "
                "Coming soon. The Candida Genome Database and UniProt "
                "provide gene and protein annotations to support marker "
                "selection and interpretation."
            ), ()),
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
    },
}
"""Organism guides keyed by ``toxoplasma``, ``plasmodium`` and ``candida``.

Each record supplies a display name, description, biology source URL, bundled
diagram filename and diagram note. ``sections`` contains heading, prose and
linked assay-key triples; ``links`` contains display-label and URL pairs.
``modules`` contains route-key, title, description and icon-key tuples. A
``None`` route denotes a planned assay whose tile cannot start an analysis.
``starplast`` launches an external app from the Toxoplasma page; it is not
a spaCR analysis registry key or segmentation backend.
Display prose is translated at use; route keys, URLs and asset names stay fixed.
"""
