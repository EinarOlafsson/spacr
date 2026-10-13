

def test_every_organism_the_instruction_names_resolves():
    """The list decided for instruction 239 item 7, asserted one by one.

    Hosts people image, every studied apicomplexan, and the other
    parasites spaCR is pointed at. A name that does not resolve leaves a
    screen unannotated with no way for the user to tell why, so the list
    is checked rather than trusted -- three of these (Sarcocystis,
    Cyclospora, Cystoisospora) were missing when it was.
    """
    from spacr import uniprot

    named = (
        # Hosts people image.
        "human", "mouse", "rat", "hamster", "rhesus macaque", "bovine",
        "porcine", "canine", "chicken", "zebrafish", "Drosophila",
        "C. elegans", "Xenopus",
        # Apicomplexa.
        "Toxoplasma gondii", "Plasmodium falciparum", "Plasmodium vivax",
        "Neospora caninum", "Eimeria", "Cryptosporidium", "Babesia",
        "Theileria", "Sarcocystis", "Cyclospora", "Cystoisospora",
        # Other parasites.
        "Trypanosoma brucei", "Trypanosoma cruzi", "Leishmania", "Giardia",
        "Entamoeba", "Trichomonas", "Schistosoma",
    )
    unresolved = [name for name in named
                  if uniprot.resolve(name).kind == "unknown"]
    assert not unresolved, f"these do not resolve: {unresolved}"


def test_toxoplasma_still_needs_no_network():
    """The default stays the offline path, whatever the field accepts."""
    from spacr import uniprot

    assert uniprot.resolve("toxoplasma").kind == "bundled"
    assert uniprot.resolve("").kind == "bundled"
