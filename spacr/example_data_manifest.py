"""Manifest for the downloadable example-screen data files.

Each entry records an asset's role, plate, expected byte size, and SHA-256
digest. Download validation uses these values to identify incomplete or
corrupt files before they are opened as tables.
"""

#: Metadata used to download and validate each example-screen file.
FILES = [
    {
        "name": "plate_1_unique_combinations.csv",
        "kind": "counts",
        "plate": 1,
        "bytes": 4325565,
        "sha256": "9bbfd6de409ec501b94abd2f20230611a695b142228892f5c3b8b2b564ba1f13"
    },
    {
        "name": "plate1_dv.csv",
        "kind": "scores",
        "plate": 1,
        "bytes": 5088290,
        "sha256": "06aa6e51e17eb2423f9e534deb984533b98d10945dd4a7f9a8bac2f93d088594"
    },
    {
        "name": "plate_2_unique_combinations.csv",
        "kind": "counts",
        "plate": 2,
        "bytes": 4186654,
        "sha256": "88142a5b8df993f881748845d15f2290e662cccddde269cbeacea8f28b119bf8"
    },
    {
        "name": "plate2_dv.csv",
        "kind": "scores",
        "plate": 2,
        "bytes": 5003197,
        "sha256": "6f079cad146f0d507524317fa026235a2f76a7be04da4be396cec3628c1d1fef"
    },
    {
        "name": "plate_3_unique_combinations.csv",
        "kind": "counts",
        "plate": 3,
        "bytes": 3577071,
        "sha256": "3fdb0c1d28f6074211775e72a93ce5a2531f4049093578251a8ee36bbc85765e"
    },
    {
        "name": "plate3_dv.csv",
        "kind": "scores",
        "plate": 3,
        "bytes": 4824810,
        "sha256": "5cd7b59d55504680272b47060c7056dd76eb85e59d3571bbe8974e62da08d369"
    },
    {
        "name": "plate_4_unique_combinations.csv",
        "kind": "counts",
        "plate": 4,
        "bytes": 3983853,
        "sha256": "7888648e1e45155e6ef2efe44bfe02b7577a8469ca75c51a7abc6b4981abad34"
    },
    {
        "name": "plate4_dv.csv",
        "kind": "scores",
        "plate": 4,
        "bytes": 4240228,
        "sha256": "7d5800071c15098d9f3fe02a84580d884db7a145b06e72bf5fe048286d23d07e"
    }
]
