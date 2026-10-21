` --dir ` chooses the checkout path; ` --branch ` selects a branch or tag; ` --no-install ` fetches without installing; ` --help ` lists the options. ` packaging/source_install_excludes.txt ` lists every skipped path.

The full nightly clone downloaded 9.25 GiB. Adding ` --filter=blob:none ` to the shallow clone does not reduce the checkout: its object store is still 2032 MB. Git does not report lazy blob fetches, so no download total was measured. The main tracked tree is a 5059 MB checkout (measured 2026-10-21), excluding Git history. Download sizes and times vary with the branch.
