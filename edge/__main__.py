"""Package entry point so ``python -m worker`` runs the edge worker CLI.

Kept intentionally minimal and unfenced per the Python ``__main__`` idiom
(https://docs.python.org/3/library/__main__.html). ``python -m worker.edge_worker``
remains valid for prod/scripts; this only adds the discoverable package form.
"""

from edge.runtime.edge_worker import main

raise SystemExit(main())
