"""Loads pieces of the vendored Google Product Search tool (see
google_product_search/, an unmodified copy of the standalone repo) in
isolation, so its code can be reused from this app without ever touching
the copied files themselves.

Why isolation is needed at all: that tool's own code imports bare
top-level names — `models`, `search`, `tool`, `scraper`, `config` — and
relies on its own src/ directory being on sys.path to resolve them (see
google_product_search/src/main.py). This repo already has its own
top-level `models.py`, so loading both under the same import namespace
unmodified would collide: whichever one gets cached into sys.modules
first wins, and the loser either shadows or outright breaks depending on
load order. Since the copied tool must stay byte-for-byte unchanged, the
isolation happens here instead — every load below temporarily swaps out
sys.modules/sys.path just long enough to exec one file, then restores
everything exactly as it was. The functions/classes pulled out keep
working fine afterward as ordinary Python objects — removing their
origin module's name from sys.modules doesn't unbind anything already
created and referenced elsewhere.
"""
import importlib.util
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "google_product_search", "src")
_COLLIDING_NAMES = ("models", "search", "tool", "scraper", "config")


def _load_isolated(rel_path: str, module_name: str):
    """Execs one file from the vendored tool's src/ tree with src/ on
    sys.path just for the duration of the load, then restores this repo's
    own sys.modules state for every name that could collide."""
    file_path = os.path.join(_SRC_DIR, rel_path)
    stashed = {name: sys.modules.pop(name, None) for name in _COLLIDING_NAMES}
    sys.path.insert(0, _SRC_DIR)
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(_SRC_DIR)
        for name in list(sys.modules):
            if name in _COLLIDING_NAMES or any(name.startswith(f"{n}.") for n in _COLLIDING_NAMES):
                del sys.modules[name]
        for name, mod in stashed.items():
            if mod is not None:
                sys.modules[name] = mod


def load_google_product_search_tool():
    """Returns (google_product_search, get_tool_schema, SearchResponse,
    BatchSearchResponse) straight out of the tool's own
    tool/google_product_search.py — for wiring up as real routes on this
    app's own router (routers/google_product_search.py) so they appear on
    this app's main Swagger docs, instead of living behind a separate
    mounted sub-app's own docs page."""
    module = _load_isolated(os.path.join("tool", "google_product_search.py"), "_google_product_search_tool")
    return module.google_product_search, module.get_tool_schema, module.SearchResponse, module.BatchSearchResponse
