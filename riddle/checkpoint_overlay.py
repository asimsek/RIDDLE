import pickle
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection, PolyCollection, QuadMesh
from matplotlib.colors import to_rgb
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import StepPatch
from matplotlib.quiver import Quiver
from matplotlib.transforms import IdentityTransform


class CheckpointOverlay:
    def __init__(self, methods, colors, labels, *, last_epoch=None):
        self.methods = {labels[key] for key in methods}
        self.colors = {tuple(to_rgb(colors[key])) for key in methods}
        self.curves = {}
        self.surfaces = set()
        self.tables = {}
        self.scan_points = {}
        self.last_epoch = last_epoch
        self.last = True
        self.spool = None
        self.buffered = 0

    def take(self, key):
        path = self.curves.pop(key)
        if isinstance(path, tuple):
            factory, legend = path
            return (*factory(), legend)
        if isinstance(path, bytes):
            self.buffered -= len(path)
            return pickle.loads(path)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        path.unlink()
        return payload

    def table_rows(self, key, rows):
        if self.last:
            if rows:
                self.tables[key] = rows
            return []
        last = self.tables.pop(key, [])
        remaining = {(row["Method"], row["N_inj"]): row for row in last}
        combined = []
        for row in rows:
            other = remaining.pop((row["Method"], row["N_inj"]), None)
            if other is not None:
                combined.append((other, "last"))
            policy = "lowest-loss" if row["Method"] in self.methods else "native"
            combined.append((row, policy))
        combined.extend((row, "last") for row in remaining.values())
        result = [{**row, "Checkpoint policy": policy} for row, policy in combined]
        if self.last_epoch is not None:
            for row in result:
                row["Last epoch"] = self.last_epoch if row["Checkpoint policy"] == "last" else None
        return result

    @staticmethod
    def _artists(ax, stem, methods, colors):
        single_method = any(part in methods for part in stem.parts)
        for artist in [*ax.lines, *ax.patches, *ax.collections]:
            if artist.get_gid() == "signal_region_boundary" or artist.get_transform() != ax.transData:
                continue
            if isinstance(artist, (Line2D, LineCollection)):
                color = artist.get_color()
            elif isinstance(artist, StepPatch):
                color = artist.get_edgecolor()
            elif isinstance(artist, PolyCollection):
                color = artist.get_facecolor()
            else:
                continue
            if isinstance(color, np.ndarray) and color.ndim == 2:
                if not len(color):
                    continue
                color = color[0]
            label = artist.get_label() or ""
            if single_method or label in methods or tuple(to_rgb(color)) in colors:
                yield artist, color

    @staticmethod
    def _clone(artist):
        if isinstance(artist, StepPatch):
            data = artist.get_data()
            result = StepPatch(data.values.copy(), data.edges.copy(), baseline=data.baseline,
                               orientation=artist.orientation)
        elif isinstance(artist, Line2D):
            result = Line2D(*artist.get_data())
        elif isinstance(artist, LineCollection):
            result = LineCollection(artist.get_segments())
        else:
            result = PolyCollection([])
            result.set_verts_and_codes([p.vertices for p in artist.get_paths()], [p.codes for p in artist.get_paths()])
        result.update_from(artist)
        result.set_transform(IdentityTransform())
        if isinstance(result, (LineCollection, PolyCollection)):
            result.set_offset_transform(IdentityTransform())
        result.set_label("_checkpoint_overlay")
        result.set_clip_path(None)
        result.set_clip_box(None)
        return result

    @staticmethod
    def _fade(artist, color):
        light = 0.75 * np.asarray(to_rgb(color)) + 0.25
        if isinstance(artist, PolyCollection):
            artist.set_facecolor((*light, 0.09))
            artist.set_edgecolor((*light, 0.38))
            artist.set_alpha(None)
            artist.set_hatch("/")
            artist.set_linewidth(0)
        else:
            if isinstance(artist, StepPatch):
                artist.set_edgecolor(light)
            else:
                artist.set_color(light)
            artist.set_linestyle("--")
            artist.set_alpha(0.78)
        artist.set_zorder(artist.get_zorder() - 0.15)

    def compose(self, fig, ax, stem, root, legend=None):
        key = stem.relative_to(root)
        if "02_training" in key.parts:
            return None if self.last else stem
        surface = any(isinstance(collection, (PathCollection, QuadMesh, Quiver))
                      for axis in fig.axes for collection in axis.collections)
        if surface or any(axis.images for axis in fig.axes):
            if self.last:
                self.surfaces.add(key)
                return stem
            return None if key in self.surfaces or "05_stein_witness" in key.parts else stem
        if self.last:
            if any(next(self._artists(axis, key, self.methods, self.colors), None) is not None for axis in fig.axes):
                payload = pickle.dumps((fig, ax, legend), protocol=pickle.HIGHEST_PROTOCOL)
                if self.buffered + len(payload) <= 128 * 1024**2:
                    self.curves[key] = payload
                    self.buffered += len(payload)
                else:
                    if self.spool is None:
                        self.spool = tempfile.TemporaryDirectory(prefix="riddle-checkpoint-figures-")
                    path = Path(self.spool.name) / f"{len(self.curves)}.pickle"
                    path.write_bytes(payload)
                    self.curves[key] = path
            return None
        other = self.take(key)[0] if key in self.curves else None
        self.compose_curves(fig, key, self.methods, self.colors, other)
        return stem

    @classmethod
    def compose_curves(cls, fig, key, methods, colors, other):
        additions = []
        limits = []
        if other is not None:
            additions = [[cls._clone(artist) for artist, _ in cls._artists(axis, key, methods, colors)] for axis in other.axes]
            limits = [axis.get_ylim() for axis in other.axes]
            plt.close(other)
        for i, axis in enumerate(fig.axes):
            artists = list(cls._artists(axis, key, methods, colors))
            handles, labels = axis.get_legend_handles_labels()
            native_artists = {artist for artist, _ in artists}
            original = {}
            for handle, label in zip(handles, labels):
                if handle in native_artists:
                    original[label] = cls._clone(handle)
                elif isinstance(handle, ErrorbarContainer):
                    line, caps, bars = handle.lines
                    if any(a in native_artists for a in (line, *caps, *bars)):
                        elements = (cls._clone(line) if line is not None else None,
                                    tuple(cls._clone(a) for a in caps), tuple(cls._clone(a) for a in bars))
                        original[label] = ErrorbarContainer(elements, has_xerr=handle.has_xerr, has_yerr=handle.has_yerr)
            if original:
                axis._checkpoint_legend_handles = original
            for artist, color in artists:
                cls._fade(artist, color)
            for artist in additions[i] if i < len(additions) else ():
                artist.set_transform(axis.transData)
                if isinstance(artist, Line2D):
                    axis.add_line(artist)
                elif isinstance(artist, StepPatch):
                    axis.add_patch(artist)
                else:
                    axis.add_collection(artist)
            if i < len(additions) and additions[i]:
                axis.autoscale_view()
                if not axis.get_autoscaley_on():
                    current = axis.get_ylim()
                    bounds = sorted((*current, *limits[i]))
                    axis.set_ylim((bounds[-1], bounds[0]) if current[0] > current[1] else (bounds[0], bounds[-1]))
