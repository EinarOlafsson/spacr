from pathlib import Path
root=Path(__file__).resolve().parent
source=Path('spacr/qt/widgets/ambient.py').read_text()
(root/'before.py').write_text(source)
a=source.index('class _FungalGrowthEngine(');b=source.index('\nclass _ThoreEngine(',a)
part=source[a:b]
part=part.replace('        self._fungal_raster_failed = False\n','        self._fungal_raster_failed = False\n        self._fungal_strokes = {}\n',1)
part=part.replace('        self._fungal_observed.clear()\n','        self._fungal_observed.clear()\n        self._fungal_strokes.clear()\n',1)
start=part.index('    def _paint_cached_fungal_paths(');end=part.index('    def _paint_fungal_tips(',start)
method=part[start:end]
needle='            painter.drawPath(path)\n'
replacement='''            if stroke < 1.0:
                painter.drawPath(path)
            else:
                from PySide6.QtGui import QPainterPathStroker
                bounds = path.boundingRect()
                shape_key = (stroke, path.elementCount(), bounds.x(), bounds.y(),
                             bounds.width(), bounds.height())
                outline = self._fungal_strokes.get(shape_key)
                if outline is None or outline[0] != path:
                    stroker = QPainterPathStroker()
                    stroker.setWidth(stroke)
                    stroker.setCapStyle(Qt.RoundCap)
                    stroker.setJoinStyle(Qt.RoundJoin)
                    outline = (QPainterPath(path), stroker.createStroke(path))
                    self._fungal_strokes[shape_key] = outline
                    if len(self._fungal_strokes) > 64:
                        del self._fungal_strokes[next(iter(self._fungal_strokes))]
                painter.fillPath(outline[1], color)
'''
assert method.count(needle)==1;method=method.replace(needle,replacement)
part=part[:start]+method+part[end:]
(root/'candidate.py').write_text(source[:a]+part+source[b:])
