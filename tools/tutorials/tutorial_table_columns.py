"""Use actual draggable headers to make recorded table columns legible."""


def resize_visible_columns(table, settle):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QHeaderView

    header=table.horizontalHeader(); viewport=header.viewport()
    font=header.fontMetrics(); cell_font=table.fontMetrics(); changes=[]
    for column in range(table.columnCount()):
        if header.sectionResizeMode(column)!=QHeaderView.Interactive:
            continue
        start=header.sectionViewportPosition(column)
        old=header.sectionSize(column)
        edge=start+old-1
        if start<0 or edge>=viewport.width()-3:
            break
        label=table.horizontalHeaderItem(column).text()
        content=max((cell_font.horizontalAdvance(table.item(row,column).text())
            for row in range(table.rowCount()) if table.item(row,column)),default=0)
        wanted=min(750,max(95,font.horizontalAdvance(label)+36,content+28))
        wanted=min(wanted,viewport.width()-start-6)
        if wanted<old or abs(wanted-old)<3:
            continue
        y=viewport.height()//2
        source=QPoint(edge,y); target=QPoint(start+wanted-1,y)
        QTest.mouseMove(viewport,source)
        QTest.mousePress(viewport,Qt.LeftButton,pos=source)
        QTest.mouseMove(viewport,target,delay=50)
        QTest.mouseRelease(viewport,Qt.LeftButton,pos=target)
        settle(.1)
        actual=header.sectionSize(column)
        if abs(actual-wanted)>3:
            raise ValueError(f'Native header drag failed for {label}: {old} -> {actual}, expected {wanted}')
        changes.append(dict(column=label,before=old,after=actual))
    return changes
