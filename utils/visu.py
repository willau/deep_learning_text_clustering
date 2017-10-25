# -*- coding: utf-8 -*-

import os
import numpy as np
from bokeh.resources import INLINE
from bokeh.plotting import figure, show, save, ColumnDataSource
from bokeh.models import HoverTool, CrosshairTool, WheelZoomTool, ResetTool, RedoTool, BoxZoomTool
from bokeh import __base_version__ as bokeh_version
from utils.variable import COLORNAMES

CURDIR = os.getcwd()
DATE_FORMATS = dict(
    hours=["%a %d %b %y"],
    days=["%a %d %b %y"],
    months=["%b %y"],
    years=["%y"],
)

# DatetimeTickFormatter: formats property has been removed in 0.12.6
if bokeh_version >= '0.12.6':
    _date_formatter_kwargs = DATE_FORMATS
else:
    _date_formatter_kwargs = dict(
        formats=DATE_FORMATS
    )


def colorize(labels, random_state=None):
    labels_list = np.unique(labels)
    n_labels = len(labels_list)
    color_list = COLORNAMES[:n_labels]
    np.random.seed(random_state)
    np.random.shuffle(color_list)
    colormap = {label: color for label, color in zip(labels_list, color_list)}
    return [colormap[label] for label in labels]


def save_html(obj, title, output_path=None):
    if output_path is None:
        output_path = os.path.join(CURDIR, title + ".html")
    save(obj, filename=output_path, resources=INLINE, title='Bokeh Plot')
    print("Visualization saved in %s" % output_path)


def cluster2d(x, y, height=700, width=700, size=10, alpha=0.7,
              colors=None, title="t-sne plot", html_output=False, output_path=None):

    tools = [
        # HoverTool(),
        # CrosshairTool(),
        WheelZoomTool(),
        ResetTool(),
        RedoTool(),
        BoxZoomTool(),
    ]

    # source
    data = dict(x=x, y=y)
    if colors is not None:
        data['color'] = colors
    source = ColumnDataSource(data=data)

    # figure
    n = np.size(x)
    p = figure(title="%s - %d samples" % (title, n), tools=tools)
    p.plot_height = height
    p.plot_width = width
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None

    # sample plot
    circle_kwargs = dict(size=size, source=source, fill_alpha=alpha)
    if colors is not None:
        circle_kwargs['color'] = 'color'
    p.circle('x', 'y', **circle_kwargs)

    if html_output:
        save_html(p, title, output_path=output_path)
    show(p)

