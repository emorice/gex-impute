"""
Default settings for plotly
"""

import plotly
import plotly.graph_objects as go

def topx(pts):
    return pts * 96 / 72.27

pt_width = 419.0
pt_normal = 10.95
pt_footnote = 9.0
pt_tiny = 7.0
px_normal = topx(pt_normal)
px_footnote = topx(pt_footnote)
px_tiny = topx(pt_tiny)
px_rule = topx(0.4)

plotly.io.templates.default = go.layout.Template(layout={
    'width': topx(pt_width),
    'height': topx(pt_width * 0.75),
    'autosize': False,
    'font_family': 'Palatino',
    'margin': {'r': 0, 'b': 45, 't': 0},
    ** {
        f'{a}axis': {
            'showline': True,
            'ticks': 'outside',
            'constrain': 'domain',
            'exponentformat': 'power',
            'minexponent': 4,
            'color': 'black',
            'title.font.size': px_normal,
            'linewidth': px_rule,
            'gridwidth': px_rule,
            'tickwidth': px_rule,
            'tickfont.size': px_footnote,
        } for a in 'xy'
    },
    'xaxis.title.standoff': 0,
    'yaxis.title.standoff': 5,
    },
    data={
        'contour': [{'colorbar': {'exponentformat': 'power'}, 'opacity': 0.97}],
        'scatter': [{
            'line': {'width': 2*px_rule},
            'marker': {'symbol': 'cross-thin', 'line.width': 2*px_rule, 'line.color': plotly.colors.DEFAULT_PLOTLY_COLORS[i]}
        } for i in range(10)],
        },
)
