"""
Default settings for plotly
"""

import plotly
import plotly.graph_objects as go

def topx(pts: float) -> float:
    """Conversion from TeX points to pixels at 96 DPI"""
    return pts * 96 / 72.27

PT_WIDTH = 419.0
PT_NORMAL = 10.95
PT_FOOTNOTE = 9.0
PT_TINY = 7.0
PX_NORMAL = topx(PT_NORMAL)
PX_FOOTNOTE = topx(PT_FOOTNOTE)
PX_TINY = topx(PT_TINY)
PX_RULE = topx(0.4)

plotly.io.templates.default = go.layout.Template(layout={
    'width': topx(PT_WIDTH),
    'height': topx(PT_WIDTH * 0.75),
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
            'title.font.size': PX_NORMAL,
            'linewidth': PX_RULE,
            'gridwidth': PX_RULE,
            'tickwidth': PX_RULE,
            'tickfont.size': PX_FOOTNOTE,
        } for a in 'xy'
    },
    'xaxis.title.standoff': 10,
    'yaxis.title.standoff': 5,
    },
    data={
        'contour': [{'colorbar': {'exponentformat': 'power'}, 'opacity': 0.97}],
        'scatter': [{
            'line': {'width': 2*PX_RULE},
            'marker': {
                'symbol': 'cross-thin',
                'line.width': 2*PX_RULE,
                'line.color': plotly.colors.DEFAULT_PLOTLY_COLORS[i]
                }
        } for i in range(10)],
        'scattergl': [{
            'line': {'width': 3*PX_RULE},
        }],
        'histogram': [{
            'marker': {'line': {'width': PX_RULE, 'color': 'black'}}
            }],
        },
)
