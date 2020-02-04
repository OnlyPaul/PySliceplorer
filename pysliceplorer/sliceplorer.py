from bokeh.io import output_file
from bokeh.layouts import column, row
from bokeh.models import HoverTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Select
from bokeh.plotting import figure, ColumnDataSource, save

import json
import numpy as np
import sobol_seq


class Slice1D:
    def __init__(self, plot, dim, x_grid, n_fpoint):
        self.__plot = plot
        self.x_grid = x_grid
        self.dim = dim
        self.size = n_fpoint

    def data(self, i):
        if i > self.size-1:
            raise Exception('Trying to get non-exist data. Axis number should not exceed {}'.format(self.size - 1))
        elif i < 0:
            raise Exception('Cannot achieve plot data at negative index')

        return self.__plot[i]

    def to_json(self):
        json_dict = {
            'x_grid': self.x_grid.tolist(),
            'size': self.size,
            'dim': self.dim,
            'entries': self.__plot
        }
        return json.dumps(json_dict)


def generate_sample_point(mn, mx, dim, n_fpoint, method='sobol'):
    # in order to provide possibility of having other method in the future
    if method == 'sobol':
        sobol_points = sobol_seq.i4_sobol_generate(dim, n_fpoint)
        return np.interp(sobol_points, [0, 1], [mn, mx])
    else:
        pass


# sliceplorer(function_spec, n_focus_point), which has function_spec spreading to
# f: function defined from outer program
# mn, mx: min and max range of computation
# dim: number of dimension of the function, f
# sampling method now only supports sobol sequences
def sliceplorer_core(f, mn, mx, dim: int, n_fpoint: int, n_seg: int=100, method='sobol'):
    if mx <= mn:
        raise Exception('Input min exceeds max value. (Error: min >= max)')

    if n_fpoint <= 0:
        raise Exception('Program requires at least 1 focus point.')

    if dim < 1:
        raise Exception('Sliceplorer does not support less than 1 dimension. (Error: dim < 1)')

    if n_seg <= 0:
        raise Exception('Number of linear space must be positive integer.')

    sample_points = generate_sample_point(mn, mx, dim, n_fpoint, method=method)
    f_vec = np.vectorize(f)
    x = np.linspace(mn, mx, n_seg)

    result = []
    for point in sample_points:
        data = []
        for i in range(0, dim):
            # create an argument list from point while having the i-th argument replaced
            # with the array x, the array acts as free variable of our 1D slice
            parg = []
            parg += point[0:i].tolist()
            parg.append(x)
            parg += point[i + 1:].tolist()
            v = f_vec(*parg)
            data.append(v.tolist())
        result.append({
            'point': point.tolist(),
            'data': data
        })

    return Slice1D(result, dim, x, n_fpoint)


def sliceplorer(f, mn, mx, dim, n_fpoint, output=None, n_seg=100, method='sobol', width=-1, height=-1, title=None):
    calc_data = sliceplorer_core(f, mn, mx, dim, n_fpoint, n_seg, method)

    source = []
    for i in range(0, calc_data.size):
        data = {
            'x': calc_data.x_grid,
            'fp': [calc_data.data(i)['point']] * len(calc_data.x_grid)
        }
        for j in range(0, calc_data.dim):
            var_name = 'y' + str(j)
            data[var_name] = calc_data.data(i)['data'][j]
        source.append(ColumnDataSource(data=data))

    tooltips = [
        ("(x,y)", "($x, $y)"),
        ("focus point", "@fp")
    ]
    hover = HoverTool(tooltips=tooltips)
    trace = [None] * calc_data.dim
    for j in range(0, calc_data.size):
        for i in range(0, calc_data.dim):
            if not trace[i]:
                if i == 0:
                    trace[i] = figure(
                        tools="wheel_zoom, pan",
                        title="x" + str(i + 1),
                        x_range=(mn, mx),
                    )
                else:
                    trace[i] = figure(
                        tools="wheel_zoom, pan",
                        title="x" + str(i + 1),
                        x_range=trace[0].x_range,
                        y_range=trace[0].y_range
                    )
                trace[i].add_tools(hover)

            trace[i].line(
                'x',
                'y' + str(i),
                source=source[j],
                color="black",
                alpha=0.1,
                hover_color="firebrick",
                hover_alpha=1,
                name=str(i) + str(j)
            )

    data = {
        'x': [],
        'fp': []
    }
    for i in range(0, calc_data.dim):
        var_name = 'y' + str(i)
        data[var_name] = []
    reset_source = ColumnDataSource(data=data)
    hidden_source = ColumnDataSource(data=data)

    for i in range(0, calc_data.dim):
        trace[i].line(
            'x',
            'y' + str(i),
            source=hidden_source,
            color='firebrick',
            alpha=1,
            line_width=2
        )

    callback = CustomJS(args=dict(src=source, hsrc=hidden_source, resrc=reset_source), code="""
        var sel_index = parseInt(cb_obj.value);
        var data;

        if (sel_index < 0)
            data = resrc.data;
        else
            data = src[sel_index].data;

        hsrc.data = data
    """)

    menu = [("-1", 'None')]
    for i in range(0, calc_data.size):
        menu.append((str(i), str(calc_data.data(i)['point'])))

    select = Select(title="Select Focus Point:", value="-1", options=menu)
    select.js_on_change('value', callback)

    if width > 0:
        for t in trace:
            t.plot_width = width

    if height > 0:
        for t in trace:
            t.plot_height = height

    col = column(trace)
    if output:
        output_file(output)
    save(row(col, select), title=title if title else 'Sliceplorer')
