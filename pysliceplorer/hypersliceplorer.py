from bokeh.io import output_file
from bokeh.layouts import gridplot, row, widgetbox
from bokeh.models import HoverTool, Title
from bokeh.plotting import figure, ColumnDataSource, save
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Toggle, MultiSelect
from cffi import FFI
from rpy2 import robjects

import json
import numpy as np
import os
import platform
import sobol_seq


class HSlice2D:
    def __init__(self, result, vertices, config, n_fpoint):
        self.dim = len(vertices[0])
        if self.dim < 3:
            raise Exception('Hypersliceplorer does not support less than 3 dimensions. (Error: dim < 3)')
        self.__result = result
        self.vertices = vertices
        self.config = config
        self.size = n_fpoint

    def data_by_axes(self, d1, d2):
        return self.__result[(d1, d2)]

    def data_by_point(self, fpoint):
        sort_by_point = {}
        for axes_key in self.__result:
            sort_by_point[axes_key] = self.__result[axes_key][fpoint]
        return sort_by_point

    def to_json(self):
        json_dict = {
            'vertices': self.vertices,
            'config': self.config,
            'size': self.size,
            'entries': self.__result
        }
        return json.dumps(json_dict)


def transpose(mat, nc, nr):
    ret = [None]*len(mat)
    for row in range(0, nc):
        for col in range(0, nr):
            ret[nr*row + col] = mat[nc*col + row]
    return ret


def generate_simplices(vertices, config):
    ret = []
    for row in config:
        simp = []
        for col in row:
            simp = simp + vertices[col]
        # The simplex needs to be transposed due to different data structure in Rust core.
        # The code here delivers correct aligned matrix in Rust
        ret.append(transpose(simp, len(vertices[0]), len(config[0])))
    return ret


def generate_simplices_nt(vertices, config):
    ret = []
    for row in config:
        simp = []
        for col in row:
            simp = simp + vertices[col]
        # This is no-transpose version for R script.
        # As R-version seems to work fine. We will stick to this function.
        ret.append(simp)
    return ret


def generate_sample_point(mn, mx, dim, n_fpoint, method='sobol'):
    # in order to provide possibility of having other method in the future
    if method == 'sobol':
        sobol_points = sobol_seq.i4_sobol_generate(dim, n_fpoint)
        return np.interp(sobol_points, [0, 1], [mn, mx])
    else:
        pass


def load_spi():
    # this should handle OS-difference in loading Rust (normally the file extension)
    # "rslice2d" is the name set from Cargo.toml
    # This version renders the extension of the library based on platform.system()
    # which might cause some error due to that we possibly don't cover everything.
    # This could be update if any sources provided later on.
    # For now, we follow https://doc.rust-lang.org/reference/linkage.html
    if platform.system() == "Windows":
        extension = ".dll"
    elif platform.system() == "Linux":
        extension = ".so"
    elif platform.system() == "Darwin":
        extension = ".dylib"
    else:
        raise Exception("Unknown operating system is not supported.")

    # This is not so beautiful, however, as python scripts are mostly put somewhere else at runtime,
    # an absolute path is necessary. It works for now, but it is fragile to some circumstances,
    # for instance, when rust library is under different structure by any mean
    dll_path = os.path.dirname(os.path.abspath(__file__)) + "/rust_spi/target/release/rslice2d" + extension

    ffi = FFI()
    ffi.cdef("""
        typedef struct { double p1_1; double p1_2; double p2_1; double p2_2; } SliceSeg;
        SliceSeg* r_spi(const double*, uintptr_t, uintptr_t, const double*, uintptr_t, uintptr_t, uintptr_t);
        SliceSeg py_spi(const double*, uintptr_t, uintptr_t, const double*, uintptr_t, uintptr_t, uintptr_t);
        """)
    return ffi.dlopen(dll_path)


def load_r_spi():
    # load R source and return simplex.point.intersection function
    r_source = robjects.r['source']
    r_source(os.path.dirname(os.path.abspath(__file__)) + "/r_spi/slice2d.R")
    return robjects.globalenv['simplex.point.intersection']


def hypersliceplorer_core(vertices, config, mn, mx, n_fpoint, method='sobol'):
    # nc and nr represent number of rows and columns of a single simplex respectively
    nc = dim = len(vertices[0])
    nr = len(config[0])

    if mx <= mn:
        raise Exception('Input min exceeds max value. (Error: min >= max)')

    if n_fpoint <= 0:
        raise Exception('Program requires at least 1 focus point.')

    if dim < 3:
        raise Exception('Hypersliceplorer does not support less than 3 dimensions. (Error: dim < 3)')

    sample_points = generate_sample_point(mn, mx, dim, n_fpoint, method=method)
    simplices = generate_simplices_nt(vertices, config)

    # ready r_spi from R script
    r_spi = load_r_spi()

    # Organize the loops (Alg. 1 from Hypersliceplorer paper)
    result = {}
    for d1 in range(0, dim-1):
        for d2 in range(d1, dim):
            result[(d1, d2)] = {}
            # for each points, invoke lib.py_spi() for simplices
            for point in sample_points:
                fp = robjects.FloatVector(point.tolist())
                result[(d1, d2)][tuple(point)] = []
                for simplex in simplices:
                    r_simplex = robjects.r.matrix(robjects.FloatVector(simplex), nrow=nr, byrow=True)
                    slice_seg = r_spi(r_simplex, fp, d1+1, d2+1)
                    if slice_seg.rx2('p1_1') != robjects.rinterface.NULL:
                        # we found intersection, add the result to the archive
                        result[(d1, d2)][tuple(point)].append({
                            'p1_1': slice_seg.rx2('p1_1')[0],
                            'p1_2': slice_seg.rx2('p1_2')[0],
                            'p2_1': slice_seg.rx2('p2_1')[0],
                            'p2_2': slice_seg.rx2('p2_2')[0]
                        })
    return HSlice2D(result, vertices, config, n_fpoint)


def hypersliceplorer(vertices, config, mn, mx, n_fpoint, method='sobol', width=-1, height=-1, output=None, title=None):
    calc_data = hypersliceplorer_core(vertices, config, mn, mx, n_fpoint, method=method)

    # These will be the tooltips shown in our figures while hovering on the glyphs
    tooltips = [
        ("(x,y)", "($x, $y)"),
        ("focus point", "@fp")
    ]
    circle_tooltips = [
        ("(x,y)", "(@x, @y)")
    ]

    # every dictionaries here have their string key of (i, j) referring to d1, d2
    # meaning each trace, source, ray are assigned according to the dimensional pair.
    # So, we can plot only (d1, d2) = (0, 0) by plotting trace(0, 0).
    #   - source(i, j) will be an array containing the data from each fp
    #   - ray(i, j) is used in callback, so we can access the segments via CustomJS
    source = {}
    trace = {}
    ray = {}
    circle = {}
    colors = [
        '#e6194B', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4',
        '#4363d8', '#911eb4', '#f032e6', '#000075', '#e6beff', '#800000'
    ]
    for i in range(0, calc_data.dim - 1):
        for j in range(i, calc_data.dim):
            # create a figure at (d1, d2) = (i, j)
            if (i, j) == (0, 0):
                trace[str((i, j))] = figure(
                    tools="wheel_zoom",
                    x_range=(mn, mx),
                    y_range=(mn, mx)
                )
            else:
                trace[str((i, j))] = figure(
                    tools="wheel_zoom",
                    x_range=trace[str((0, 0))].x_range,
                    y_range=trace[str((0, 0))].y_range
                )

            # each source(i, j) contains data at focus points, we create the sources as variable
            # just in case that we might need them later
            source[str((i, j))] = []
            ray[str((i, j))] = []
            circle[str((i, j))] = []
            for point in calc_data.data_by_axes(i, j):
                data = {
                    'fp': [],
                    'x0': [],
                    'y0': [],
                    'x1': [],
                    'y1': []
                }
                for seg in calc_data.data_by_axes(i, j)[point]:
                    data['fp'].append(str(point))
                    data['x0'].append(seg['p1_1'])
                    data['y0'].append(seg['p1_2'])
                    data['x1'].append(seg['p2_1'])
                    data['y1'].append(seg['p2_2'])
                source[str((i, j))].append(ColumnDataSource(data=data))
                circle[str((i, j))].append(trace[str((i, j))].circle(
                    x=[point[i]],
                    y=[point[j]],
                    size=6,
                    color="black",
                    alpha=0.1,
                    visible=False
                ))
                circle_hover = HoverTool(tooltips=circle_tooltips, renderers=[circle[str((i, j))][-1]])
                trace[str((i, j))].add_tools(circle_hover)

            # now plot those sources in the figure
            for s in source[str((i, j))]:
                ray[str((i, j))].append(trace[str((i, j))].segment(
                    x0='x0',
                    x1='x1',
                    y0='y0',
                    y1='y1',
                    source=s,
                    line_color="black",
                    line_alpha=0.1,
                    line_width=2
                ))
                hover = HoverTool(tooltips=tooltips, renderers=[ray[str((i, j))][-1]])
                trace[str((i, j))].add_tools(hover)

    # set up a tool to toggle visibility
    multi_menu = []
    for point in calc_data.data_by_axes(0, 0):
        multi_menu.append((str(point), str(point)))
    sorted_menu = sorted(multi_menu)
    # menu index is irrelevant, but just in case that we want the color to
    # be faithful to the indices of the focus points
    menu_ind = {}
    cnt = 0
    for m in sorted_menu:
        menu_ind[m[0]] = cnt
        cnt += 1
    # making the segments visible according to selected focus points can be simpler
    # but here we go for the different colors for each selected lines, therefore,
    # the code will be longer and more complex
    multi_select_callback = CustomJS(args=dict(rays=ray, circles=circle, colors=colors, ind=menu_ind), code="""
        var color_count = 0;
        if (cb_obj.value.length > 0) {
            for (var i=0; i<cb_obj.value.length; i++) {
                for (r in rays) {
                    for (var j=0; j<rays[r].length; j++) {
                        var diffFp = cb_obj.value[i].localeCompare(rays[r][j].data_source.data['fp'][0]);
                        var isActive = cb_obj.value.includes(rays[r][j].data_source.data['fp'][0]);

                        if (isActive) {
                            rays[r][j].visible = true;
                            if (!diffFp) {
                                rays[r][j].glyph.line_color = colors[color_count%12];
                                rays[r][j].glyph.line_alpha = 0.8;

                                circles[r][j].visible = true;
                                circles[r][j].glyph.fill_color = colors[color_count%12];
                                circles[r][j].glyph.fill_alpha = 0.8;
                            }
                        } else {
                            rays[r][j].visible = false;
                            rays[r][j].glyph.line_color = 'black';
                            rays[r][j].glyph.line_alpha = 0.1;

                            circles[r][j].visible = false;
                            circles[r][j].glyph.fill_color = 'black';
                            circles[r][j].glyph.fill_alpha = 0.1;
                        }
                    }
                }
                color_count++;
            }
        } else {
            for (r in rays) {
                for (var j=0; j<rays[r].length; j++) {
                    rays[r][j].visible = false;
                    rays[r][j].glyph.line_color = 'black';
                    rays[r][j].glyph.line_alpha = 0.1;

                    circles[r][j].visible = false;
                    circles[r][j].glyph.fill_color = 'black';
                    circles[r][j].glyph.fill_alpha = 0.1;
                }
            }
        }
    """)
    multi_select = MultiSelect(title="Select Focus Points:", value=[], options=sorted_menu, disabled=True, size=25)
    multi_select.js_on_change('value', multi_select_callback)
    toggle_callback = CustomJS(args=dict(plots=ray, check=multi_select), code="""
        if (cb_obj.active == true) {
            check.disabled = false;
            cb_obj.label = "Local View Mode";
            for (p in plots) {
                for (var i=0; i<plots[p].length; i++) {
                    plots[p][i].visible = false;
                }
            }
        } else {
            check.disabled = true;
            check.value = [];
            cb_obj.label = "Global View Mode";
            for (p in plots) {
                for (var i=0; i<plots[p].length; i++) {
                    plots[p][i].visible = true;
                }
            }
        }
    """)
    toggle = Toggle(label="Global View Mode", button_type="success", callback=toggle_callback)

    # set up the grid
    trace_grid = []
    for j in range(calc_data.dim - 1, 0, -1):
        trace_row = []
        for i in range(0, calc_data.dim - 1):
            if i != j:
                trace_row.append(trace[str((i, j))])
                if j == calc_data.dim - 1:
                    trace[str((i, j))].add_layout(Title(text=("x" + str(i + 1)), align="center"), "above")
                if i == 0:
                    trace[str((i, j))].add_layout(Title(text=("x" + str(j + 1)), align="center"), "left")
        trace_grid.append(trace_row)

    grid = gridplot(trace_grid)

    if width > 0:
        for i in range(0, calc_data.dim - 1):
            for j in range(i, calc_data.dim):
                trace[str((i, j))].plot_width = width

    if height > 0:
        for i in range(0, calc_data.dim - 1):
            for j in range(i, calc_data.dim):
                trace[str((i, j))].plot_height = height

    if output:
        output_file(output)
    save(row(grid, widgetbox(toggle, multi_select)), title=title if title else 'Hypersliceplorer')
