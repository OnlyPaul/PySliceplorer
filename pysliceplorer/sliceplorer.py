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
def sliceplorer(f, mn, mx, dim, n_fpoint, n_seg=100, method='sobol'):
    if mx <= mn:
        raise Exception('Input min exceeds max value. (Error: min >= max)')

    if n_fpoint <= 0:
        raise Exception('Program requires at least 1 focus point.')

    if dim < 1:
        raise Exception('Sliceplorer does not support less than 1 dimension. (Error: dim < 1)')

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
